import time
import cv2
import numpy as np
import os
import json
import subprocess
from PIL import Image
from typing import List, Dict, Tuple
import yaml
from skimage.measure import shannon_entropy
import torch
from xml.etree import ElementTree as ET
import glob
import re
import shutil
import pydiffvg
import skimage.io
from typing import List
import random
from pathlib import Path
from datetime import datetime
from utils.file_utils import load_image, batch_input, get_png_name_without_suffix, get_all_png_images, delete_files, \
    ensure_clean_dir, read_config, json_save_info,mask_to_svg
from utils.init_utils import get_adaptive_mask_crop, compute_average_rgba_per_mask, adaptive_close_mask, \
    suggest_opttolerance_from_complexity, \
    suggest_alphamax_from_complexity, compute_contour_complexity, \
    restore_mask_to_original_size, resize_for_segmentation
from utils import file_utils, data_utils
from segment_anything import sam_model_registry_baseline,sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import sys
import os
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize

# === 自动添加 UnSAM 的路径 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
unsam_path = os.path.abspath(os.path.join(current_dir, "/home/wenhuaszhgc/users/ngx/SAM/segment-anything/UnSAM-main/whole_image_segmentation"))
if unsam_path not in sys.path:
    sys.path.insert(0, unsam_path)
# print("✅ UnSAM path:", unsam_path)
from detectron2.engine import DefaultPredictor, default_setup
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from mask2former import add_maskformer2_config

from scipy import ndimage





def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    保留二值 mask 中最大的连通域，其余置 0。

    Args:
        mask (np.ndarray): 二值 mask，dtype=np.uint8 或 bool
    Returns:
        np.ndarray: 保留最大连通域后的二值 mask (uint8, 0/1)
    """
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        # 没有连通域
        return mask_uint8

    # 找到最大连通域（忽略背景 0）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8)
    return largest_mask


def run_unsam_on_blank_region(
        unsam_mask_generator,
        image_rgb: np.ndarray,
        sam_masks: list,
        full_foreground_mask: np.ndarray = None  # 新增参数，全局前景 mask
):
    """
    在 SAM 未分割的空白区域上执行 UnSAM 全图分割，只在前景区域操作。
    """
    H, W, _ = image_rgb.shape

    # === 1️⃣ 合并 SAM 掩码 ===
    sam_cover = np.zeros((H, W), dtype=bool)
    for m in sam_masks:
        sam_cover |= m.astype(bool)

    # === 2️⃣ 提取空白区域 ===
    blank_region = ~sam_cover

    if full_foreground_mask is not None:
        blank_region = blank_region.astype(bool) & full_foreground_mask  # 仅前景区域

    # 2.1 形态学闭运算
    kernel_size = max(3, int(min(H, W) * 0.002))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blank_region_closed = cv2.morphologyEx(blank_region.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 2.2 填充小孔
    blank_region_filled = ndimage.binary_fill_holes(blank_region_closed).astype(bool)

    # 2.3 小面积区域过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blank_region_filled.astype(np.uint8))
    min_area = int(0.0005 * H * W)
    refined_blank = np.zeros_like(blank_region_filled)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            refined_blank[labels == i] = True
    blank_region = refined_blank

    # === 3️⃣ 空白比例过小则跳过 ===
    if blank_region.sum() / (H * W) < 0.01:
        return []

    # === 4️⃣ 构造空白图像 ===
    blank_image = image_rgb.copy()
    blank_image[~blank_region] = 0

    # === 5️⃣ 执行 UnSAM 分割 ===
    try:
        unsam_results = unsam_mask_generator.generate(blank_image)
        unsam_results = add_source(unsam_results, "unsam")
    except Exception:
        return []

    # === 6️⃣ 保留空白区域内的掩码 ===
    new_masks = []
    for m in unsam_results:
        mask = m["segmentation"].astype(bool)&blank_region
        if full_foreground_mask is not None:
            mask = mask.astype(bool)&full_foreground_mask
        if mask.any() and mask.sum() / (H * W) >= 0.01:
            new_masks.append({"segmentation": mask, "area": mask.sum()})

    return new_masks


def process_masks_with_unsam_adaptive(image_rgb, sam_masks: list, unsam_mask_generator,
                                      output_json_path, sam, bg_seg=False,
                                      area_ratio_threshold=0.01, area_pixel_threshold=4 * 4,
                                      min_complexity_threshold=0.4,
                                      full_foreground_mask: np.ndarray = None):
    """
    对 SAM 输出的 mask 进行面积+自适应复杂度分析，并必要时进行二次分割。
    只在前景区域操作。
    """
    H, W = image_rgb.shape[:2]
    all_masks = []
    unsam_masks = []

    # Step 0: 如果没提供全局 mask，默认全部前景
    if full_foreground_mask is None:
        full_foreground_mask = np.ones((H, W), dtype=bool)

    # Step 1: 计算 SAM mask 复杂度
    for m in sam_masks:
        m['segmentation'] = m['segmentation'].astype(bool) & full_foreground_mask
        m['complexity'] = analyze_mask_complexity(image_rgb, m['segmentation'])
        m['is_secondary'] = False
        m['parent_idx'] = None

    # 自适应阈值
    non_large_complexities = [m['complexity'] for m in sam_masks
                              if m['area'] <= max(area_ratio_threshold * H * W, area_pixel_threshold)]
    adaptive_threshold = np.mean(non_large_complexities) if non_large_complexities else min_complexity_threshold
    # adaptive_threshold = max(adaptive_threshold, min_complexity_threshold)
    adaptive_threshold = min_complexity_threshold

    secondary_count = 0
    parent_count = len(sam_masks)

    # Step 2: 遍历父 SAM mask
    for idx, m in enumerate(sam_masks):
        all_masks.append(m)
        mask_area = m['area']
        complexity = m['complexity']

        if mask_area > max(area_ratio_threshold * H * W, area_pixel_threshold) and complexity > adaptive_threshold:
            patch_img, patch_mask, bbox = get_adaptive_mask_crop(image_rgb, m['segmentation'])
            if patch_img is not None:
                x_min, y_min, x_max, y_max = bbox

                # 扩张 bbox
                pad = max(3, int(min(H, W) * 0.005))

                x_min_p = max(0, x_min - pad)
                y_min_p = max(0, y_min - pad)
                x_max_p = min(W, x_max + pad)
                y_max_p = min(H, y_max + pad)

                # 扩张后的裁剪图
                patch_img_exp = image_rgb[y_min_p:y_max_p, x_min_p:x_max_p]

                # Run UNSAM or SAM on expanded crop
                if unsam_mask_generator is None:
                    sam_mask_generator = create_mask_generator(sam, patch_img_exp, 1)
                    new_masks = run_global_segmentation(sam_mask_generator, patch_img_exp)
                    new_masks = add_source(new_masks, "sam")
                else:
                    new_masks = run_global_segmentation(unsam_mask_generator, patch_img_exp)
                    new_masks = add_source(new_masks, "unsam")

                for u_mask_dict in new_masks:
                    u_mask = u_mask_dict['segmentation'].astype(np.uint8)
                    mask_h, mask_w = u_mask.shape

                    # 创建全图 mask
                    full_mask = np.zeros((H, W), dtype=np.uint8)

                    # --- 安全裁剪，确保不越界 ---
                    h_end = min(y_min_p + mask_h, H)
                    w_end = min(x_min_p + mask_w, W)

                    # u_mask 对应裁剪（避免越界）
                    u_mask_crop = u_mask[:h_end - y_min_p, :w_end - x_min_p]

                    # 贴回全图
                    full_mask[y_min_p:h_end, x_min_p:w_end] = u_mask_crop

                    # 父 mask 约束
                    m_seg = m['segmentation'].astype(bool)
                    full_mask &= m_seg
                    full_mask = keep_largest_connected_component(full_mask)

                    sec_mask = {
                        'segmentation': full_mask.astype(bool),
                        'area': np.count_nonzero(full_mask),
                        'score': u_mask_dict.get('score', 1.0),
                        'complexity': analyze_mask_complexity(image_rgb, full_mask),
                        'is_secondary': True,
                        'parent_idx': idx
                    }
                    all_masks.append(sec_mask)
                    unsam_masks.append(sec_mask)
                    secondary_count += 1

    # Step 3: 空白区域掩码
    if bg_seg:
        unsam_masks_bg = run_unsam_on_blank_region(unsam_mask_generator, image_rgb,
                                                   [m['segmentation'] for m in sam_masks],
                                                   full_foreground_mask)
    else:
        unsam_masks_bg = []

    blank_count = len(unsam_masks_bg)
    for u_mask_dict in unsam_masks_bg:
        mask = u_mask_dict['segmentation'].astype(bool)
        all_masks.append({
            'segmentation': mask,
            'area': np.count_nonzero(mask),
            'score': u_mask_dict.get('score', 1.0),
            'complexity': analyze_mask_complexity(image_rgb, mask),
            'is_secondary': True,
            'parent_idx': None
        })

    # Step 4: 保存 JSON
    mask_info = []
    for idx, m in enumerate(all_masks):
        mask_info.append({
            'mask_idx': idx,
            'area': m['area'],
            'complexity': m['complexity'],
            'is_secondary': m['is_secondary'],
            'parent_idx': m['parent_idx']
        })
    if output_json_path is not None:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump({
                'mask_info': mask_info,
                'counts': {'parent': parent_count, 'secondary': secondary_count, 'blank': blank_count}
            }, f, indent=2)

    return all_masks, parent_count, unsam_masks, unsam_masks_bg

def segment_with_alpha_mask(
        image_rgb,
        alpha_mask,
        sam=None,
        unsam_mask_generator=None,
        c=None,
        occlusion_threshold=0.9       # ← 新增：遮挡比例阈值（90%）
):
    """
    输入：
        image_rgb           (H,W,3) uint8
        alpha_mask          (H,W) uint8 or bool，1=前景
        occlusion_threshold float，原 mask 被遮挡超过多少比例就丢弃
    """

    H, W = image_rgb.shape[:2]
    alpha_bool = alpha_mask.astype(bool)

    # ========= 1. 全图分割 =========
    if unsam_mask_generator is None:
        mask_generator = create_mask_generator(sam, image_rgb, c)
        raw_masks = run_global_segmentation(mask_generator, image_rgb)
        raw_masks = add_source(raw_masks, "sam")
    else:
        raw_masks = run_global_segmentation(unsam_mask_generator, image_rgb)
        raw_masks = add_source(raw_masks, "unsam")

    # ========= 2. 与 alpha mask 求交 + 过滤遮挡严重的 =========
    all_masks = []
    for m in raw_masks:

        raw_seg = m["segmentation"].astype(bool)
        raw_area = int(raw_seg.sum())

        if raw_area == 0:
            continue

        # ---- 求交集 ----
        intersect_seg = raw_seg & alpha_bool
        intersect_area = int(intersect_seg.sum())

        if intersect_area == 0:
            continue

        # ---- 计算遮挡比例 ----
        remain_ratio = intersect_area / raw_area     # 剩余面积比例
        occluded_ratio = 1 - remain_ratio            # 遮挡比例

        # ---- 遮挡90%以上 → 丢弃 ----
        if occluded_ratio >= occlusion_threshold:
            # print(f"丢弃 mask：遮挡比例 {occluded_ratio:.2f}")
            continue

        # ---- 保留 ----
        all_masks.append({
            "segmentation": intersect_seg,
            "area": intersect_area,
            "score": float(m.get("score", 1.0)),
            "source": m.get("source", "unknown"),
            "raw_area": raw_area,
            "remain_ratio": remain_ratio
        })

    return all_masks


def analyze_mask_complexity(image_rgb, mask=None):
    """
    分析图像或掩码区域的复杂度:
    - 如果 mask=None → 分析整图复杂度
    - 如果 mask 非空 → 仅分析 mask 内部区域复杂度
    返回值: 复杂度评分 ∈ [0,1]
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if mask is not None:
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return 0.0
        # 仅保留 mask 区域
        gray_masked = gray.copy()
        gray_masked[~mask] = 0
        edges = cv2.Canny(gray_masked, 100, 200)
        edge_density = (edges[mask].mean() / 255.0)
        color_var = np.var(image_rgb[mask] / 255.0)
        entropy = shannon_entropy(gray[mask])
    else:
        # 整图分析
        edges = cv2.Canny(gray, 100, 200)
        edge_density = edges.mean() / 255.0
        color_var = np.var(image_rgb / 255.0)
        entropy = shannon_entropy(gray)

    # 复杂度评分归一化
    score = 0.5 * edge_density + 0.3 * (color_var / 0.1) + 0.2 * (entropy / 8.0)
    return float(np.clip(score, 0, 1))


def adaptive_sam_params(image_rgb, c):
    H, W = image_rgb.shape[:2]
    area = H * W
    if c is None:
        c = analyze_mask_complexity(image_rgb)



    if c < 0.2:
        points_per_side=32
        pred_iou_thresh = 0.88
        stability_score_thresh = 0.90
        area_ratio = 1e-4
    elif c < 0.4:
        points_per_side=64

        pred_iou_thresh = 0.85
        stability_score_thresh = 0.88
        area_ratio = 1e-4
    else:
        points_per_side=128
        pred_iou_thresh = 0.80
        stability_score_thresh = 0.86
        area_ratio = 1e-4

    # 更严格的最小区域过滤
    min_mask_region_area = int(area * area_ratio)
    min_mask_region_area = max(min_mask_region_area, 4)  # 不要太小

    # # # 新增去重参数（可在 sam 生成器中使用）
    # sam_extra_params = dict(
    #     box_nms_thresh=0.7,
    #     crop_n_layers=2,
    #     crop_overlap_ratio=0.6,  # 降低重叠率
    #     crop_n_points_downscale_factor=2,
    # )

    return dict(
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        # **sam_extra_params
    )


# === 加载 UnSAM 模型 ===
def load_unsam_model(config_file: str, weight_path: str, device: str,obj_num: int) -> torch.nn.Module:
    """
    加载 UnSAM 模型 (Mask2Former 架构)
    """

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weight_path
    import os
    if device != "cpu":
        device_id = device.split(":")[-1]
        # print(device_id)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)  # 显示控制可见GPU
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"
    if hasattr(cfg, "TEST"):
        if hasattr(cfg.TEST, "DETECTIONS_PER_IMAGE"):
            cfg.TEST.DETECTIONS_PER_IMAGE = obj_num*2
        else:
            print("ERR NO attr")
    if hasattr(cfg, "MODEL"):
        if hasattr(cfg.MODEL, "MASK_FORMER"):
            cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = obj_num*4
        else:
            print("ERR NO attr")
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return predictor


def run_unsam_segmentation(predictor, image_rgb: np.ndarray, conf_thresh: float = 0.1):
    """
    用 UnSAM / Mask2Former 预测整图 mask，返回 dict 列表
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # HWC -> CHW
    img_tensor = torch.from_numpy(image_rgb).to(device)
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()
    img_tensor = img_tensor.byte()

    batched_inputs = [{"image": img_tensor}]

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = predictor.model(batched_inputs)[0]

    instances = outputs["instances"].to("cpu")
    pred_masks = instances.pred_masks.numpy()  # [N,H,W]
    scores = instances.scores.detach().cpu().numpy()  # [N]

    masks = []
    for i in range(len(pred_masks)):
        if scores[i] < conf_thresh:
            continue
        mask_bool = pred_masks[i].astype(bool)
        masks.append({
            "segmentation": mask_bool,
            "area": int(mask_bool.sum()),
            "score": float(scores[i])
        })
    return masks

# === 替代 SAM 的分割器构建（保持接口一致） ===
def create_unsam_mask_generator(model):
    """
    模拟 SAM 的 mask_generator 形式，使旧代码兼容
    """

    class UnSAMWrapper:
        def __init__(self, predictor):
            self.predictor = predictor

        def generate(self, image_rgb):
            return run_unsam_segmentation(self.predictor, image_rgb)

    return UnSAMWrapper(model)


# === 加载 SAM 模型 ===
def load_sam_model(model_type: str, checkpoint_path: str, device: str, flag=False):
    print("Loading SAM model...")
    if flag:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    else:
        sam = sam_model_registry_baseline[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam


# === 创建分割器 ===
def create_mask_generator(model, image_rgb, c=None, flag=False):
    if flag:
        params = adaptive_sam_params(image_rgb, c)
    else:
        # 默认 SAMAutomaticMaskGenerator 参数
        H, W = image_rgb.shape[:2]
        area = H * W
        min_mask_region_area = int(area * 1e-4)
        min_mask_region_area = max(min_mask_region_area, 4)  # 不要太小
        params = {
            "points_per_side": 128,  # 每边采样点
            "points_per_batch": 64,  # 每批次处理点
            "pred_iou_thresh": 0.8,  # mask 置信度阈值
            "stability_score_thresh": 0.9,  # mask 稳定性阈值
            "min_mask_region_area": min_mask_region_area,  # 最小 mask 面积（像素）
        }

    return SamAutomaticMaskGenerator(
        model=model,
        output_mode="binary_mask",
        **params
    )




def split_connected_components_with_properties(mask_dict, original_h, original_w):
    """
    输入：SAM 的单个 mask_dict（包含 segmentation 等属性）
    输出：拆分后的多个 mask_dict，每个只包含一个连通域
    """

    seg = mask_dict["segmentation"].astype(np.uint8)

    # 连通域
    num_labels, labels = cv2.connectedComponents(seg)

    out_masks = []

    for label in range(1, num_labels):  # 0 为背景
        cc_mask = (labels == label).astype(np.uint8)

        if cc_mask.sum() == 0:
            continue

        # 计算 bbox
        ys, xs = np.where(cc_mask > 0)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # 复制属性并更新
        new_mask = dict(mask_dict)  # 浅拷贝属性
        new_mask["segmentation"] = cc_mask.astype(bool)
        new_mask["area"] = int(cc_mask.sum())
        new_mask["bbox"] = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]

        out_masks.append(new_mask)

    return out_masks



def run_global_segmentation(mask_generator, image_rgb: np.ndarray):
    """
    自适应缩放策略：
    1. 若短边 < 512：等比例放大，使短边 = 512
    2. 若长边 > 1024：等比例缩小，使长边 = 1024
    3. 否则：保持原尺寸
    4. SAM 全图分割（在缩放后的图上）
    5. segmentation mask 映射回原图尺寸
    6. 对每个 mask 做连通域拆分
    """

    H, W = image_rgb.shape[:2]
    orig_H, orig_W = H, W

    short_side = min(H, W)
    long_side = max(H, W)

    target_short = 512
    target_long = 1024

    # ------------------------------
    # 1) 根据规则计算缩放比例
    # ------------------------------
    scale = 1.0

    # 短边太小：放大
    if short_side < target_short:
        scale = target_short / short_side

    # 长边太大：缩小
    if long_side * scale > target_long:
        scale = target_long / long_side

    # 如果 scale==1 表示无需缩放
    if scale != 1.0:
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))
        img_small = cv2.resize(image_rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    else:
        img_small = image_rgb

    # ------------------------------
    # 2) SAM/HQ-SAM 全图分割
    # ------------------------------
    masks_small = mask_generator.generate(img_small)

    all_masks = []

    # ------------------------------
    # 3) 映射 mask 回原图尺寸 + 连通域分离
    # ------------------------------
    for m in masks_small:

        # segmentation 是 bool mask
        seg_small = m["segmentation"].astype(np.uint8)

        if scale != 1.0:
            # 映射回原图尺寸
            seg_large = cv2.resize(seg_small, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
        else:
            seg_large = seg_small

        # 更新 mask 字典
        m_large = dict(m)
        m_large["segmentation"] = seg_large.astype(bool)

        # 连通域拆分
        cc_masks = split_connected_components_with_properties(m_large, orig_H, orig_W)

        all_masks.extend(cc_masks)

    return all_masks


def remove_duplicate_masks(
        masks: List[Dict],
        image_shape: Tuple[int, int],
        min_area: int = 16,
        min_area_ratio: float = 1 / 10,
        min_area_whole_ratio: float = 1 / 1000,
        iou_threshold: float = 0.7,
        flag=False
):
    """
    综合去重策略（增强版）：
      (1) 先进行面积过滤
      (2) 两两 IoU 去重（关键增强）
      (3) 遮挡可见面积过滤
      (4) 递增区域覆盖

    返回：按原输入顺序保留的 masks 列表
    """

    H, W = image_shape
    total_pixels = H * W
    n = len(masks)

    masks_bin = [m['segmentation'].astype(bool) for m in masks]
    areas = [int(m['area']) if 'area' in m else int(np.count_nonzero(masks_bin[i])) for i, m in enumerate(masks)]

    # ----------------------------------------------------------------------
    # Step 1：面积过滤（小于最小面积 + 整图比例）
    # ----------------------------------------------------------------------
    keep = [True] * n
    for i in range(n):
        if areas[i] < min_area and areas[i] / total_pixels < min_area_whole_ratio:
            keep[i] = False
        # if masks[i].get("source", "") == "sam" and is_fake_sam_mask(masks[i]):
        #     keep[i] = False

    # ----------------------------------------------------------------------
    # Step 2：两两 IoU 强力去重（保留面积大的）
    # ----------------------------------------------------------------------
    for i in range(n):
        if not keep[i]:
            continue
        mi = masks_bin[i]
        source_i = masks[i].get("source", "")

        for j in range(i + 1, n):
            if not keep[j]:
                continue
            mj = masks_bin[j]
            source_j = masks[j].get("source", "")

            inter = np.logical_and(mi, mj).sum()
            union = np.logical_or(mi, mj).sum()
            iou = inter / (union + 1e-9)

            if iou > iou_threshold:
                if source_i != source_j:
                    if source_i == "sam":
                        keep[j] = False
                    elif source_j == "sam":
                        keep[i] = False
                        break
                    continue
                # 保留更大的 mask
                if areas[i] >= areas[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i 被剔除，不需要继续比较 j

    # ----------------------------------------------------------------------
    # Step 3：遮挡后的可见面积过滤
    # ----------------------------------------------------------------------
    # 重要：按面积从小到大排序（小的优先保留）
    sorted_idx = sorted([i for i in range(n) if keep[i]], key=lambda x: areas[x])
    covered = np.zeros((H, W), dtype=bool)

    for i in sorted_idx:
        if not keep[i]:
            continue

        mask_i = masks_bin[i]

        # 多少是未被覆盖的可见区域？
        visible = np.logical_and(mask_i, ~covered)
        visible_area = visible.sum()

        if visible_area < min_area or (visible_area / (areas[i] + 1e-9)) < min_area_ratio:
            keep[i] = False
            continue

        # 加入覆盖区域
        covered |= mask_i

    # ----------------------------------------------------------------------
    # Step 4：按原顺序输出
    # ----------------------------------------------------------------------
    kept_masks = [m for m, k in zip(masks, keep) if k]
    kept_sam_masks = []
    kept_unsam_masks = []

    for m, k in zip(masks, keep):
        if k:
            if m.get("source", "") == "sam":
                kept_sam_masks.append(m)
            else:
                kept_unsam_masks.append(m)
    if flag:
        return kept_masks, kept_sam_masks, kept_unsam_masks
    else:
        return kept_masks


# === 按面积排序 ===
def sort_masks_by_area(masks: List[Dict]):
    if len(masks) > 0:
        return sorted(masks, key=lambda x: x["area"], reverse=True)
    else:
        return []


# === 可视化：将所有掩码叠加在原图上 ===
def visualize_masks(image: np.ndarray, masks: List[Dict], output_path_vis: str, output_path_masks_only: str,
                    output_path_mask, flag: bool):
    vis_image = image.copy()
    colors = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)

    # 创建一张全黑的掩码合成图（背景为0）
    masks_only_image = np.zeros_like(image, dtype=np.uint8)

    # 构造RGBA纯白图像

    white_rgba = np.ones_like(image, dtype=np.uint8) * 255  # 全白，4通道

    for i, m in enumerate(masks):
        mask = m['segmentation'].astype(bool)

        # 原图上叠加半透明颜色
        vis_image[mask] = (0.5 * vis_image[mask] + 0.5 * colors[i]).astype(np.uint8)

        # 纯掩码图上叠加颜色（完全覆盖，方便看出区域）
        masks_only_image[mask] = colors[i]
    if flag:
        # 保存可视化结果（原图+半透明掩码）
        cv2.imwrite(output_path_vis, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    # 保存掩码合成图（纯色叠加）
    cv2.imwrite(output_path_masks_only, cv2.cvtColor(masks_only_image, cv2.COLOR_RGB2BGR))
    if output_path_mask is not None:
        # 有效掩码区域（默认就全部了）
        cv2.imwrite(output_path_mask, cv2.cvtColor(white_rgba, cv2.COLOR_RGB2BGR))


# === 保存掩码 ===
def simplify_binary_mask(
    mask: np.ndarray,
    epsilon_ratio: float = 0.01
):
    """
    对 binary mask 的轮廓进行 RDP 简化

    epsilon_ratio: 相对于轮廓周长的比例（推荐 0.005 ~ 0.02）
    """
    import cv2
    import numpy as np

    h, w = mask.shape
    simplified = np.zeros_like(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    for cnt in contours:
        if len(cnt) < 10:
            continue

        # 轮廓周长
        perimeter = cv2.arcLength(cnt, closed=True)
        epsilon = epsilon_ratio * perimeter

        approx = cv2.approxPolyDP(
            cnt,
            epsilon=epsilon,
            closed=True
        )

        cv2.fillPoly(simplified, [approx], 1)

    return simplified

def save_masks(masks: List[Dict], output_dir: str, base_size=256, simplify_eps_ratio=0):
    for i, m in enumerate(masks):
        mask = m['segmentation'].astype(np.uint8)   # 消除细小断裂
        mask = adaptive_close_mask(mask, base_size)
        if simplify_eps_ratio != 0:
            mask = simplify_binary_mask(
                mask,
                epsilon_ratio=simplify_eps_ratio
            )
        mask = 1-mask
        mask = mask * 255

        cv2.imwrite(f"{output_dir}/mask_{i}.png", mask)


def save_masked_regions(image: np.ndarray,
                        masks: List[Dict],
                        output_dir: str,
                        base_size: int = 256,
                        input_color_format: str = "rgb"):
    """
    保存每个掩码对应的原图区域（按 mask 裁剪，mask 外为透明）。
    保证颜色正确：使用 PIL 以 RGBA 保存，适配输入是 'rgb' 或 'bgr'。

    Args:
        image: 原始图像 numpy array, HxWx3. (默认 RGB, 若是 OpenCV BGR 请传 input_color_format='bgr')
        masks: List[Dict], 每个 dict 包含 'segmentation' 二值掩码（bool 或 0/1/255）。
        output_dir: 输出文件夹路径，会创建。
        base_size: 只用于 adaptive_close_mask 的尺寸参数以保持接口一致。
        input_color_format: "rgb" 或 "bgr"，表示传入图像的通道顺序。
    Returns:
        List[str]: 每个保存文件的路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]

    # 统一把输入转为 RGB numpy (uint8)
    img_rgb = image.copy()
    if img_rgb.dtype != np.uint8:
        # 可能是 float，归一化到 0-255
        img_rgb = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8) if img_rgb.max() <= 1.0 else img_rgb.astype(np.uint8)

    if input_color_format.lower() == "bgr":
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    saved_paths = []
    for i, m in enumerate(masks):
        mask = m.get('segmentation')
        if mask is None:
            continue

        # 保证 mask 为 HxW 二值 (0/1)
        mask_arr = np.array(mask)
        if mask_arr.dtype != np.uint8:
            mask_arr = mask_arr.astype(np.uint8)
        # 统一到 0/1
        mask_bin = (mask_arr > 0).astype(np.uint8)

        # 如果 mask 尺寸不一致则最近邻插值调整
        if mask_bin.shape != (h, w):
            mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        # 对 mask 做闭运算补断裂 (adaptive_close_mask 期待 0/255 input)
        mask_closed = adaptive_close_mask(mask_bin * 255, base_size)
        mask_closed = (mask_closed > 127).astype(np.uint8)  # 0/1

        # 用 mask 作为 alpha 通道：alpha 0-255
        alpha = (mask_closed * 255).astype(np.uint8)

        # 构造 RGBA 数据 (PIL expects RGB order)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = img_rgb  # 已是 RGB
        rgba[..., 3] = alpha

        # 使用 PIL 保存，保持颜色不变
        out_path = os.path.join(output_dir, f"region_{i}.png")
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        saved_paths.append(out_path)

    return saved_paths


def split_connected_components(sam_masks: List[Dict], min_area: int = 1) -> List[Dict]:
    """
    将 SAM mask 中的连通域分开，生成独立 mask。

    Args:
        sam_masks (List[Dict]): 原始 SAM mask 列表，每个元素至少包含 'segmentation' (bool 或 uint8 mask)
        min_area (int): 最小面积阈值，过滤过小连通域

    Returns:
        List[Dict]: 新的 mask 列表，格式与 SAM 输出一致
    """
    new_masks = []

    for m in sam_masks:
        mask = m['segmentation'].astype(np.uint8)
        # 连通域标记
        num_labels, labels = cv2.connectedComponents(mask)
        for label_idx in range(1, num_labels):  # 0 是背景
            comp_mask = (labels == label_idx).astype(np.uint8)
            area = int(comp_mask.sum())
            if area < min_area:
                continue
            new_mask = {
                'segmentation': comp_mask,
                'area': area,
                'complexity': m.get('complexity', None),
                'score': m.get('score', 1.0),
                'is_secondary': m.get('is_secondary', False),
                'parent_idx': m.get('parent_idx', None)
            }
            new_masks.append(new_mask)

    return new_masks


def add_source(masks, who):
    for m in masks:
        m["source"] = f"{who}"
    return masks


# === seg主函数 ===
def structure_seg(cfg):
    if cfg is not None:
        config = cfg
    else:
        raise ValueError("config is required!")
    output_top_dir = config["basic"]["output_dir"]
    base_size = config["basic"]["base_size"]
    input_dir = config["basic"]["input_image"]
    double_seg_enable = config["basic"]["double_seg"]
    color_type = config["optimization"]["color_type"]
    input_batch_dir = config["basic"].get("input_images_dir")
    simplify_eps_ratio = config["vtracer"]["simplify_eps_ratio"]
    if config["basic"]["use_hq_sam"]:
        sam_type = config["hqsam"]["model_type"]
        pth = config["hqsam"]["checkpoint_path"]
    else:
        sam_type = config["sam"]["model_type"]
        pth = config["sam"]["checkpoint_path"]
    json_info, json_file_name = file_utils.read_top_json(config, True)

    if config["basic"]["shape_num"] is None:
        # free 不需要补全
        config["basic"]["detail_ratio"] = 0
        # 最大掩码数量为2048
        if config["sam"]["mask_num"] is not None:
            mask_num = config["sam"]["mask_num"]
        else:
            mask_num = 2048
    else:
        # 设置具体数量可以设置细节丰富度 正值一定会有细节
        if config["basic"]["detail_ratio"] is None:
            # 默认0.5
            ratio = 0.5
            mask_num = round(config["basic"]["shape_num"] * (1 - ratio))
        else:
            # 如果小于等于0则没有细节限制
            if config["basic"]["detail_ratio"] <= 0:
                mask_num = config["basic"]["shape_num"]
            else:
                ratio = config["basic"]["detail_ratio"]
                mask_num = round(config["basic"]["shape_num"] * (1 - ratio))
    images_path = batch_input(input_batch_dir, input_dir)
    colors_list = []
    image_names = []
# if not (os.path.exists(f"{output_top_dir}/structure") \
#         and os.path.isfile(f"{output_top_dir}/structure/structure_init.svg")):
    # 跳过此阶段 直接读取mask 颜色 json

    os.makedirs(output_top_dir, exist_ok=True)

    # print(f"!!!{mask_num} mask num!!!")
    sam_load_time_s = time.time()
    if not double_seg_enable:
        if not config["unsam"]["use"]:
            sam = load_sam_model(sam_type, pth, config["basic"]["device"])

        else:
            # print(config["basic"]["device"])
            unsam = load_unsam_model(
                config_file=config["unsam"]["config_file"],
                weight_path=config["unsam"]["weight_path"],
                device=config["basic"]["device"]
            )

    else:
        sam = load_sam_model(sam_type, pth, config["basic"]["device"], config["basic"]["use_hq_sam"])
        if config["double_seg"]["use_unsam_start"]:
            unsam_whole_img = load_unsam_model(
                config_file=config["unsam"]["config_file"],
                weight_path=config["unsam"]["weight_path"],
                device=config["basic"]["device"],
                obj_num=mask_num
            )
    sam_load_time_e = time.time()
    sam_load_time = sam_load_time_e - sam_load_time_s
    print(f"model loading finished, cost {sam_load_time} seconds.")
    json_info["structure"] = {}
    json_info["structure"]["sam_load_time"] = sam_load_time

    json_save = config["save"]["json"]
    # print(input_batch_dir)
    # print(input_dir)

    # print(images_path)

    total_time_s = time.time()
    total_initial_mask_len = 0
    total_unique_mask_len = 0
    total_color_init_time = 0
    total_sam_mask_len = 0
    total_unsam_mask_len = 0
    total_1_mask_len = 0
    total_2_mask_len = 0
    total_2_unsam_mask_len = 0
    total_2_unsam_bg_mask_len = 0

    for i, image_path in enumerate(images_path):
        structure_time_s = time.time()
        # 目标图像名称 用于生成文件夹名称
        image_name = get_png_name_without_suffix(image_path)
        image_names.append(image_name)
        now_json_file_name = f"{output_top_dir}/{image_name}/info.json"
        now_json = {
            "change_time": "",
            "image_name": image_name,
            "image_path": image_path,
            "bg": config["basic"]["bg"],
            "structure": {},
        }
        print(f"{i + 1}/{len(images_path)}: {image_name} under structure processing...")
        output_dir = f"{output_top_dir}/{image_name}/structure"
        os.makedirs(output_dir, exist_ok=True)
        image, image_rgb, a_mask = load_image(image_path, output_dir, config["basic"]["bg"],
                                              config["save"]["sam"]["sam_input"])
        # image_rgb = cv2.bilateralFilter(image_rgb, d=9, sigmaColor=75, sigmaSpace=75)

        final_masks = []
        # print(mask_num)
        if not double_seg_enable:
            if not config["unsam"]["use"]:
                initial_masks = segment_with_alpha_mask(image_rgb, a_mask, sam, None, None)

            else:
                mask_generator = create_unsam_mask_generator(unsam)
                initial_masks = segment_with_alpha_mask(image_rgb, a_mask, None,mask_generator, None)

            # initial_masks = run_global_segmentation(mask_generator, image_rgb)
            initial_masks = sort_masks_by_area(initial_masks)
            initial_masks_len = len(initial_masks)

            # 先去重
            if config["sam"]["iou"] is not None:
                iou = config["sam"]["iou"]
            else:
                iou = 0.7
            if config["unsam"]["use"]:
                unique_masks = remove_duplicate_masks(initial_masks, iou_threshold=iou,
                                                      image_shape=image_rgb.shape[:2])
            else:
                unique_masks = initial_masks
            # print(f"Remaining masks after deduplication: {len(unique_masks)}")
            unique_masks_len = len(unique_masks)

            # print(f"SAM effective mask number:{len(sorted_masks)}.")
            # 限制mask数量,按照面积的顺序

        else:
            seg_iou = config["double_seg"]["iou"]
            min_ratio = config["double_seg"]["min_ratio"]
            min_seg_ratio = config["double_seg"]["min_seg_ratio"]
            min_covered_ratio = config["double_seg"]["min_covered_ratio"]
            min_area = config["double_seg"]["min_area"]
            seg_complexity = config["double_seg"]["seg_complexity"]
            c = analyze_mask_complexity(image_rgb,a_mask)
            print(f"image c:{c}")

            # 一轮：自适应sam分割
            sam_seg_s = time.time()
            sam_masks = segment_with_alpha_mask(image_rgb,a_mask,sam,None,c)
            # sam_mask_generator = create_mask_generator(sam, image_rgb, c)
            # sam_masks = run_global_segmentation(sam_mask_generator, image_rgb)
            sam_masks = add_source(sam_masks, "sam")
            sam_masks_len = len(sam_masks)
            sam_seg_e = time.time()
            sam_seg_time = sam_seg_e - sam_seg_s

            print(
                f"{i + 1}/{len(images_path)}: {image_name} 1-sam-mask len:{sam_masks_len}, seg time:{sam_seg_time}.")
            if config["double_seg"]["use_unsam_start"] and (mask_num > sam_masks_len or mask_num >= 1024):

                # 一轮：unsam分割
                unsam_seg_s = time.time()
                unsam_whole_img_mask_generator = create_unsam_mask_generator(unsam_whole_img)
                unsam_masks = segment_with_alpha_mask(image_rgb, a_mask, None,unsam_whole_img_mask_generator, None)

                # unsam_masks = run_global_segmentation(unsam_mask_generator, image_rgb)
                unsam_masks = add_source(unsam_masks, "unsam")
                unsam_masks_len = len(unsam_masks)
                unsam_seg_e = time.time()
                unsam_seg_time = unsam_seg_e - unsam_seg_s
                print(
                    f"{i + 1}/{len(images_path)}: {image_name} 1-unsam-mask len:{unsam_masks_len}, seg time:{unsam_seg_time}.")
            else:
                unsam_masks = []
                unsam_masks_len = 0

            # 一轮：合并
            unsam_masks = sort_masks_by_area(unsam_masks)
            sam_masks = sort_masks_by_area(sam_masks)

            level_1_initial_masks = sam_masks + unsam_masks

            level_1_initial_masks = sort_masks_by_area(level_1_initial_masks)
            # 一轮：去重
            level_1_unique_masks, l1_sam_masks, l1_unsam_masks = remove_duplicate_masks(level_1_initial_masks,
                                                                                        image_shape=image_rgb.shape[
                                                                                                    :2],
                                                                                        iou_threshold=seg_iou,
                                                                                        min_area=min_area,
                                                                                        min_area_ratio=min_covered_ratio,
                                                                                        min_area_whole_ratio=min_ratio,
                                                                                        flag=True)
            level_1_unique_masks_len = len(level_1_unique_masks)
            level_1_initial_masks_len = len(level_1_initial_masks)
            print(
                f"{i + 1}/{len(images_path)}: {image_name} level 1 initial mask number: {level_1_initial_masks_len}, loss {level_1_initial_masks_len - level_1_unique_masks_len} masks, final mask num:{level_1_unique_masks_len}")

            seg_res = f"{output_dir}/double_seg"
            level2_sam_seg_e = time.time()
            if config["double_seg"]["second_sam"]:
                unsam_mask_generator = None
            else:
                unsam = load_unsam_model(
                    config_file=config["unsam"]["config_file"],
                    weight_path=config["unsam"]["weight_path"],
                    device=config["basic"]["device"],
                    obj_num=50
                )
                unsam_mask_generator = create_unsam_mask_generator(unsam)
            if level_1_unique_masks_len < mask_num:

                level2_initial_masks, level1_masks_len, level2_unsam_masks, level2_unsam_masks_bg = process_masks_with_unsam_adaptive(
                    image_rgb,
                    level_1_unique_masks, unsam_mask_generator, f"{seg_res}/info.json",
                    area_ratio_threshold=min_seg_ratio, area_pixel_threshold=min_area, full_foreground_mask=a_mask,
                    min_complexity_threshold=seg_complexity, sam=sam, bg_seg=config["double_seg"]["bg_seg"])
                level2_unsam_seg_time = time.time() - level2_sam_seg_e
                level2_unsam_masks_len = len(level2_unsam_masks)
                level2_unsam_masks_bg_len = len(level2_unsam_masks_bg)
                print(
                    f"{i + 1}/{len(images_path)}: {image_name} 2-unsam-mask len:{level2_unsam_masks_len} + {level2_unsam_masks_bg_len}, seg time:{level2_unsam_seg_time}.")
                level2_unsam_masks = sort_masks_by_area(level2_unsam_masks)
                level2_unsam_masks_bg = sort_masks_by_area(level2_unsam_masks_bg)
                level2_initial_masks = sort_masks_by_area(level2_initial_masks)

                level2_initial_masks_len = len(level2_initial_masks)
                # 去重
                level2_unique_masks, l2_sam_masks, l2_unsam_masks = remove_duplicate_masks(level2_initial_masks,
                                                                                           image_shape=image_rgb.shape[
                                                                                                       :2],
                                                                                           iou_threshold=seg_iou,
                                                                                           min_area=min_area,
                                                                                           min_area_ratio=min_covered_ratio,
                                                                                           min_area_whole_ratio=min_ratio,
                                                                                           flag=True)
                level2_unique_masks_len = len(level2_unique_masks)
                print(
                    f"{i + 1}/{len(images_path)}: {image_name} level 2 initial mask number: {level2_initial_masks_len}, loss {level2_initial_masks_len - level2_unique_masks_len} masks, final mask num:{level2_unique_masks_len}")
            else:
                level2_initial_masks = level_1_initial_masks
                level2_unique_masks = level_1_unique_masks
                level2_unsam_masks_len = 0
                level2_unsam_masks_bg_len = 0
                level2_unique_masks_len = 0
                level2_initial_masks_len = 0
                level2_unsam_masks = []
                level2_unsam_masks_bg = []
                l2_sam_masks = l1_sam_masks
                l2_unsam_masks = l1_unsam_masks
            initial_masks = level2_initial_masks
            initial_masks_len = len(initial_masks)
            unique_masks = level2_unique_masks
            unique_masks_len = len(unique_masks)

            now_json["structure"]["1_sam_masks_lem"] = sam_masks_len
            now_json["structure"]["1_unsam_masks_len"] = unsam_masks_len
            now_json["structure"]["1_seg"] = level_1_unique_masks_len
            now_json["structure"]["2_unsam_masks_len"] = level2_unsam_masks_len
            now_json["structure"]["2_unsam_masks_bg_len"] = level2_unsam_masks_bg_len
            now_json["structure"]["2_seg"] = level2_unique_masks_len

            total_sam_mask_len += sam_masks_len
            total_unsam_mask_len += unsam_masks_len
            total_1_mask_len += level_1_unique_masks_len
            total_2_mask_len += level2_unique_masks_len
            total_2_unsam_bg_mask_len += level2_unsam_masks_bg_len
            total_2_unsam_mask_len += level2_unsam_masks_len
            double_seg_save = config["save"]["double_seg"]

            if double_seg_save:

                os.makedirs(seg_res, exist_ok=True)

                visualize_masks(image_rgb, sam_masks, f"{seg_res}/sam_a.png", f"{seg_res}/sam.png", None, True)
                if unsam_masks_len > 0:
                    visualize_masks(image_rgb, unsam_masks, f"{seg_res}/unsam_a.png", f"{seg_res}/unsam.png", None,
                                    True)

                if level2_unsam_masks_len > 0:
                    visualize_masks(image_rgb, level2_unsam_masks, f"{seg_res}/unsam_2_a.png",
                                    f"{seg_res}/unsam_2.png",
                                    None,
                                    True)

                if level2_unsam_masks_bg_len > 0:
                    visualize_masks(image_rgb, level2_unsam_masks_bg, f"{seg_res}/unsam_2_bg_a.png",
                                    f"{seg_res}/unsam_2_bg.png",
                                    None, True)
                if level_1_unique_masks_len > 0:
                    visualize_masks(image_rgb, level_1_unique_masks, f"{seg_res}/1_a.png",
                                    f"{seg_res}/1.png", None, True)
                if level2_unique_masks_len > 0:
                    visualize_masks(image_rgb, level2_unique_masks, f"{seg_res}/2_a.png",
                                    f"{seg_res}/2.png", None, True)
                if level_1_initial_masks_len > 0:
                    visualize_masks(image_rgb, level_1_initial_masks, f"{seg_res}/1_init_a.png",
                                    f"{seg_res}/1_init.png", None, True)
                if level2_initial_masks_len > 0:
                    visualize_masks(image_rgb, level2_initial_masks, f"{seg_res}/2_init_a.png",
                                    f"{seg_res}/2_init.png", None, True)
            else:
                if os.path.exists(seg_res):
                    shutil.rmtree(seg_res)
        # 1sam 1unsam 2unsam all
        if config["basic"]["shape_num"] is not None:
            if unique_masks_len < mask_num:
                #     如果结构数量不够，且限制数量，复制一轮的mask直到数量足够
                if config["double_seg"]["copy"]:
                    # 需要复制补齐
                    final_len = 0
                    # 循环平铺复制直到数量达到 mask_num
                    while final_len < mask_num:
                        # 逐个追加 unique_masks
                        for m in unique_masks:
                            final_masks.append(m.copy())  # 建议 copy 以免引用同一个 dict
                            final_len +=1
                            if final_len >= mask_num:
                                break
                    final_masks = sort_masks_by_area(final_masks)

                else:
                    final_masks = unique_masks

            else:
                # 结构数量过多，需要筛选
                sam_masks_len = len(l2_sam_masks)
                unsam_masks_len = len(l2_unsam_masks)
                # sam unsam
                if mask_num <= sam_masks_len:
                    final_masks = l2_sam_masks[:mask_num]
                elif sam_masks_len < mask_num <= sam_masks_len + unsam_masks_len:
                    final_masks = l2_unsam_masks[:mask_num - sam_masks_len] + l2_sam_masks
                    final_masks = sort_masks_by_area(final_masks)
                elif mask_num > sam_masks_len + unsam_masks_len:
                    final_masks = unique_masks[:mask_num]
        else:
            final_masks = unique_masks[:mask_num]

        if len(final_masks) > 0:
            visualize_masks(image_rgb, final_masks, f"{output_dir}/final_init_a.png",
                            f"{output_dir}/final_init.png", None, True)

        # mask保存：/mask 同时保存生成pbm需要的png
        mask_path = f"{output_dir}/mask"
        os.makedirs(mask_path, exist_ok=True)
        p_s = time.time()
        save_masks(final_masks, mask_path, base_size,simplify_eps_ratio)
        # ai_mask_path = f"{output_dir}/ai_mask"
        # save_masked_regions(image_rgb, final_masks, ai_mask_path, base_size)
        final_mask_len = 0
        for j, m in enumerate(final_masks):
            if j < mask_num:
                final_mask_len += 1
                masks_path = f"{output_dir}/mask/mask_{j}.png"
                mask_to_svg(masks_path, f"{output_dir}/mask/mask_{j}.svg",config["vtracer"]["c"],config["vtracer"]["s"])
        print(
            f"{i + 1}/{len(images_path)}: {image_name} final mask num:{final_mask_len}")
        if not config["save"]["sam"]["sam_pbm"]:
            delete_files(mask_path, ".pbm")
        if not config["save"]["sam"]["sam_mask_png"]:
            delete_files(mask_path, ".png")
        IVVLD_path = f"{output_dir}/IVVLD"
        if config["save"]["sam"]["IVVLD"]:
            # 用于ImageVectorViaLayerDecomposition方法实验
            IVVLD_filter_path = f"{IVVLD_path}/filter"
            os.makedirs(IVVLD_path, exist_ok=True)
            cv2.imwrite(f"{IVVLD_path}/input.png", image)
            visualize_masks(image_rgb, final_masks, f"{IVVLD_path}/mask/vis_mask_overlay.png",
                            f"{IVVLD_path}/seg.png",
                            f"{IVVLD_path}/mask.png", config["save"]["sam"]["sam_mask_png"])
            # 指定数量的mask
            os.makedirs(IVVLD_filter_path, exist_ok=True)
            cv2.imwrite(f"{IVVLD_filter_path}/input.png", image)
            visualize_masks(image_rgb, final_masks[:mask_num],
                            f"{IVVLD_filter_path}/mask/vis_filter_mask_overlay.png",
                            f"{IVVLD_filter_path}/seg.png",
                            f"{IVVLD_filter_path}/mask.png", config["save"]["sam"]["sam_mask_png"])

        else:
            if os.path.exists(IVVLD_path):
                shutil.rmtree(IVVLD_path)
        potrace_time = time.time() - p_s
        print(
            f"{i + 1}/{len(images_path)}: {image_name} "
            f" potrace time:{potrace_time}.")
        # 保存的mask对应的颜色(不透明) 和颜色类别
        color_s = time.time()
        colors, color_types = compute_average_rgba_per_mask(image_path, final_masks,max_stop_num=config["optimization"]["max_stop_number"], stop_num=config["optimization"]["stop_number"], fill_type=color_type,
                                                            device=config["basic"]["device"])
        color_time = time.time() - color_s
        print(
            f"{i + 1}/{len(images_path)}: {image_name} "
            f" color init time:{color_time}.")
        colors_list.append([colors, color_types])
        structure_time_e = time.time()
        structure_time = structure_time_e - structure_time_s
        total_color_init_time += color_time
        total_initial_mask_len += initial_masks_len
        total_unique_mask_len += unique_masks_len

        now_json["structure"]["structure_time"] = structure_time
        now_json["structure"]["initial_masks_len"] = initial_masks_len
        now_json["structure"]["unique_masks_len"] = unique_masks_len
        now_json["structure"]["final_masks_len"] = final_mask_len
        now_json["structure"]["color_init_time"] = color_time
        now_json["structure"]["mask_colors"] = json.dumps([colors, color_types])
        json_save_info(json_save, now_json_file_name, now_json)

    print(f"all {len(images_path)} images finish structure processing")
    total_time_e = time.time()
    total_time = total_time_e - total_time_s

    if double_seg_enable:
        json_info["structure"]["avg_1_sam_mask_len"] = total_sam_mask_len / len(images_path)
        json_info["structure"]["avg_1_unsam_mask_len"] = total_unsam_mask_len / len(images_path)
        json_info["structure"]["avg_1_unique_mask_len"] = total_1_mask_len / len(images_path)
        json_info["structure"]["avg_2_unsam_bg_mask_len"] = total_2_unsam_bg_mask_len / len(images_path)
        json_info["structure"]["avg_2_unsam_mask_len"] = total_2_unsam_mask_len / len(images_path)
        json_info["structure"]["avg_2_unique_mask_len"] = total_2_mask_len / len(images_path)

    json_info["structure"]["sam_total_time"] = total_time
    json_info["structure"]["sam_avg_time"] = total_time / len(images_path)
    json_info["structure"]["avg_color_init_time"] = total_color_init_time / len(images_path)
    json_info["structure"]["avg_unique_mask_len"] = total_unique_mask_len / len(images_path)
    json_info["structure"]["avg_initial_mask_len"] = total_initial_mask_len / len(images_path)
    json_save_info(json_save, json_file_name, json_info)
    # else:
    #     for i, image_path in enumerate(images_path):
    #         image_name = get_png_name_without_suffix(image_path)
    #         svg_dir_path = f"{output_top_dir}/structure/mask"
    #         p = Path(svg_dir_path)
    #         svg_count = sum(1 for f in p.iterdir() if f.is_file() and f.name.startswith("mask"))
    #         if os.path.exists(f"{svg_dir_path}/mask_{mask_num - 1}.svg") and \
    #                 os.path.isfile(f"{svg_dir_path}/mask_{mask_num - 1}.svg") and \
    #                 svg_count == mask_num:
    #
    #             image_names.append(image_name)
    #             now_json_file_name = f"{output_top_dir}/{image_name}/info.json"
    #             if os.path.exists(now_json_file_name):
    #                 with open(now_json_file_name, 'r', encoding='utf-8') as f:
    #                     now_json_info = json.load(f)
    #                 colors_list.append(now_json_info["structure"]["mask_colors"])
    #             else:
    #                 raise FileNotFoundError(now_json_file_name)
    #         else:
    #             raise FileNotFoundError(f"{image_name} doesn't have right masks!")
    # print(f"Strcut mask_num: {mask_num}")
    return colors_list, image_names, json_file_name, mask_num
