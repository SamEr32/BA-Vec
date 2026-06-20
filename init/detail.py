import time

import cv2
import numpy as np
import os
import json
import subprocess
from typing import List, Dict, Tuple
import utils.init_utils as init_utils
import utils.file_utils as file_utils
from init.structure import compute_average_rgba_per_mask
from utils.file_utils import read_config, load_image, ensure_clean_dir, delete_if_empty,mask_to_svg
from utils.init_utils import adaptive_close_mask
from PIL import Image
import numpy as np
from skimage.filters import threshold_multiotsu



def read_image_with_white_bg(path):
    """
    读取图像，将 P 模式或 RGBA 图像转换为 RGB 图像，自动使用白色背景进行 alpha 合成。

    返回：
    - numpy.ndarray, dtype=uint8, shape=(H, W, 3), 范围 [0, 255]
    """
    img = Image.open(path)

    if img.mode == 'P':
        # P 模式：先转为 RGBA
        img = img.convert('RGBA')

    if img.mode == 'RGBA':
        # RGBA 模式：使用白色背景合成为 RGB
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])  # 使用 alpha 通道
        img = white_bg

    elif img.mode != 'RGB':
        img = img.convert('RGB')
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_bgr  # 返回 uint8, [0, 255]


# 防止细节为空
def get_max_gradient_contour(gray_img: np.ndarray, r: int = 0.1):
    """
    在灰度图中寻找梯度最大点，以该点为中心返回一个矩形轮廓（可当作补充路径）

    参数：
        gray_img: 输入的灰度图，shape=(H, W)
        r: 半径，返回的轮廓为 2r x 2r 的矩形
    返回：
        contour: list[list[int, int]]，表示轮廓点序列（闭环）
    """
    # 确保图像是灰度的
    assert len(gray_img.shape) == 2, "输入图像应为灰度图"

    # 计算梯度（Sobel）
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 找到梯度最大位置
    max_y, max_x = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)

    # 构造一个矩形轮廓 centered at (max_x, max_y)
    H, W = gray_img.shape
    x1 = max(max_x - r, 0)
    y1 = max(max_y - r, 0)
    x2 = min(max_x + r, W - 1)
    y2 = min(max_y + r, H - 1)

    # 构造矩形轮廓（闭环）
    contour = [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
        [x1, y1]  # 闭环
    ]
    return contour


# === 步骤 1：误差区域轮廓提取（带可选 SSIM+absdiff 融合） ===
def extract_significant_contours(target_path, rendered_path, save_dir, save, json_info,
                                 top_n=5, area_weight=0.5, perimeter_weight=0.5,
                                 min_area_ratio=0.00001,  # 占总面积的比例
                                 min_perimeter_ratio=0.00001,  # 占总周长的比例
                                 blur_type="bilateral", ratio_method=0, zero=False,
                                 use_ssim=False, ssim_weight=0.8, save_fused=False,copy=True):
    """
    原函数基础上增加可选 SSIM+absdiff 融合：
      - use_ssim: 是否启用 SSIM map（默认 False 保持原行为）
      - ssim_weight: 融合权重，范围 [0,1]，越大越偏向 SSIM 差异
      - save_fused: 若 True 且 save["difference"] True，会保存融合后的差异图（便于调参）
    其它行为与原函数一致。
    """
    import os
    import cv2
    import numpy as np
    import json

    os.makedirs(save_dir, exist_ok=True)

    target_img = read_image_with_white_bg(target_path)
    rendered_img = read_image_with_white_bg(rendered_path)

    # --- 基本 absdiff 灰度残差 ---
    residual = cv2.absdiff(target_img, rendered_img)
    residual_gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

    # save 原始残差（保持原逻辑）
    if save.get("difference", False):
        cv2.imwrite(os.path.join(save_dir, "residual_gray.png"), residual_gray)

    # --- 可选：计算 SSIM 差异图并与 absdiff 加权融合 ---
    fused_gray = residual_gray  # 默认为原始残差
    if use_ssim:
        try:
            from skimage.metrics import structural_similarity as ssim
            # skimage ssim expects 2D arrays, uint8
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            rendered_gray = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2GRAY)
            # compute full SSIM map; returns (score, map) where map in [-1,1], close to 1 means similar
            _, ssim_map = ssim(target_gray, rendered_gray, full=True)
            # convert ssim map to difference intensity in [0,255]: diff = (1 - ssim_map) -> [0,2], normalize
            diff_map = (1.0 - ssim_map).astype(np.float32)
            # normalize diff_map to 0..255
            dm_min, dm_max = float(diff_map.min()), float(diff_map.max())
            if dm_max - dm_min > 1e-6:
                diff_map_u8 = ((diff_map - dm_min) / (dm_max - dm_min) * 255.0).astype(np.uint8)
            else:
                diff_map_u8 = (diff_map * 0).astype(np.uint8)

            # fused = weight * ssim_diff + (1-weight) * absdiff
            w = float(np.clip(ssim_weight, 0.0, 1.0))
            fused = (w * diff_map_u8.astype(np.float32) + (1.0 - w) * residual_gray.astype(np.float32))
            # convert to uint8 normalized
            fused = np.clip(fused, 0, 255).astype(np.uint8)
            fused_gray = fused

            if save_fused and save.get("difference", False):
                cv2.imwrite(os.path.join(save_dir, "fused_ssim_absdiff.png"), fused_gray)
        except Exception as e:
            # 若计算或导入失败，回退为 residual_gray（并且不抛错）
            fused_gray = residual_gray
            # 记录到 json_info 以便调试
            json_info.setdefault("detail", {})["ssim_error"] = str(e)

    # --- 后续保持原处理流程：模糊 -> 自适应阈值 -> findContours ---
    if blur_type == "gaussian":
        blurred = cv2.GaussianBlur(fused_gray, (5, 5), 0)
    elif blur_type == "bilateral":
        blurred = cv2.bilateralFilter(fused_gray, d=9, sigmaColor=75, sigmaSpace=75)
    elif blur_type == "adaptive_bilateral":
        blurred = init_utils.adaptive_bilateral_filter(fused_gray)
    elif blur_type == "adaptive_gaussian":
        blurred = init_utils.adaptive_gaussian_blur(fused_gray)
    elif blur_type == "direct_threshold":
        # 新增：直接对融合后的灰度图进行自适应阈值二值化
        # 不进行任何模糊操作，直接使用 fused_gray
        blurred = fused_gray
    else:
        raise ValueError("不支持的模糊类型。")

    thresholds = threshold_multiotsu(blurred, classes=2)
    regions = np.digitize(fused_gray, bins=thresholds)
    adaptive_mask = (regions > 0).astype(np.uint8) * 255


    if save.get("difference", False):
        cv2.imwrite(os.path.join(save_dir, f"{blur_type}.png"), blurred)
        cv2.imwrite(os.path.join(save_dir, f"{blur_type}_mask.png"), adaptive_mask)

    # 提取内外所有轮廓
    contours, hierarchy = cv2.findContours(adaptive_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)


    height, width = target_img.shape[:2]
    image_area = width * height
    min_area = min_area_ratio * image_area
    min_perimeter = min_perimeter_ratio * (width + height)

    sorted_masks = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        sorted_masks.append((area, perimeter, contour))
    sorted_masks = sorted(sorted_masks, key=lambda x: area_weight * x[0] + perimeter_weight * x[1], reverse=True)
    json_info.setdefault("detail", {})["initial_contours_len"] = len(sorted_masks)

    filtered = []
    for area, perimeter, contour in sorted_masks:
        if area >= min_area and perimeter >= min_perimeter:
            filtered.append((area, perimeter, contour))

    json_info["detail"]["filtered_contours_len"] = len(filtered)
    final = []
    if top_n > len(filtered):
        if top_n <= len(sorted_masks):
            final = sorted_masks[:top_n]
        else:
            if copy:
                final_len = 0
                # 循环平铺复制直到数量达到 mask_num
                while final_len < top_n:
                    # 逐个追加 unique_masks
                    for m in sorted_masks:
                        final.append(m)  # 建议 copy 以免引用同一个 dict
                        final_len += 1
                        if final_len >= top_n:
                            break
            else:
                final = sorted_masks
    else:
        final = filtered[:top_n]
    json_info["detail"]["final_contours_len"] = len(final)

    if len(final) == 0:
        if zero:
            # fallback: 使用 fused_gray 的梯度点
            fallback_contour = np.array(get_max_gradient_contour(fused_gray, r=3)).reshape(-1, 1, 2)
            final.append((0.0, 0.0, fallback_contour))
        else:
            return False

    output_contours = [contour[:, 0, :].tolist() for _, _, contour in final]

    with open(os.path.join(save_dir, "init_paths.json"), "w") as f:
        json.dump(output_contours, f, indent=2)

    vis = target_img.copy()
    cv2.drawContours(vis, [np.array(c, dtype=np.int32) for c in output_contours], -1, (255, 0, 0), 2)
    if save.get("difference", False):
        cv2.imwrite(os.path.join(save_dir, "contours_overlay.png"), vis)

    return os.path.join(save_dir, "init_paths.json"), target_img.shape, json_info


# === 步骤 2：从轮廓生成掩码图 ===
def simplify_contour_rdp(
    contour,
    epsilon_ratio: float = 0.01
):
    """
    对单个轮廓做 RDP 简化
    contour: (N, 1, 2) or (N, 2)
    epsilon_ratio: 相对于轮廓周长的比例
    """
    import cv2
    import numpy as np

    cnt = np.asarray(contour, dtype=np.int32)
    if cnt.ndim == 2:
        cnt = cnt.reshape((-1, 1, 2))

    if len(cnt) < 10:
        return cnt

    perimeter = cv2.arcLength(cnt, closed=True)
    epsilon = epsilon_ratio * perimeter

    approx = cv2.approxPolyDP(
        cnt,
        epsilon=epsilon,
        closed=True
    )

    return approx

def contours_to_masks(json_path: str, image_shape: Tuple[int, int], save_dir: str, base_size=256, simplify_eps_ratio=0) -> Tuple[
    List[str], List[Dict]]:
    """
    从 JSON 中加载轮廓，生成掩码图像，保存为 PNG 文件，并返回文件路径和用于颜色计算的掩码列表。
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(json_path, "r") as f:
        contours = json.load(f)

    h, w = image_shape[:2]
    mask_paths = []
    masks: List[Dict] = []

    for i, contour in enumerate(contours):
        mask = np.zeros((h, w), dtype=np.uint8)
        # === RDP 简化轮廓 ===
        if simplify_eps_ratio != 0:
            contour = simplify_contour_rdp(
                contour,
                epsilon_ratio=simplify_eps_ratio
            )
        points = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))

        cv2.drawContours(mask, [points], -1, 255, thickness=cv2.FILLED)

        # 保存图像
        path = os.path.join(save_dir, f"mask_{i}.png")
        # 消除细小断裂 孔洞等
        mask = adaptive_close_mask(mask, base_size)
        cv2.imwrite(path, 255 -mask)

        mask_paths.append(path)

        # 添加到掩码列表（注意 segmentation 需要是 bool 类型）
        masks.append({"segmentation": (mask > 0).astype(np.uint8)})

    os.remove(json_path)
    return mask_paths, masks


# 批量转换mask
def batch_convert_masks_to_svg(mask_paths, output_dir,c,s):
    for mask_path in mask_paths:
        # print(mask_path)
        name = os.path.splitext(os.path.basename(mask_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_svg_file = os.path.join(output_dir, f"{name}.svg")
        mask_to_svg(mask_path, output_svg_file,c,s)



# === 主控封装函数 ===
def detail_capture(target_path, rendered_path, save_dir, save, json_info, area_weight, perimeter_weight,max_s_n,s_n, potrace_O=1.5,
                   potrace_a=1.2,
                   top_n=5, min_area_ratio=0.00001, min_perimeter_ratio=0.00001,
                   blur_type="gaussian", base_size=256, ratio_method=0, color_type=3, device="cuda",copy=True,c=100,s=80,simplify_eps_ratio=0):
    contours_detail_dir = f"{save_dir}/detail"
    difference_dir = f"{contours_detail_dir}/difference"
    ensure_clean_dir(contours_detail_dir)
    ensure_clean_dir(difference_dir)
    res = extract_significant_contours(
        target_path, rendered_path, difference_dir, save, json_info,
        top_n, area_weight, perimeter_weight, min_area_ratio, min_perimeter_ratio, blur_type, ratio_method, copy    )
    if res:
        contours_json_path, shape, json_info = res
    else:
        return False
        # print(json_info)
    mask_dir = os.path.join(contours_detail_dir, "mask")

    mask_paths, masks = contours_to_masks(contours_json_path, shape, mask_dir, base_size,simplify_eps_ratio)
    c_s = time.time()
    colors, color_types = compute_average_rgba_per_mask(target_path, masks,max_stop_num=max_s_n,stop_num=s_n, fill_type=color_type, device=device)
    c_time = time.time() - c_s
    # print(mask_paths)
    batch_convert_masks_to_svg(mask_paths, mask_dir,c,s)
    if not save["detail_png"]:
        file_utils.delete_files(mask_dir, ".png")
    if not save["detail_pbm"]:
        file_utils.delete_files(mask_dir, ".pbm")

    # print(f"🎉 全部完成！SVG 输出目录：{svg_dir}")
    return colors, color_types, json_info
