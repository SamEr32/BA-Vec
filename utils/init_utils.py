from PIL import Image
import torch
from torchvision import transforms
from sklearn.cluster import KMeans
import sys
import os
from color_net.net.net import ColorParamNet
import time
import cv2
import numpy as np
import json
import subprocess
from typing import List, Dict, Tuple
import yaml
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import torch
import cv2
from sklearn.cluster import MiniBatchKMeans
from xml.etree import ElementTree as ET
import glob
import re
import shutil
import pydiffvg
import skimage.io
import random
from datetime import datetime
from utils.file_utils import load_image, batch_input, get_png_name_without_suffix, get_all_png_images, delete_files, \
    ensure_clean_dir, read_config, json_save_info
import numpy as np
import torch
from typing import List, Dict
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def get_adaptive_mask_crop(image_rgb, mask, margin_ratio=0.05, tight=True):
    """
    获取自适应 mask 裁剪区域（避免细长/倾斜目标过大）
    - tight=True 使用轮廓裁剪（推荐）
    - margin_ratio 控制边缘扩展
    """
    H, W = mask.shape[:2]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None, None

    # --- 获取mask轮廓 ---
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    if tight:
        # 使用旋转矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(int)
        x_min, y_min = np.clip(box[:, 0].min(), 0, W - 1), np.clip(box[:, 1].min(), 0, H - 1)
        x_max, y_max = np.clip(box[:, 0].max(), 0, W - 1), np.clip(box[:, 1].max(), 0, H - 1)
    else:
        x_min, y_min, w, h = cv2.boundingRect(cnt)
        x_max, y_max = x_min + w, y_min + h

    # --- margin 扩展 ---
    margin_x = int((x_max - x_min) * margin_ratio)
    margin_y = int((y_max - y_min) * margin_ratio)
    x_min, y_min = max(x_min - margin_x, 0), max(y_min - margin_y, 0)
    x_max, y_max = min(x_max + margin_x, W - 1), min(y_max + margin_y, H - 1)

    # --- 裁剪 ---
    patch_img = image_rgb[y_min:y_max, x_min:x_max]
    patch_mask = mask[y_min:y_max, x_min:x_max]

    return patch_img, patch_mask, (x_min, y_min, x_max, y_max)


def compute_contour_complexity(contour):
    """
    输入：
        contour: np.ndarray, shape=(N, 2)，轮廓点
    输出：
        complexity: float, 综合复杂度指标 [0,1]，越大越复杂
    """
    contour = contour.reshape(-1, 2)

    # 1️⃣ 周长归一化
    perimeter = cv2.arcLength(contour, True)

    # 2️⃣ 面积与凸包差
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        convexity = 0.0
    else:
        convexity = (hull_area - area) / hull_area  # 凸凹变化占比

    # 3️⃣ 拐角数量（Douglas-Peucker算法简化点数）
    epsilon = 0.01 * perimeter  # 精度可调
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corner_count = len(approx)
    corner_norm = corner_count / len(contour)

    # 综合指标
    complexity = 0.4 * min(perimeter / 1000, 1.0) + 0.4 * min(convexity, 1.0) + 0.2 * min(corner_norm, 1.0)
    complexity = np.clip(complexity, 0.0, 1.0)
    return complexity


def suggest_alphamax_from_complexity(complexity, min_val=0.5, max_val=1.5):
    """
    根据复杂度建议 Potrace 的 alphamax 参数
    complexity: [0,1], 越大越复杂
    alphamax 越小保留越多细节
    """
    alphamax = max_val - complexity * (max_val - min_val)
    return alphamax


def suggest_opttolerance_from_complexity(complexity, min_val=0.1, max_val=0.3):
    """
    根据复杂度建议 Potrace 的 opttolerance 参数
    complexity: [0,1], 越大越复杂
    opttolerance 越小越保留细节，越大越平滑
    """
    opttolerance = max_val - complexity * (max_val - min_val)
    return opttolerance


def analyze_mask_colors(image_rgb, mask, bins=32, min_pixels_rate=0.001, n=4):
    colors = image_rgb[mask]
    if len(colors) == 0:
        return []

    # 量化到 bins×bins×bins
    quant = (colors // (256 // bins)).astype(int)
    idx = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    counts = np.bincount(idx, minlength=bins ** 3)
    top_idx = counts.argsort()[-n:][::-1]

    clusters = []
    min_pixels = max(10, len(colors) * min_pixels_rate)
    for i in top_idx:
        count = counts[i]
        if count < min_pixels:
            continue
        r = (i // (bins * bins)) * (256 // bins) + (128 // bins)
        g = ((i // bins) % bins) * (256 // bins) + (128 // bins)
        b = (i % bins) * (256 // bins) + (128 // bins)
        clusters.append({"color": [int(r), int(g), int(b)], "count": int(count)})

    return clusters


# 统计points shapes
def count_points_and_shapes(svg_path):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    s_len = len(shapes)
    p_len = 0.0

    for s in shapes:
        p_len += len(s.points)

    return p_len, s_len


# 统计透明度低的shape数量
def count_low_alpha_primitives(svg_path=None, svg=None, alpha_threshold=0.5):
    if svg is None:
        if svg_path is not None:
            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
        else:
            raise ValueError("Provide either svg_path or svg.")
    else:
        canvas_width, canvas_height, shapes, shape_groups = svg

    count = 0

    for shape, group in zip(shapes, shape_groups):
        fill = group.fill_color
        if isinstance(fill, torch.Tensor):
            # 固定颜色填充
            alpha = fill[3].item()
            if alpha < alpha_threshold:
                count += 1
        elif isinstance(fill, (pydiffvg.LinearGradient, pydiffvg.RadialGradient)):
            # 渐变填充
            alphas = fill.stop_colors[:, 3]
            if torch.all(alphas < alpha_threshold):
                count += 1

    return count, len(shapes)


# 移除不透明度低的矢量
def remove_high_alpha_primitives(svg_path=None,
                                 svg=None,
                                 alpha_threshold=0.2):
    if svg is None:
        if svg_path is None:
            raise ValueError("Provide either svg_path or svg.")
        canvas_w, canvas_h, shapes, groups = pydiffvg.svg_to_scene(svg_path)
    else:
        canvas_w, canvas_h, shapes, groups = svg

    new_shapes, new_groups = [], []
    keep_map = {}  # old_id -> new_id

    # ---------- 1. 过滤 & 记录保留 shape ----------
    for old_idx, (shape, group) in enumerate(zip(shapes, groups)):
        remove = False
        fill = group.fill_color

        if isinstance(fill, torch.Tensor):  # flat color
            remove = fill[3].item() < alpha_threshold
        elif isinstance(fill,
                        (pydiffvg.LinearGradient,
                         pydiffvg.RadialGradient)):  # gradient
            remove = torch.all(fill.stop_colors[:, 3] < alpha_threshold)

        if not remove:
            new_id = len(new_shapes)
            shape.id = new_id  # ★ 同步 shape.id
            new_shapes.append(shape)
            keep_map[old_idx] = new_id

    # ---------- 2. 修复每个 group 的 shape_ids ----------
    for group in groups:
        new_ids = [keep_map[i] for i in group.shape_ids.tolist()
                   if i in keep_map]
        if new_ids:  # 至少还有一个 shape 属于该 group
            group.shape_ids = torch.tensor(new_ids,
                                           dtype=torch.int64,
                                           device=group.shape_ids.device)
            new_groups.append(group)

    loss_n = len(shapes) - len(new_shapes)
    print(f"移除了透明度低于{alpha_threshold}的基元{loss_n}个")
    return canvas_w, canvas_h, new_shapes, new_groups, loss_n


# 计算shape面积
def compute_shape_area(shape):
    """
    估算一个封闭 shape（如 pydiffvg.Path）的面积。
    基于控制点构建多边形并计算面积。
    """
    if not isinstance(shape, pydiffvg.Path):
        raise TypeError("Only pydiffvg.Path is supported.")

    # shape.points: [num_segments * 3, 2] (每段Bezier曲线用3个控制点）
    points = shape.points.cpu().detach().numpy().reshape(-1, 2)

    # 简单估算面积的方法：连接控制点近似多边形，使用多边形面积公式
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


# 找出最大面积shape
def find_largest_shape(shapes):
    """
    输入：形状列表（pydiffvg.Path 类型）
    输出：面积最大的 shape 对象
    """
    if not shapes:
        return None

    max_area = -1
    largest_shape = None
    for shape in shapes:
        try:
            area = compute_shape_area(shape)
            if area > max_area:
                max_area = area
                largest_shape = shape
        except Exception as e:
            print(f"跳过 shape（原因：{e}）")

    return largest_shape


# 闭运算避免断裂 消除孔洞
def adaptive_close_mask(mask: np.ndarray, base_size: int = 256, max_kernel: int = 7, min_kernel: int = 1) -> np.ndarray:
    """
    对 mask 进行自适应闭运算，小图（≤72x72）跳过处理。

    Parameters:
        mask (np.ndarray): 输入的二值图像（0/255 或 0/1）。
        base_size (int): 控制核大小增长速率的基准图像尺寸。
        max_kernel (int): 最大核尺寸（必须为奇数）。
        min_kernel (int): 最小核尺寸（必须为奇数）。

    Returns:
        np.ndarray: 处理后的 mask。
    """
    h, w = mask.shape[:2]

    # 小图直接跳过闭运算
    if h <= 72 and w <= 72:
        return mask.copy()

    # 按比例自适应核大小
    scale = max(h, w) / base_size
    kernel_size = int(round(scale * min_kernel))
    kernel_size = max(min_kernel, min(kernel_size, max_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1  # 保证为奇数

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return closed_mask


def infer_fill_type_and_params(image_patch: Image.Image, model: ColorParamNet, device='cuda', min_size=32):
    """
    对单个 mask 区域裁剪图像，预测 fill_type 和颜色参数
    自动处理过小的 patch，防止 cuDNN 卷积报错。
    """
    # --- 保证最小尺寸 ---
    w, h = image_patch.size
    scale = max(1, min_size / min(w, h))
    new_w, new_h = max(min_size, int(w * scale)), max(min_size, int(h * scale))
    if (new_w, new_h) != (w, h):
        image_patch = image_patch.resize((new_w, new_h), Image.BILINEAR)

    # --- 转 Tensor ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.0], std=[0.5, 0.5, 0.5, 1.0])
    ])
    img_tensor = transform(image_patch).unsqueeze(0).to(device)

    # --- 确保 cuDNN 使用优化算法 ---
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model.eval()
    with torch.no_grad():
        cls_logits, params = model(img_tensor)
        cls_pred = torch.argmax(cls_logits, dim=1).item()
        params = params.squeeze(0).cpu().tolist()

    return cls_pred, params


def compute_average_rgba_per_mask(
        image_path: str,
        masks: List[Dict],
        model_checkpoint='/HOME/scw6ecz/run/ngx/ours/color_net/model/model_final.pth',
        device='cuda', stop_num=4, max_stop_num=4,
        fill_type=3  # 0/1/2/3
):
    """
    综合版颜色初始化：
    - 网络预测填充类型（0/1/2/3）
    - 算法计算每个mask的颜色参数（stop colors）
    """

    # --- 加载图像 ---
    try:
        _, image_rgb, a_mask = load_image(image_path, None, "white", False)

        image_rgb_np = np.array(image_rgb)
    except Exception as e:
        raise IOError(f"Error loading image: {image_path}\n{e}")

    # --- 初始化网络 ---
    model = None
    if fill_type == 3 and model_checkpoint is not None:
        model = ColorParamNet().to(device)
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model.eval()

    average_colors = []
    colors_type = []
    if stop_num == 0:
        stop_num = max_stop_num
    for m in masks:
        mask = m['segmentation'].astype(bool)
        if mask.sum() == 0:
            average_colors.append([[0, 0, 0, 0]])
            colors_type.append(0)
            continue

        masked_rgb = image_rgb_np[mask]
        avg = [float(c) for c in masked_rgb.mean(axis=0)] + [255]

        # --- 网络预测填充类型 ---
        current_fill_type = fill_type
        if model is not None and fill_type == 3:
            y_idx, x_idx = np.where(mask)
            patch_h = y_idx.max() - y_idx.min() + 1
            patch_w = x_idx.max() - x_idx.min() + 1
            patch_rgba = np.zeros((patch_h, patch_w, 4), dtype=np.uint8)
            patch_rgba[:, :, :3] = image_rgb_np[y_idx.min():y_idx.max() + 1, x_idx.min():x_idx.max() + 1, :]
            patch_rgba[:, :, 3] = (
                    mask[y_idx.min():y_idx.max() + 1, x_idx.min():x_idx.max() + 1].astype(np.uint8) * 255)
            patch_img = Image.fromarray(patch_rgba)
            cls_pred, _ = infer_fill_type_and_params(patch_img, model, device)
            current_fill_type = cls_pred

        # --- 基于算法的颜色参数提取 ---
        if current_fill_type in [1, 2]:  # 线性或径向渐变 → 提取多个 stop colors

            clusters = analyze_mask_colors(image_rgb_np, mask, n=stop_num)
            if len(clusters) <= 1:
                # 无法聚出有效颜色 → 使用平均色代替
                current_fill_type = 0
                average_colors.append([avg])
            else:
                # 取颜色均值并加上 alpha
                sorted_colors = [[int(x) for x in np.clip(c["color"], 0, 255)] + [255] for c in clusters]
                average_colors.append(sorted_colors)
        else:
            # 非渐变（平面填充） → 平均色
            average_colors.append([avg])

        colors_type.append(current_fill_type)

    return average_colors, colors_type

    # 自适应核双边滤波


def adaptive_bilateral_filter(image, base_size=None, max_d=20, max_sigma=100):
    if base_size is None:
        base_size = min(image.shape[:2])
    d = min(max_d, max(5, base_size // 50))
    sigmaColor = sigmaSpace = min(max_sigma, max(25, base_size // 10))
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    # 自适应核高斯滤波


def adaptive_gaussian_blur(image, base_size=None, max_ksize=3, min_blur_size=72):
    """
    自适应高斯模糊：仅当图片尺寸≥min_blur_size×min_blur_size时执行模糊，否则返回原图
    参数说明：
        image: 输入图像（cv2格式，H×W×C 或 H×W）
        base_size: 用于计算核尺寸的基准值，默认取图像短边长度
        max_ksize: 最大模糊核尺寸（需为奇数），默认13
        min_blur_size: 执行模糊的最小尺寸阈值，默认72（即≥72×72才模糊）
    返回：
        处理后的图像（模糊/原图）
    """
    # 获取图像的高、宽（兼容单通道/多通道）
    h, w = image.shape[:2]

    # 判定：仅当高和宽都≥min_blur_size时才执行模糊
    if h > min_blur_size and w > min_blur_size:
        if base_size is None:
            base_size = min(h, w)  # 原逻辑：基准值取短边
        # 计算自适应核尺寸（保证是奇数，且≤max_ksize、≥3）
        ksize = min(max_ksize, max(3, int(base_size / 100) * 2 + 1))
        # 确保核尺寸为奇数（cv2.GaussianBlur要求）
        if ksize % 2 == 0:
            ksize += 1
        # 执行高斯模糊
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return blurred
    else:
        # 尺寸不足，直接返回原图（避免拷贝，节省内存）
        return image


def resize_for_segmentation(image_rgb,
                            min_long_side=72,
                            max_long_side=1024):
    """
    将图像缩放到 [min_long_side, max_long_side] 范围内。
    - 如果图像太小 -> 放大
    - 如果图像太大 -> 缩小
    - 保持宽高比

    返回：
        resized_img: 缩放后的图像
        scale:        缩放比例（float）
    """
    H, W = image_rgb.shape[:2]
    long_side = max(H, W)

    # ① 长边小于 min_long_side：放大
    if long_side < min_long_side:
        scale = min_long_side / long_side

    # ② 长边大于 max_long_side：缩小
    elif long_side > max_long_side:
        scale = max_long_side / long_side

    # ③ 无需缩放
    else:
        return image_rgb, 1.0

    new_w = int(W * scale)
    new_h = int(H * scale)

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def restore_mask_to_original_size(mask, original_shape):
    """
    将 mask 从缩放后的尺寸还原回原图尺寸。
    使用最近邻插值保持二值性。
    """
    H, W = original_shape
    mask_uint8 = mask.astype(np.uint8)

    restored = cv2.resize(
        mask_uint8,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )
    return restored.astype(bool)
