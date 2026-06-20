import os
import torch
import xml.etree.ElementTree as ET
import pydiffvg
import yaml
import lpips
import copy
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import skimage.io
from typing import List
import random
import numpy as np
import cv2
from pathlib import Path
import svgwrite
from math import log10
from skimage.measure import label
from skimage.io import imread
from scipy.ndimage import label
import math
from scipy import ndimage
from PIL import Image
import argparse
from easydict import EasyDict as edict
from utils import file_utils

lpips_model = lpips.LPIPS(net='alex')  # 可选 'alex' / 'vgg' / 'squeeze'
lpips_model.eval()


# 保存线稿及其png
def render_svg_outline(svg_path, output_png_path, output_svg_path, stroke_width=0.1):
    """
    用 diffvg 直接渲染 SVG，保存原始 RGBA 图像，不做任何后处理。
    """
    import pydiffvg
    import torch
    from pathlib import Path

    svg_path = Path(svg_path)
    output_png_path = Path(output_png_path)
    output_svg_path = Path(output_svg_path)

    pydiffvg.set_use_gpu(torch.cuda.is_available())

    # 1. 加载 SVG
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(str(svg_path))
    # diagonal = (canvas_width ** 2 + canvas_height ** 2) ** 0.5
    # stroke_width_tensor = torch.tensor(diagonal * 0.003)
    stroke_width_tensor = torch.tensor(stroke_width)
    # 2. 设置颜色属性（黑线，白底透明填充）
    for group in shape_groups:
        group.fill_color = torch.tensor([1.0, 1.0, 1.0, 1.0])  # 白色不透明填充
        group.stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])  # 黑色不透明描边
    for shape in shapes:
        shape.stroke_width = stroke_width_tensor

    # 3. 渲染
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 64, 64, 0, None, *scene_args)  # float tensor [H, W, 4], [0,1]

    # 4. 直接保存为 PNG，保持原始 RGBA
    pydiffvg.imwrite(img.cpu(), str(output_png_path))

    # 5. 保存 SVG
    pydiffvg.save_svg(str(output_svg_path), canvas_width, canvas_height, shapes, shape_groups)


# 渲染并保存图片RGBA
def render_svg(svg_path, output_png_path):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    pydiffvg.imwrite(img.cpu(), output_png_path, gamma=1.0)


# LIVE的衰减器
class linear_decay_lrlambda_f(object):
    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n // self.decay_every
        decay_step = n % self.decay_every
        lr_s = self.decay_ratio ** decay_time
        lr_e = self.decay_ratio ** (decay_time + 1)
        r = decay_step / self.decay_every
        lr = lr_s * (1 - r) + lr_e * r
        return lr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--vtracer_c", type=int, default=100, help=" corner threshold of vtracer")
    parser.add_argument("--vtracer_s", type=int, default=80, help=" splice threshold of vtracer")

    parser.add_argument("--potrace_O", type=float, default=0.2, help=" coners control of potrace")
    parser.add_argument("--potrace_a", type=float, default=1, help=" line control of potrace")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--no_copy_masks", action='store_true', help=" don't copy masks")
    parser.add_argument("--no_bg_seg", action='store_true', help=" don't do seg on sam background")
    parser.add_argument("--no_clamp_gradient", action='store_true', help=" don't clamp gradient paras")
    parser.add_argument("--no_double_seg", action='store_true', help=" don't use muti-seg")
    parser.add_argument("--no_use_hq_sam", action='store_true', help=" don't use hq sam")
    parser.add_argument("--once_unsam", action='store_true', help=" only use unsam to seg once")
    parser.add_argument("--no_use_unsam_start", action='store_true',
                         help=" don't double seg start from unsam additionally")
    parser.add_argument("--second_sam", action='store_true',
                         help=" use sam to seg twice")
    parser.add_argument("--seg_complexity", type=float,
                        help=" mask complexity above this value will be segmented again  ")
    parser.add_argument("--blur_type", type=str, help=" detail mask filtered way")
    parser.add_argument("--loss_type", type=str, help=" opt loss type")


    parser.add_argument("--simplify_eps_ratio", type=float, default=0.0, help="RDP threshold, 0 means no simplification")

    parser.add_argument("--color_type", type=int, default=2, help="fill color types: normal 0, linear gradient 1, "
                                                                "radial gradient 2, adaptive 3")
    parser.add_argument("--stop_number", type=int, default=2, help="offset number")
    parser.add_argument("--target", type=str, help="target image path")
    parser.add_argument("--input_images_dir", type=str, help="batch input dir")
    parser.add_argument("--num_iters", type=int, default=200, help="number of iterations")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("--shape_num", type=int, help="fixed shape number")
    parser.add_argument("--detail_ratio", type=float, default=0.5, help=" detail vector ratio among total vectors")
    parser.add_argument("--stroke_enable", action='store_true', default=False, help=" stroke training enable")
    parser.add_argument("--offsets_enable", action='store_true', default=False, help=" offset training enable")
    parser.add_argument("--base_loss", type=str, default="MSE_RGBA", help="base loss")
    parser.add_argument("--device", type=str, help="gpu or cpu")
    parser.add_argument("--metrics",action='store_true', default=False, help=" exp analyse")
    args = parser.parse_args()
    cfg = file_utils.read_config(args.config)
    cfg["metrics"] = args.metrics
    cfg["basic"]["exp_name"] = args.exp_name
    cfg["basic"]["seed"] = args.seed
    cfg["basic"]["input_image"] = args.target
    cfg["optimization"]["color_type"] = args.color_type
    cfg["optimization"]["stop_number"] = args.stop_number
    cfg["vtracer"]["c"] = args.vtracer_c
    cfg["vtracer"]["s"] = args.vtracer_s
    cfg["vtracer"]["simplify_eps_ratio"] = args.simplify_eps_ratio


    cfg["potrace"]["O"] = args.potrace_O
    cfg["potrace"]["a"] = args.potrace_a
    if args.input_images_dir is not None:
        cfg["basic"]["input_images_dir"] = args.input_images_dir
    if args.target is not None:
        cfg["basic"]["input_image"] = args.target
        cfg["basic"]["input_images_dir"] = None
    cfg["optimization"]["num_iters"] = args.num_iters
    cfg["basic"]["output_dir"] = args.output_dir
    cfg["optimization"]["clamp_enable"] = not args.no_clamp_gradient
    if cfg["basic"]["device"] is None:
        cfg["basic"]["device"] = "cuda:0"
    if args.device is not None:
        cfg["basic"]["device"] = args.device
    if args.shape_num is not None:
        cfg["basic"]["shape_num"] = args.shape_num
    if args.offsets_enable:
        cfg["optimization"]["offsets_enable"] = args.offsets_enable
    if args.no_double_seg:
        cfg["basic"]["double_seg"] = False
    if args.no_use_hq_sam:
        cfg["basic"]["use_hq_sam"] = False
    if args.no_copy_masks:
        cfg["double_seg"]["copy"] = False
    if args.once_unsam:
        cfg["unsam"]["use"] = True
    if args.no_use_unsam_start:
        cfg["double_seg"]["use_unsam_start"] = False
    if args.no_bg_seg:
        cfg["double_seg"]["bg_seg"] = False
    if args.second_sam:
        cfg["double_seg"]["second_sam"] = True
    if args.seg_complexity is not None:
        cfg["double_seg"]["seg_complexity"] = args.seg_complexity
    if args.blur_type is not None:
        cfg["detail"]["blur_type"] = args.blur_type
    if args.loss_type is not None:
        cfg["optimization"]["loss"]["base"] = args.loss_type
    cfg["basic"]["detail_ratio"] = args.detail_ratio
    cfg["optimization"]["stroke_enable"] = args.stroke_enable


    return cfg


def read_image(path, to_rgb=True, bg_color="white"):
    """读取图像，支持 P 模式和 RGBA，返回 RGB 或 RGBA numpy，范围 [0, 1]"""
    img = Image.open(path)

    if img.mode == 'P':
        # 转换调色板图像为 RGBA
        img = img.convert('RGBA')

    if img.mode == 'RGBA' and to_rgb:
        # RGBA -> RGB，混合背景
        background = Image.new("RGB", img.size, bg_color)
        background.paste(img, mask=img.split()[3])  # alpha 混合
        img = background
    elif img.mode == 'RGB' and not to_rgb:
        # RGB -> RGBA，补全 alpha 通道为全 1
        alpha = Image.new("L", img.size, 255)  # 全不透明
        img.putalpha(alpha)
    elif img.mode != 'RGB' and not to_rgb:
        # 其他模式统一转为 RGBA
        img = img.convert('RGBA')

    return np.asarray(img).astype(np.float32) / 255.0






def compute_mse(img1, img2):
    """
    计算两个图像之间的均方误差（MSE），自动处理图像大小和数据类型。

    参数：
    - img1: numpy.ndarray，参考图像，范围 [0,1] 或 [0,255]
    - img2: numpy.ndarray，待比较图像

    返回：
    - float，均方误差
    """
    # 类型和范围标准化
    def to_uint8(img):
        img = np.clip(img, 0, 1) if img.dtype == np.float32 else img
        return (img * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8)

    if img1.shape != img2.shape:
        img2_uint8 = to_uint8(img2)
        try:
            img2_pil = Image.fromarray(img2_uint8)
        except Exception as e:
            raise ValueError(f"无法转换 img2 为 PIL 图像，shape={img2.shape}, dtype={img2.dtype}") from e

        img2_resized = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.BILINEAR)
        img2 = np.asarray(img2_resized).astype(np.float32) / 255.0

    # 再次确保类型匹配
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    assert img1.shape == img2.shape, f"图像尺寸不一致: {img1.shape} vs {img2.shape}"
    return np.mean((img1 - img2) ** 2)

def compute_mse_rgba(img1, img2, premultiplied=False):
    """
    计算两个图像（RGB或RGBA）之间的均方误差（MSE）。

    参数：
    - img1, img2: numpy.ndarray，shape=(H,W,3) 或 (H,W,4)，范围 [0,1] 或 [0,255]
    - premultiplied: bool, 是否对RGB通道进行预乘 alpha 计算 MSE（只对RGBA有效）

    返回：
    - float，MSE
    """
    # 类型和范围标准化
    def to_float(img):
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        return img

    img1 = to_float(img1)
    img2 = to_float(img2)

    # 尺寸对齐
    if img1.shape != img2.shape:
        img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
        img2_resized = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.BILINEAR)
        img2 = np.asarray(img2_resized).astype(np.float32) / 255.0

    # 预乘处理
    if premultiplied and img1.shape[2] == 4:
        img1_rgb = img1[:, :, :3] * img1[:, :, 3:4]
        img2_rgb = img2[:, :, :3] * img2[:, :, 3:4]
        img1 = np.concatenate([img1_rgb, img1[:, :, 3:4]], axis=2)
        img2 = np.concatenate([img2_rgb, img2[:, :, 3:4]], axis=2)

    assert img1.shape == img2.shape, f"图像尺寸不一致: {img1.shape} vs {img2.shape}"
    return np.mean((img1 - img2) ** 2)


def compute_psnr(img1, img2):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    assert img1.shape == img2.shape
    return ssim(img1, img2, channel_axis=-1, data_range=1.0)


def compute_lpips(img1, img2):
    # 转为 tensor，形状为 (1, 3, H, W)
    def to_tensor(img):
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0) * 2 - 1  # [-1,1]
        return img.float()

    t1 = to_tensor(img1)
    t2 = to_tensor(img2)
    with torch.no_grad():
        dist = lpips_model(t1, t2)
    return dist.item()




# 计算RGBA的白色联通域
def count_connected_components(image_path):
    # 读取图像
    image = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)

    # 二值化图像，确保是黑白的
    # 这里使用128作为阈值，你可以根据需要调整
    binary_image = image >= 254

    # 使用scipy的ndimage.label来标记连通域
    labeled_array, num_features = ndimage.label(binary_image)
    cv2.imwrite("test_code/white_regions_binary.png", binary_image * 255)
    # 返回连通域的数量
    return num_features


def compute_vbq(target_img_path, render_img_path, sketch_img_path, n_ctrpts):
    # 读取图像
    target = read_image(target_img_path)
    render = read_image(render_img_path)

    psnr = compute_psnr(target, render)
    n_conn = count_connected_components(sketch_img_path)

    combined_value = n_ctrpts * n_conn
    vbq = log10(psnr * combined_value + 1e-5)  # 防止 log(0)

    return vbq

def create_background_path(canvas_width, canvas_height):
    """
    创建一个闭合的贝塞尔曲线矩形，模拟背景，id 格式如 shape_0。
    """
    # 矩形四个顶点 (顺时针)
    p0 = [0.0, 0.0]
    p1 = [canvas_width, 0.0]
    p2 = [canvas_width, canvas_height]
    p3 = [0.0, canvas_height]

    # 每条边是 1 段贝塞尔曲线 (2 控制点)
    points = torch.tensor([
        p0, p0, p1,  # 左上 -> 右上
        p1, p1, p2,  # 右上 -> 右下
        p2, p2, p3,  # 右下 -> 左下
        p3, p3, p0,  # 左下 -> 左上 (闭合)
    ], dtype=torch.float32)

    num_segments = 4
    num_control_points = torch.LongTensor([2] * num_segments)

    background_path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=points,
        stroke_width=torch.tensor(0.0),
        is_closed=True,
    )

    return background_path

def add_white_background(shapes, shape_groups, canvas_width, canvas_height):
    """
    在最前面插入白色背景 shape + shape_group，并修正所有原有 shape_id。
    """
    # 创建背景矩形
    background_path = create_background_path(canvas_width, canvas_height)

    # 1. 在 shapes 开头插入
    shapes.insert(0, background_path)

    # 2. 修正原来所有 shape_groups 的 id：通通 +1
    for sg in shape_groups:
        sg.shape_ids = sg.shape_ids + 1

    # 3. 创建背景 ShapeGroup（新的 index = 0）
    background_group = pydiffvg.ShapeGroup(
        shape_ids=torch.LongTensor([0]),
        fill_color=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        use_even_odd_rule=True,
        stroke_color=None,
        shape_to_canvas=torch.eye(3),
    )

    # 4. 插到最前面
    shape_groups.insert(0, background_group)

    return shapes, shape_groups

def remove_white_background(shapes, shape_groups):
    """
    删除 shapes[0] 和 shape_groups[0]（白色背景），
    并同步修正所有 shape_group 的 shape_ids（整体 -1）。
    """

    # 1. 删除背景 shape 和 shape_group
    shapes = shapes[1:]
    shape_groups = shape_groups[1:]

    # 2. 所有剩余 shape_group 的 shape_ids -= 1
    for sg in shape_groups:
        sg.shape_ids = sg.shape_ids - 1

    return shapes, shape_groups
