from pathlib import Path
import time
import cv2
import numpy as np
import os
from PIL import Image
import json
import subprocess
from typing import List, Dict, Tuple
import yaml
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from xml.etree import ElementTree as ET
import glob
import re
import shutil
import pydiffvg
import skimage.io
from typing import List
import random
from datetime import datetime
import os.path as osp
import os
import psutil, os




# === 将掩码转换为 SVG ===
def mask_to_svg(png_path:str, svg_path: str,c=100,s=80):
    import vtracer

    """
    mask: 二值 numpy 数组，True/1=前景, False/0=背景
    svg_path: 输出 SVG 路径
    """

    vtracer.convert_image_to_svg_py(
        png_path,  # 输入图像路径
        svg_path,  # 输出 SVG 路径
        colormode='binary',
        filter_speckle=2,
        corner_threshold=c,
        length_threshold=2,
        splice_threshold=s,  # 拼接更多平滑路径，减少分段感
        max_iterations=20,

    )


# 复制重命名
def copy_and_rename(src: str, dst_dir: str, new_name: str):
    """
    使用 pathlib 复制并重命名文件。
    """
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / new_name
    shutil.copy2(src, dst_path)
    return dst_path

# 读取批次json
def read_top_json(config, first_flag=False, path="../exp/ours"):
    input_batch_dir = config["basic"]["input_images_dir"]
    if input_batch_dir is None or input_batch_dir == "" or input_batch_dir == " ":
        # print(input_batch_dir)
        # print(config["basic"]["input_image"])
        input_batch_dir = os.path.dirname(config["basic"]["input_image"])

    dataset_path = input_batch_dir
    if config["basic"]["shape_num"] is not None:
        shape_num = config["basic"]["shape_num"]
    else:
        shape_num = "free"
    if config["basic"]["exp_name"] is not None:
        exp_name = config["basic"]["exp_name"]
    else:
        exp_name = "null"
    norm_path = os.path.normpath(dataset_path)
    parts = norm_path.split(os.sep)  # 按路径分隔
    last_part = parts[-1]

    # 如果最后一级是 'png'，返回上一级（即第二级）
    if last_part.lower() == "png" and len(parts) >= 2:
        dataset_name = parts[-2]
    else:
        dataset_name = last_part
    if config["basic"]["modern"] is None:
        config["basic"]["modern"] = "test"
    if config["basic"]["modern"] == "exp":
        json_file_name = f"{path}/{dataset_name}_{shape_num}_{exp_name}.json"
    else:
        json_file_name = f"{input_batch_dir}/info.json"
    if not os.path.exists(json_file_name):
        print(f"{json_file_name} does not exist.")
        first_flag = True
    if first_flag:
        json_save_info(config["save"]["json"], json_file_name, {"change_time": ""})
        print("the JSON file created.")
    # 从 JSON 文件中读取数据
    if os.path.exists(json_file_name):
        with open(json_file_name, 'r', encoding='utf-8') as f:
            json_info = json.load(f)
    else:
        return None, None

    return json_info, json_file_name


# 删除空文件夹
def delete_if_empty(dir_path):
    """
    删除空文件夹。
    如果文件夹存在且为空，则将其删除。

    参数:
        dir_path (str): 要检查的文件夹路径。
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        if not os.listdir(dir_path):  # 文件夹为空
            os.rmdir(dir_path)
    else:
        print(f"路径不存在或不是文件夹: {dir_path}")


# 返回路径中的图像名称
def get_png_name_without_suffix(png_path):
    """
    从 PNG 文件路径中提取文件名（不带后缀）。

    参数:
        png_path (str): PNG 图片的完整路径。

    返回:
        str: 不带扩展名的图片名称。
    """
    return os.path.splitext(os.path.basename(png_path))[0]


# 返回路径下所有png图像
def get_all_png_images(directory):
    """
    获取指定目录下所有 PNG 图像的完整路径。

    参数:
        directory (str): 要搜索的文件夹路径。

    返回:
        List[str]: 所有 PNG 文件的完整路径列表。
    """
    png_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files


# 删除文件夹
def delete_folder(folder_path):
    """删除文件夹及其所有内容"""
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        # print(f"✅ 已删除文件夹: {folder_path}")


# 删除文件夹中指定后缀类型文件
def delete_files(folder_path, type):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    for filename in os.listdir(folder_path):
        if filename.endswith(f"{type}"):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


# 创建文件夹
def ensure_clean_dir(path):
    """
    确保指定目录是一个干净的空目录。
    如果目录存在，则删除其中所有内容。
    然后重新创建目录。
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# 读配置
def read_config(file_name):
    # with open('./config/config.yaml', 'r') as file:
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)

    return config


# 存json文件
def json_save_info(json_save, json_file_name, json_info):
    if json_save:
        if os.path.exists(json_file_name):
            os.remove(json_file_name)
        json_info["change_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(json_file_name, "w") as f:
            json.dump(json_info, f, ensure_ascii=False, indent=4)


def load_image(image_path: str, output_dir: str, bg: str, flag: bool):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 保留 alpha 通道（若有）
    assert image is not None, f"Image not found: {image_path}"

    # 保存路径
    sam_input_path = f"{output_dir}/sam_input.png"
    sam_mask_path = f"{output_dir}/sam_input_mask.png"

    # 删除旧文件
    for p in [sam_input_path, sam_mask_path]:
        if os.path.exists(p):
            os.remove(p)

    h, w = image.shape[:2]

    # =============== RGBA 输入 ===============
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32) / 255.0
        alpha = image[:, :, 3].astype(np.float32) / 255.0  # 0~1
        alpha_mask = (image[:, :, 3] > 0).astype(np.uint8)  # (H,W)
        alpha = np.expand_dims(alpha, axis=2)

        # 背景设定
        if bg == "white":
            bg_color = np.ones_like(bgr)
        else:
            bg_color = np.zeros_like(bgr)

        # alpha 合成
        bgr_composited = bgr * alpha + bg_color * (1 - alpha)
        bgr_composited = (bgr_composited * 255).astype(np.uint8)

        # 保存SAM输入与mask
        if flag:
            cv2.imwrite(sam_input_path, bgr_composited)
            cv2.imwrite(sam_mask_path, alpha_mask * 255)

        image_rgb = cv2.cvtColor(bgr_composited, cv2.COLOR_BGR2RGB)
        return bgr_composited, image_rgb, alpha_mask

    # =============== RGB 输入 ===============
    else:
        bgr = image.astype(np.uint8)
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # mask 全1
        alpha_mask = np.ones((h, w), dtype=np.uint8)

        # 保存
        if flag:
            cv2.imwrite(sam_input_path, bgr)
            cv2.imwrite(sam_mask_path, alpha_mask * 255)

        return bgr, image_rgb, alpha_mask

# 获取所有要处理的图像名称
def get_input_image_names(config):
    image_names = []
    input_batch_dir = config["basic"]["input_images_dir"]
    input_dir = config["basic"]["input_image"]
    images_path = batch_input(input_batch_dir, input_dir)
    for i, image_path in enumerate(images_path):
        image_name = get_png_name_without_suffix(image_path)
        image_names.append(image_name)
    return image_names


# 批处理判断
def batch_input(input_batch_dir, input_dir):
    images_path = []
    if input_batch_dir is None:
        if input_dir is None:
            raise ValueError("please input your image!")
        else:
            images_path.append(input_dir)
    else:
        images_path = get_all_png_images(input_batch_dir)
    # print(images_path)
    return images_path


# LIVE的工具包
def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)


# 训练过程保存成mp4
def save_mp4(num_iter, png_dir, video_name, w, h):
    print("saving iteration video...")
    img_array = []
    for ii in range(0, num_iter):
        # filename = os.path.join(
        #     png_dir, "video-png",
        #     "{}-iter{}.png".format(pathn_record_str, ii))
        filename = f"{png_dir}/iter_{ii}.png"
        img = cv2.imread(filename)
        img_array.append(img)

    # videoname = os.path.join(
    #     cfg.experiment_dir, "video-avi",
    #     "{}.avi".format(pathn_record_str))
    check_and_create_dir(video_name)
    out = cv2.VideoWriter(
        video_name,
        # cv2.VideoWriter_fourcc(*'mp4v'),
        cv2.VideoWriter_fourcc(*'FFV1'),
        20.0, (w, h))
    for iii in range(len(img_array)):
        out.write(img_array[iii])
    out.release()
    # shutil.rmtree(os.path.join(cfg.experiment_dir, "video-png"))


# 将 JPG 图像转换为 PNG 格式。
def convert_jpg_to_png(input_path, output_path=None):
    """
    将 JPG 图像转换为 PNG 格式。

    参数：
        input_path (str): 输入 JPG 图像路径。
        output_path (str, optional): 输出 PNG 图像路径。默认与输入路径相同，仅扩展名更改为 .png。

    返回：
        str: 实际保存的 PNG 图像路径。
    """
    if not input_path.lower().endswith(".jpg") and not input_path.lower().endswith(".jpeg"):
        raise ValueError("输入文件必须是 JPG 格式")

    # 读取 JPG 图像
    img = Image.open(input_path).convert("RGB")

    # 自动生成输出路径
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".png"

    # 保存为 PNG
    img.save(output_path, format="PNG")
    return output_path
