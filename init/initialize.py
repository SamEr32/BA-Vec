import time
import cv2
import numpy as np
import os
import json
import subprocess
from typing import List, Dict, Tuple
from itertools import cycle, islice
import copy
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
import utils.init_utils as init_utils
import utils.file_utils as file_utils
from utils.file_utils import get_png_name_without_suffix, get_all_png_images, delete_files, ensure_clean_dir, \
    read_config, json_save_info, delete_if_empty, batch_input
from utils.init_utils import find_largest_shape
from init.structure import structure_seg
from init.detail import detail_capture


def create_radial_gradient(shape, rgba, stop_n=2):
    """
    使用传入的颜色列表创建径向渐变，不再自动生成颜色。
    :param shape: pydiffvg.Shape 实例，具有 .points 属性
    :param rgba: list[list[float]]，长度=stop_n，每项=[R,G,B,A] (0-255)
    :param stop_n: 渐变停靠点数
    """
    import torch
    import pydiffvg

    assert len(rgba) == stop_n, f"传入的颜色数量({len(rgba)})应与 stop_n({stop_n}) 一致"
    assert stop_n >= 2, f"渐变颜色至少需要两种"

    # 归一化到 [0,1]
    stop_colors = torch.tensor(
        [[r / 255.0, g / 255.0, b / 255.0, a / 255.0] for (r, g, b, a) in rgba],
        dtype=torch.float32,
        requires_grad=True
    )

    # 均匀分布的 offset
    offsets = torch.linspace(0.0, 1.0, steps=stop_n, dtype=torch.float32, requires_grad=True)

    # 获取边界框
    min_point = torch.min(shape.points, dim=0).values
    max_point = torch.max(shape.points, dim=0).values

    # 中心和半径
    center = ((min_point + max_point) / 2.0).detach().clone().requires_grad_()
    radius = ((max_point - min_point) / 2.0).detach().clone().requires_grad_()

    return pydiffvg.RadialGradient(
        center=center,
        radius=radius,
        offsets=offsets,
        stop_colors=stop_colors
    )


def create_linear_gradient(shape, rgba, stop_n=2):
    """
    使用传入的颜色列表创建线性渐变，不再生成或修改颜色。
    :param shape: pydiffvg.Shape 实例
    :param rgba: list[list[float]]，长度=stop_n，每项=[R,G,B,A] (0-255)
    :param stop_n: 渐变停靠点数
    """
    import torch
    import pydiffvg

    assert len(rgba) == stop_n, f"传入的颜色数量({len(rgba)})应与 stop_n({stop_n}) 一致"
    assert stop_n >= 2, f"渐变颜色至少需要两种"

    stop_colors = torch.tensor(
        [[r / 255.0, g / 255.0, b / 255.0, a / 255.0] for (r, g, b, a) in rgba],
        dtype=torch.float32,
        requires_grad=True
    )
    offsets = torch.linspace(0.0, 1.0, steps=stop_n, dtype=torch.float32, requires_grad=True)

    # 获取边界框信息
    min_point = torch.min(shape.points, dim=0).values
    max_point = torch.max(shape.points, dim=0).values
    center = (min_point + max_point) / 2.0
    radius = torch.min(max_point - min_point) / 2.0  # 半径取短边一半

    begin = (center - torch.tensor([radius, 0.0], dtype=torch.float32)).detach().clone().requires_grad_()
    end = (center + torch.tensor([radius, 0.0], dtype=torch.float32)).detach().clone().requires_grad_()

    return pydiffvg.LinearGradient(
        begin=begin,
        end=end,
        offsets=offsets,
        stop_colors=stop_colors
    )


def merge_svg_ordered(svg_dir: str, output_svg_path: str, colors, stroke_enable: bool,
                      stop_num: int, fixed_color_type: int, filter=1, mask_num=999):
    svg_files = [f for f in os.listdir(svg_dir) if f.startswith("mask_") and f.endswith(".svg")]
    svg_files_sorted = sorted(svg_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    svg_paths = [os.path.join(svg_dir, f) for f in svg_files_sorted]
    set_stop_num = stop_num
    # print(f"📂 Found {len(svg_paths)} SVG files to merge.")
    all_shapes = []
    all_shape_groups = []
    initial_shapes_len = 0
    # 取第一个 SVG 的画布大小作为输出画布大小
    if len(svg_paths) != 0:
        canvas_width, canvas_height, _, _ = pydiffvg.svg_to_scene(svg_paths[0])
    else:
        raise ValueError("No details ??")

    shape_id_offset = 0
    # for idx, svg_file in enumerate(svg_paths):
    #     print(colors[1][idx])
    #     print(colors[0][idx])
    for idx, svg_file in enumerate(svg_paths):
        # print(f"🔗 Loading {svg_file}")
        _, _, shapes, _ = pydiffvg.svg_to_scene(svg_file)
        initial_shapes_len += len(shapes)
        # 如果单个的svg由多个shapes：（1）sam分割的掩码不连续，产生多个连通域，闭运算无法消除，或者闭运算创造了新的小孔
        #                        （2）捕获的细节经过闭运算后无法消除矢量化过程的断裂，或者闭运算创造了新的小孔

        # 如果只要一个但是shapes又有很多那就选面积最大的
        if len(shapes) > 1 and filter == 1:
            print(f"📌 {svg_file} contains {len(shapes)} shapes, choose the biggest one!")
            shapes = [find_largest_shape(shapes)]

        # 此处有两个相似变量 一个来自配置的颜色模式 一个是自适应计算出的颜色模式，两个变量会进行融合成一个来指导颜色初始化
        rgba_detail = colors[0][idx]

        # print(f"{idx}:{rgba}")
        # 如果是指定类型 0不渐变 1线性渐变 2径向渐变
        if 2 >= fixed_color_type >= 0:
            shape_color_type = fixed_color_type
        # 指定自适应就使用计算出的颜色类型（确保计算RGBA颜色的部分开启了自适应模式）
        else:
            shape_color_type = colors[1][idx]

        # 保留顺序遵循顺序 处理指定数量的shape
        for index, shape in enumerate(shapes):
            if index < filter:
                # c_l = len(rgba_detail)
                # print(f"{stop_num} {c_l} {shape_color_type} {set_stop_num}")
                # 纯色修正

                # 设置为0则在最大停靠数量与纯色间自适应
                if set_stop_num == 0:
                    stop_num = len(rgba_detail)
                    if len(rgba_detail) == 1:
                        shape_color_type = 0
                else:
                    if len(rgba_detail) < set_stop_num:
                        rgba_detail = list(islice(cycle(rgba_detail), set_stop_num))
                    stop_num = set_stop_num



                # 不渐变
                if shape_color_type == 0:
                    r, g, b, a = [v / 255.0 for v in rgba_detail[0]]
                    color = torch.tensor([r, g, b, a], dtype=torch.float32, requires_grad=True)
                # 线性渐变
                elif shape_color_type == 1:
                    color = create_linear_gradient(shape, rgba_detail, stop_num)
                # 径向渐变
                elif shape_color_type == 2:
                    color = create_radial_gradient(shape, rgba_detail, stop_num)
                # 其他
                elif shape_color_type == 3:
                    print("wait >..< !!")
                else:
                    raise ValueError(f"{svg_file}: {idx} shape, Color type err!")
                if stroke_enable:
                    shape.stroke_width = torch.tensor(0.5)

                    shape_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([shape_id_offset]),
                        fill_color=color,
                        stroke_color=torch.tensor([
                            random.uniform(0.0, 1.0),
                            random.uniform(0.0, 1.0),
                            random.uniform(0.0, 1.0),
                            1.0  # alpha 固定为不透明
                        ])
                    )
                else:
                    shape_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([shape_id_offset]),
                        fill_color=color
                    )
                all_shapes.append(shape)
                all_shape_groups.append(shape_group)
                shape_id_offset += 1
    # ====== 在主循环结束后添加 ======

    current_num = len(all_shapes)

    if current_num < mask_num and current_num > 0:
        print(f"⚠ Only {current_num} shapes, padding to {mask_num}")

        base_shapes = all_shapes.copy()
        base_groups = all_shape_groups.copy()

        idx = 0
        while len(all_shapes) < mask_num:
            # 轮流复制已有 shape
            src_shape = base_shapes[idx % len(base_shapes)]
            src_group = base_groups[idx % len(base_groups)]

            # ---- 深拷贝 shape ----
            new_shape = copy.deepcopy(src_shape)

            # ---- 深拷贝颜色 ----
            new_fill = copy.deepcopy(src_group.fill_color)

            new_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([shape_id_offset]),
                fill_color=new_fill
            )

            if stroke_enable and hasattr(src_group, "stroke_color"):
                new_group.stroke_color = copy.deepcopy(src_group.stroke_color)

            all_shapes.append(new_shape)
            all_shape_groups.append(new_group)

            shape_id_offset += 1
            idx += 1
    pydiffvg.save_svg(output_svg_path, canvas_width, canvas_height, all_shapes, all_shape_groups)

    # print(f"✅ Merged SVG saved to {output_svg_path}")

    # print(f"Total {len(all_shapes)} shapes, {len(all_shape_groups)} shape groups after merge.")

    return all_shapes, all_shape_groups, canvas_width, canvas_height, initial_shapes_len


# 合并一张粗糙阶段结果与一张新捕获的细节
def merge_structure_and_detail(save_path, structure_svg, detail_svg):
    # 读取结构图层 SVG
    canvas_width, canvas_height, structure_shapes, structure_shape_groups = pydiffvg.svg_to_scene(structure_svg)

    # 读取细节图层 SVG（忽略其画布大小）
    _, _, detail_shapes, detail_shape_groups = pydiffvg.svg_to_scene(detail_svg)

    # shape_id 从结构图层的数量开始递增，避免冲突
    shape_id_offset = len(structure_shapes)

    # 更新 detail_shape_groups 的 shape_ids
    for i, group in enumerate(detail_shape_groups):
        group.shape_ids = torch.tensor([id.item() + shape_id_offset for id in group.shape_ids])

    # 合并 shape 和 shape_group
    all_shapes = structure_shapes + detail_shapes
    all_shape_groups = structure_shape_groups + detail_shape_groups
    pydiffvg.save_svg(save_path, canvas_width, canvas_height, all_shapes, all_shape_groups)
    return all_shapes, all_shape_groups, canvas_width, canvas_height, shape_id_offset


# 合并指定图像的结构和细节
def merge_structure_and_detail_svgs(image_names, cfg):
    path = cfg["basic"]["output_dir"]

    # 合并结果用于细节优化
    detail_opt_init_svgs = []
    id_offsets = []
    for index, image_name in enumerate(image_names):
        # 确保完成前面的步骤
        structure_svg_path = f"{path}/{image_name}/structure/structure_opt_output.svg"
        detail_svg_path = f"{path}/{image_name}/detail/detail_init.svg"
        now_json_file_name = f"{path}/{image_name}/info.json"
        if os.path.exists(now_json_file_name):
            with open(now_json_file_name, 'r', encoding='utf-8') as f:
                now_json_info = json.load(f)
        else:
            raise FileNotFoundError(now_json_file_name)
        if not os.path.exists(structure_svg_path):
            raise FileNotFoundError(
                f"{index}/{len(image_names)}: Finish former steps: struct-cons of {image_name} please!")
        if now_json_info["detail"]["initial_shapes_len"] == 0:
            id_offsets.append(0)
            detail_opt_init_svgs.append([])
            file_utils.copy_and_rename(structure_svg_path, f"{path}/{image_name}/detail", "detail_opt_output.svg")
            file_utils.copy_and_rename(f"{path}/{image_name}/structure/structure_opt_output.png",
                                       f"{path}/{image_name}/detail", "detail_opt_output.png")

            continue
        else:
            if not os.path.exists(detail_svg_path):
                raise FileNotFoundError(
                    f"{index}/{len(image_names)}: Finish former steps: detail-cap of {image_name} please!")
            else:
                save_path = f"{path}/{image_name}/detail/detail_opt_input.svg"
                shapes, shape_groups, canvas_width, canvas_height, id_offset = merge_structure_and_detail(save_path,
                                                                                                          structure_svg_path,
                                                                                                          detail_svg_path)
                id_offsets.append(id_offset)
                json_save_info(cfg["save"]["json"], now_json_file_name, now_json_info)
                detail_opt_init_svgs.append([shapes, shape_groups, canvas_width, canvas_height])
    print("merge_structure_and_detail finished successfully! ")
    return detail_opt_init_svgs, id_offsets


# 通过合并结构矢量完成结构矢量的初始化
def structure_init(cfg, flag_ctl=False):
    output_top_dir = cfg["basic"]['output_dir']
    stroke_enable = cfg["optimization"]["stroke_enable"]
    color_type = cfg["optimization"]["color_type"]
    stop_num = cfg["optimization"]["stop_number"]
    json_save = cfg["save"]["json"]
    colors_detail, image_names, json_file_name, str_mask_num = structure_seg(cfg)

    json_info, json_file_name = file_utils.read_top_json(cfg)

    total_merge_time_s = time.time()
    structure_svgs = []
    for i, image_name in enumerate(image_names):
        now_merge_time_s = time.time()
        now_image_path = f"{output_top_dir}/{image_name}"
        now_json_file_name = f"{now_image_path}/info.json"
        # 从 JSON 文件中读取数据
        if os.path.exists(now_json_file_name):
            with open(now_json_file_name, 'r', encoding='utf-8') as f:
                now_json_info = json.load(f)
        else:
            raise FileNotFoundError(now_json_file_name)
        if now_json_info.get('structure') is not None and flag_ctl:
            if now_json_info['structure'].get('merge_time') is not None:
                continue

        now_image_structure_path = f"{now_image_path}/structure"
        now_image_structure_mask_path = f"{now_image_structure_path}/mask"
        struc_merge_res = f"{now_image_structure_path}/structure_init.svg"
        shapes, shape_groups, canvas_width, canvas_height, initial_shapes_len = merge_svg_ordered(
            now_image_structure_mask_path, struc_merge_res, colors_detail[i], stroke_enable, stop_num, color_type, mask_num=str_mask_num)
        if not cfg["save"]["sam"]["sam_svg"]:
            delete_files(now_image_structure_mask_path, ".svg")
        delete_if_empty(now_image_structure_mask_path)
        structure_svgs.append([shapes, shape_groups, canvas_width, canvas_height])
        now_merge_time_e = time.time()
        now_merge_time = now_merge_time_e - now_merge_time_s
        now_json_info["structure"]["merge_time"] = now_merge_time
        now_json_info["structure"]["initial_shapes_len"] = initial_shapes_len
        now_json_info["structure"]["final_shapes_len"] = len(shapes)
        json_save_info(json_save, now_json_file_name, now_json_info)

    total_merge_time_e = time.time()
    total_merge_time = total_merge_time_e - total_merge_time_s

    json_info["structure"]["total_merge_time"] = total_merge_time
    json_info["structure"]["avg_merge_time"] = total_merge_time / len(image_names)
    json_save_info(json_save, json_file_name, json_info)
    return structure_svgs, image_names


# 通过合并捕获的细节完成细节矢量的初始化
def detail_init(cfg):
    if cfg is None:
        raise ValueError("config is required!")

    output_top_dir = cfg["basic"]["output_dir"]
    input_dir = cfg["basic"]["input_image"]
    input_batch_dir = cfg["basic"]["input_images_dir"]
    optimization = cfg["optimization"]
    stroke_enable = optimization["stroke_enable"]
    stop_num = optimization["stop_number"]
    color_type = optimization["color_type"]
    detail_cfg = cfg["detail"]
    ratio = cfg["basic"]["detail_ratio"]
    json_save = cfg["save"]["json"]
    detail_save = cfg["save"]["detail"]
    O = cfg["potrace"]["O"]
    a = cfg["potrace"]["a"]
    area_weight = detail_cfg["area_weight"]
    perimeter_weight = detail_cfg["perimeter_weight"]
    min_area_ratio = detail_cfg["min_area_ratio"]
    min_perimeter_ratio = detail_cfg["min_perimeter_ratio"]
    blur_type = detail_cfg["blur_type"]
    base_size = cfg["basic"]["base_size"]
    top_json_info, top_json_path = file_utils.read_top_json(cfg)

    # 获取所有要处理的图片
    images_path = batch_input(input_batch_dir, input_dir)
    image_names = []
    detail_svgs = []
    total_time = 0
    total_merge_time = 0
    total_initial_detail_len = 0
    total_filtered_detail_len = 0
    # 得到所有的colors
    for i, image_path in enumerate(images_path):
        detail_time_s = time.time()
        # 目标图像名称 用于生成文件夹名称
        image_name = get_png_name_without_suffix(image_path)
        image_names.append(image_name)
        now_json_file_name = f"{output_top_dir}/{image_name}/info.json"
        #  json
        if os.path.exists(now_json_file_name):
            with open(now_json_file_name, 'r', encoding='utf-8') as f:
                now_json_info = json.load(f)
        else:
            raise FileNotFoundError(now_json_file_name)
        now_json_info["detail"] = {}
        print(f"{i + 1}/{len(images_path)}: {image_name} under detail processing...")
        output_dir = f"{output_top_dir}/{image_name}"
        structure_opt_res_path = f"{output_dir}/structure/structure_opt_output.png"
        if not os.path.exists(structure_opt_res_path):
            raise FileNotFoundError(f"{structure_opt_res_path} miss, do the structure optimization first!")
        # 矢量数量分配
        if cfg["basic"]["shape_num"] is None:
            if cfg["detail"]["detail_num"] is not None:
                detail_num = cfg["detail"]["detail_num"]
            else:

                detail_num = 2048
        else:

            # 读出结构矢量的不合格矢量数量 和总数
            struct_bad_shape_num, struct_shape_num = init_utils.count_low_alpha_primitives(
                f"{output_dir}/structure/structure_opt_output.svg", None)
            # # 总数量-结构矢量数量 + 不合格矢量数量
            # detail_num = cfg["basic"]["shape_num"] - struct_shape_num + struct_bad_shape_num
            detail_num = cfg["basic"]["shape_num"] - struct_shape_num

        if detail_num > 0:

            res = detail_capture(
                image_path, structure_opt_res_path,
                output_dir, detail_save, now_json_info, area_weight, perimeter_weight,
                cfg["optimization"]["max_stop_number"], cfg["optimization"]["stop_number"], O, a,
                detail_num, min_area_ratio, min_perimeter_ratio, blur_type, base_size, ratio, color_type,
                device=cfg["basic"]["device"], copy=cfg["double_seg"]["copy"],c=cfg["vtracer"]["c"],s=cfg["vtracer"]["s"],simplify_eps_ratio=cfg["vtracer"]["simplify_eps_ratio"]
            )
        else:
            res = False
        if res:
            colors, color_types, now_json_info = res

            detail_time_e = time.time()
            detail_time = detail_time_e - detail_time_s
            total_time += detail_time
            now_json_info["detail"]["detail_capture_time"] = detail_time
            total_initial_detail_len += now_json_info["detail"]["initial_contours_len"]
            total_filtered_detail_len += now_json_info["detail"]["filtered_contours_len"]
            now_image_detail_mask_path = f"{output_dir}/detail/mask"
            detail_merge_res = f"{output_dir}/detail/detail_init.svg"
            detail_merge_s = time.time()
            shapes, shape_groups, canvas_width, canvas_height, initial_shapes_len = merge_svg_ordered(
                now_image_detail_mask_path, detail_merge_res, [colors, color_types], stroke_enable, stop_num,
                color_type,mask_num=detail_num)
            detail_svgs.append([shapes, shape_groups, canvas_width, canvas_height])
            detail_merge_e = time.time()
            detail_merge_time = detail_merge_e - detail_merge_s
            total_merge_time += detail_merge_time
            now_json_info["detail"]["initial_shapes_len"] = initial_shapes_len
            now_json_info["detail"]["final_shapes_len"] = len(shapes)
            now_json_info["detail"]["detail_merge_time"] = detail_merge_time

            if not detail_save["detail_svg"]:
                file_utils.delete_files(now_image_detail_mask_path, ".svg")
                file_utils.delete_if_empty(now_image_detail_mask_path)
            delete_if_empty(f"{output_dir}/detail/difference")
        else:
            now_json_info["detail"]["initial_shapes_len"] = 0
            now_json_info["detail"]["final_shapes_len"] = 0
            now_json_info["detail"]["detail_merge_time"] = 0
            detail_svgs.append([])

        json_save_info(json_save, now_json_file_name, now_json_info)

    top_json_info["detail"] = {
        "total_detail_capture_time": total_time,
        "avg_detail_capture_time": total_time / len(images_path),
        "avg_initial_len": total_initial_detail_len / len(images_path),
        "avg_filtered_len": total_filtered_detail_len / len(images_path),
        "total_merge_time": total_merge_time,
        "avg_merger_time": total_merge_time / len(images_path),
    }
    json_save_info(json_save, top_json_path, top_json_info)

    return detail_svgs, image_names
