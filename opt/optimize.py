import os
import time
import json
import torch
import xml.etree.ElementTree as ET
import pydiffvg
import torchvision.transforms.functional as TF
import imageio
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import skimage.io
from typing import List
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from utils.file_utils import batch_input, check_and_create_dir, save_mp4, json_save_info, read_config, ensure_clean_dir, \
    delete_files, get_all_png_images, delete_if_empty, delete_folder
from utils import opt_utils, file_utils, init_utils

gamma = 1.0
cuda = 'cuda:3'
device = ""
def check_render_valid(img, iter_id, w, h):
    """ 检查 diffvg 渲染结果是否正常。 img: HxWx4 tensor """
    if img is None:
        print(f"[RenderCheck][Iter {iter_id}] ❌ Render returned None")
        return False
    if torch.isnan(img).any():
        print(f"[RenderCheck][Iter {iter_id}] ❌ Found NaN in rendered image")
        return False
    if torch.isinf(img).any():
        print(f"[RenderCheck][Iter {iter_id}] ❌ Found Inf in rendered image")
        return False # 检查全 0、尺寸错误等
    if img.shape[0] != h or img.shape[1] != w:
        print(f"[RenderCheck][Iter {iter_id}] ❌ Wrong image resolution: {img.shape} vs {h}x{w}")
        return False
    if float(img.abs().sum()) == 0:
        print(f"[RenderCheck][Iter {iter_id}] ⚠️ Render result is all zeros")
        # 可能是场景序列化失败 / shape crash
        return False
    return True

# 初始化优化参数
def optimization_init(cfg):
    global gamma
    gamma = cfg["optimization"]["gamma"]
    global cuda
    cuda = cfg["basic"]["device"]
    if cuda != "cpu":
        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        pydiffvg.set_device(torch.device(cuda))
    else:
        pydiffvg.set_use_gpu(False)
        pydiffvg.set_device(torch.device(cuda))
    global device
    device = pydiffvg.get_device()


def load_and_process_image(input_image_path, bg_color="white", flag=False, gamma=1.0, save_path=None):
    img = Image.open(input_image_path)

    # 处理调色板或灰度图
    if img.mode == 'P':
        img = img.convert('RGBA')

    elif img.mode == 'L':
        img = img.convert('RGB')

    target = np.array(img)

    if target.ndim == 2:  # 灰度图
        target = np.stack([target] * 3, axis=-1)

    if target.shape[2] == 4 and not flag:
        rgb = target[:, :, :3].astype(np.float32) / 255.0
        alpha = target[:, :, 3].astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        bg = np.ones_like(rgb) if bg_color == "white" else np.zeros_like(rgb)
        target = rgb * alpha + bg * (1.0 - alpha)
    elif target.shape[2] == 4 and flag:
        target = target.astype(np.float32) / 255.0
        rgb = target[:, :, :3]
        alpha = target[:, :, 3:4]

        # 正确的 premultiplied
        rgb_premul = rgb * alpha

        target = np.concatenate([rgb_premul, alpha], axis=2)
    elif target.shape[2] == 3:
        target = target.astype(np.float32) / 255.0
    else:
        raise ValueError("Unsupported image format. Only RGB or RGBA supported.")

    target_tensor = torch.from_numpy(target).float().pow(gamma)
    target_tensor = target_tensor.to(pydiffvg.get_device())
    target_tensor = target_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # NHWC -> NCHW

    if save_path is not None:
        save_np = target_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        save_np = np.clip(save_np ** (1.0 / gamma), 0, 1)
        out_img = (save_np * 255).astype(np.uint8)

        # 自动判断保存模式
        if out_img.shape[2] == 4:
            imageio.imwrite(save_path, out_img)  # 保存 RGBA
        else:
            imageio.imwrite(save_path, out_img[:, :, :3])  # 保存 RGB
    return target_tensor


def enhanced_premult_loss(img, target, eps=1e-6, alpha_penalty_weight=10.0):
    # --------------------------------------------------------
    # 1. img: (H,W,4) → (1,4,H,W)
    # --------------------------------------------------------
    img_rgba = img.unsqueeze(0).permute(0, 3, 1, 2)  # (1,4,H,W)
    render_rgb = img_rgba[:, 0:3, :, :]
    render_alpha = img_rgba[:, 3:4, :, :]

    render_alpha = torch.clamp(render_alpha, 0.0, 1.0)
    render_alpha = torch.where(render_alpha < 1e-4,
                               torch.zeros_like(render_alpha),
                               render_alpha)

    # --------------------------------------------------------
    # 2. target 规范到 RGBA
    # --------------------------------------------------------
    if target.dim() == 3:
        target = target.unsqueeze(0)
    if target.shape[1] == 3:
        alpha_one = torch.ones(
            (target.shape[0], 1, target.shape[2], target.shape[3]),
            device=target.device, dtype=target.dtype
        )
        target_rgba = torch.cat([target, alpha_one], dim=1)
    else:
        target_rgba = target

    target_rgb = target_rgba[:, 0:3, :, :]
    target_alpha = torch.clamp(target_rgba[:, 3:4, :, :], 0.0, 1.0)
    target_alpha = torch.where(target_alpha < 1e-4,
                               torch.zeros_like(target_alpha),
                               target_alpha)

    # --------------------------------------------------------
    # 3. Premultiplied Alpha
    # --------------------------------------------------------
    premult_render = render_rgb * render_alpha
    premult_target = target_rgb * target_alpha

    # --------------------------------------------------------
    # 4. 颜色损失（核心部分）
    # --------------------------------------------------------
    diff = premult_render - premult_target
    denom = render_alpha.pow(2) + eps
    loss_map = diff.pow(2) / denom

    valid_mask = ((render_alpha > 0) | (target_alpha > 0)).float()
    valid_mask_rgb = valid_mask.expand_as(loss_map)

    if valid_mask.sum() < 1:
        color_loss = torch.tensor(0.0, device=diff.device)
    else:
        color_loss = (loss_map * valid_mask_rgb).sum() / (valid_mask_rgb.sum() + eps)

    # --------------------------------------------------------
    # 5. α 惩罚：目标透明、渲染不透明
    # --------------------------------------------------------
    target_transparent_mask = (target_alpha < 1e-4).float()
    alpha_penalty = (render_alpha * target_transparent_mask).pow(2).sum()

    if target_transparent_mask.sum() < 1:
        alpha_penalty = torch.tensor(0.0, device=diff.device)
    else:
        alpha_penalty = (
            alpha_penalty / (target_transparent_mask.sum() + eps)
        ) * alpha_penalty_weight

    # --------------------------------------------------------
    # 6. 总损失
    # --------------------------------------------------------
    return color_loss + alpha_penalty

# 优化一张svg
def optimize_svg(shapes, shape_groups, canvas_width, canvas_height, target, offsets_enable, stroke_enable,
                 num_iters, output_dir, decay_ratio, lr, loss_type, save, frozen_id=-1, background=True,
                 save_background=False, last_epoch=-1, clamp_enable=True):
    # 控制点和颜色
    points_vars = []
    color_vars = []
    # 停靠点
    offsets_vars = []
    # 描边
    stroke_color_vars = []
    stroke_width_vars = []
    # 我们将为渐变的坐标/半径创建“归一化参数”并加入 optimizer
    begin_norm_params = []  # list of (param, fill_ref) pairs
    end_norm_params = []
    center_norm_params = []
    radius_norm_params = []

    var = []
    if save["res"]:
        # 保存初始svg
        pydiffvg.save_svg(os.path.join(output_dir, "init.svg"),
                          canvas_width, canvas_height, shapes, shape_groups)
    this_time_loss_type = False
    # 判断目标类别 NCHW
    if target.shape[1] == 4:
        if loss_type["base"] != "MSE_RGBA" and loss_type["base"] != "MSE_PREMULT_RGBA_RGB":
            print("Base loss is RGB, but the target is RGBA")
        # RGB都可以用
    elif target.shape[1] == 3:
        if loss_type["base"] == "MSE_RGBA" or loss_type["base"] == "MSE_PREMULT_RGBA_RGB":

            print("Base loss is RGBA, but the target is RGB, change target to RGBA automatically!")
    else:
        raise ValueError("Unknown number of channels: {}".format(target.shape[1]))
    _, _, h, w = target.shape

    # --- collect geometry vars ---
    for s_id, shape in enumerate(shapes):
        if frozen_id < s_id:
            shape.points.requires_grad = True
            points_vars.append(shape.points)
            if stroke_enable and hasattr(shape, "stroke_width") and shape.stroke_width is not None:
                shape.stroke_width.requires_grad = True
                stroke_width_vars.append(shape.stroke_width)

    # --- collect fill / gradient vars, but create normalized params for gradient coords/radii ---
    # We'll need device for creating params
    device = pydiffvg.get_device() if hasattr(pydiffvg, "get_device") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    max_dim = float(max(canvas_width, canvas_height))

    for g_id, group in enumerate(shape_groups):
        fill = group.fill_color
        # 如果是线性渐变
        if isinstance(fill, pydiffvg.LinearGradient):
            if frozen_id < g_id:
                fill.stop_colors.requires_grad = True
                color_vars.append(fill.stop_colors)
            # keep original tensors but create normalized params (x/width, y/height)
            # initialize normalized begin/end from existing pixel coords
            with torch.no_grad():
                begin_pix = fill.begin.clone().to(device)
                end_pix = fill.end.clone().to(device)
            # create normalized torch Parameters
            begin_norm = torch.nn.Parameter(begin_pix / torch.tensor([canvas_width, canvas_height], device=device))
            end_norm = torch.nn.Parameter(end_pix / torch.tensor([canvas_width, canvas_height], device=device))
            # register for optimization
            begin_norm.requires_grad = True
            end_norm.requires_grad = True
            begin_norm_params.append((begin_norm, fill))
            end_norm_params.append((end_norm, fill))

        # 如果是径向渐变
        elif isinstance(fill, pydiffvg.RadialGradient):
            if frozen_id < g_id:
                fill.stop_colors.requires_grad = True
                color_vars.append(fill.stop_colors)
            # radius and center
            with torch.no_grad():
                center_pix = fill.center.clone().to(device)
                radius_pix = fill.radius.clone().to(device)
            # center normalized by (width, height), radius normalized by max_dim
            center_norm = torch.nn.Parameter(center_pix / torch.tensor([canvas_width, canvas_height], device=device))
            radius_norm = torch.nn.Parameter(radius_pix / torch.tensor([max_dim], device=device))
            center_norm.requires_grad = True
            radius_norm.requires_grad = True
            center_norm_params.append((center_norm, fill))
            radius_norm_params.append((radius_norm, fill))

        # 普通颜色
        else:
            if frozen_id < g_id:
                fill.requires_grad = True
                color_vars.append(fill)

        if offsets_enable and hasattr(fill, "offsets"):
            fill.offsets.requires_grad = True
            offsets_vars.append(fill.offsets)


        if stroke_enable:
            if hasattr(group, "stroke_color"):
                group.stroke_color.requires_grad = True
                stroke_color_vars.append(group.stroke_color)

    # --- learning rates ---
    points_lr = lr["points_lr"]
    color_lr = lr["color_lr"]
    offsets_lr = lr["offsets_lr"]
    begin_lr = lr["begin_lr"]
    end_lr = lr["end_lr"]
    stroke_width_lr = lr["stroke_width_lr"]
    stroke_color_lr = lr["stroke_color_lr"]
    radius_lr = lr["radius_lr"]
    center_lr = lr["center_lr"]
    # l = len(offsets_vars)
    # print(f"[DEBUG]: var {l} enable:{offsets_enable}")

    # pack optimizer param groups (use the raw tensors for most; normalized params are separate)
    if points_vars:
        var.append({'params': points_vars, 'lr': points_lr})
    if color_vars:
        var.append({'params': color_vars, 'lr': color_lr})
    if offsets_enable and offsets_vars:
        var.append({'params': offsets_vars, 'lr': offsets_lr})
    if stroke_enable and stroke_width_vars:
        var.append({'params': stroke_width_vars, 'lr': stroke_width_lr})
    if stroke_enable and stroke_color_vars:
        var.append({'params': stroke_color_vars, 'lr': stroke_color_lr})

    # add normalized params to optimizer with appropriate lrs
    if begin_norm_params:
        var.append({'params': [p for p, f in begin_norm_params], 'lr': begin_lr})
    if end_norm_params:
        var.append({'params': [p for p, f in end_norm_params], 'lr': end_lr})
    if center_norm_params:
        var.append({'params': [p for p, f in center_norm_params], 'lr': center_lr})
    if radius_norm_params:
        var.append({'params': [p for p, f in radius_norm_params], 'lr': radius_lr})
    if background:
        shapes, shape_groups = opt_utils.add_white_background(shapes, shape_groups, canvas_width, canvas_height)

    lrlambda_f = opt_utils.linear_decay_lrlambda_f(num_iters, decay_ratio)
    optimizer = torch.optim.Adam(var)
    scheduler = LambdaLR(
        optimizer, lr_lambda=lrlambda_f, last_epoch=last_epoch)
    render = pydiffvg.RenderFunction.apply
    t_range = tqdm(range(num_iters))

    # 如果要存mp4的话 一定要有png
    if save["opt_mp4"] or save["intermediate_png"]:
        ensure_clean_dir(f"{output_dir}/png")
    if save["intermediate_svg"]:
        ensure_clean_dir(f"{output_dir}/svg")
    opt_time_s = time.time()

    # Helper tensors for scaling back to pixel coords
    width_scale = torch.tensor([canvas_width, canvas_height], device=device)
    max_scale = torch.tensor([max_dim], device=device)

    for t in t_range:
        s = time.time()
        optimizer.zero_grad()

        # --- 在渲染前，把归一化参数放缩回像素并写回 fill 对象 ---
        # clamp normalized params to [0, 1] for safety, then scale back
        if clamp_enable:
            for p, fill in begin_norm_params:
                with torch.no_grad():
                    p.data.clamp_(0.0, 1.0)
                    val = (p * width_scale).to(fill.begin.device)
                    fill.begin.data.copy_(val)

            for p, fill in end_norm_params:
                with torch.no_grad():
                    p.data.clamp_(0.0, 1.0)
                    val = (p * width_scale).to(fill.end.device)
                    fill.end.data.copy_(val)

            for p, fill in center_norm_params:
                with torch.no_grad():
                    p.data.clamp_(0.0, 1.0)
                    val = (p * width_scale).to(fill.center.device)
                    fill.center.data.copy_(val)

            for p, fill in radius_norm_params:
                with torch.no_grad():
                    p.data.clamp_(0.0, 1.0)
                    # radius might be scalar-like; ensure shape matches
                    scaled = (p * max_scale).to(fill.radius.device)
                    fill.radius.data.copy_(scaled)

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
        if not check_render_valid(img,t,canvas_width,canvas_height):
            raise ValueError("retry this imag opt!")
        # 保存结果RGB
        if save["res"]:
            if t == num_iters - 1:
                # 最后一轮 final.png
                pydiffvg.imwrite(img.cpu(), f"{output_dir}/final.png", gamma=gamma)
            elif t == 0:
                pydiffvg.imwrite(img.cpu(), f"{output_dir}/init.png", gamma=gamma)
        # 如果要存MP4的话
        if save["opt_mp4"] or save["intermediate_png"]:
            pydiffvg.imwrite(img.cpu(), os.path.join(output_dir, f"png/iter_{t}.png"), gamma=gamma)
        if loss_type["base"] is None or loss_type["base"] == "MSE_RGB" :
            # 图像不处理的话默认是RGBA4维的
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                              device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
            img = img[:, :, :3]
            img = img.unsqueeze(0).permute(0, 3, 1, 2)  # NHWC -> NCHW
            loss = (img - target).pow(2).mean()
        elif loss_type["base"] == "MSE_RGBA":

            # img: (H, W, 4)  → 先变为 NCHW
            img_rgba = img.unsqueeze(0).permute(0, 3, 1, 2)  # (1,4,H,W)

            # target: 如果是 (H, W, 3)，自动补 alpha=1
            if target.shape[1] == 3:
                # target: (1,3,H,W) → 补充 alpha 通道
                alpha = torch.ones_like(target[:, :1, :, :])  # (1,1,H,W)
                target_rgba = torch.cat([target, alpha], dim=1)  # (1,4,H,W)
            else:
                target_rgba = target  # 已经是 RGBA

            # 计算 RGBA MSE
            loss = (img_rgba - target_rgba).pow(2).mean()
        elif loss_type["base"] == "MSE_PREMULT_RGBA_RGB":

            # === 调用新损失 ===
            loss = enhanced_premult_loss(img, target)

        else:
            raise ValueError("Unknown loss")

        t_range.set_postfix({'loss': loss.item()})

        try:
            loss.backward()
        except RuntimeError as e:
            if "isfinite" in str(e):  # 捕捉 diffvg 中的 NaN 异常
                print(f"[Warning] Non-finite gradient detected at iter {t}. Skipping this iteration.")
                continue
            else:
                raise e  # 其他异常照常抛出

        optimizer.step()
        # 调度器
        scheduler.step()
        single_iter_time = time.time() - s
        if single_iter_time > 10:
            raise ValueError("Itration Over Time!")
        # ========= 几何参数安全约束（防 NaN / 越界） =========
        for shape in shapes:
            # ---- Bézier Path ----
            if isinstance(shape, pydiffvg.Path):
                # points: (M,2) tensor  [x,y]
                shape.points.data[:, 0].clamp_(0.0, canvas_width)  # X 坐标
                shape.points.data[:, 1].clamp_(0.0, canvas_height)  # Y 坐标

            # ---- 圆形 ----
            elif isinstance(shape, pydiffvg.Circle):
                shape.center.data[0].clamp_(0.0, canvas_width)
                shape.center.data[1].clamp_(0.0, canvas_height)
                shape.radius.data.clamp_(1.0, max(canvas_width, canvas_height))

            # ---- 椭圆 ----
            elif isinstance(shape, pydiffvg.Ellipse):
                shape.center.data[0].clamp_(0.0, canvas_width)
                shape.center.data[1].clamp_(0.0, canvas_height)
                shape.radius.data.clamp_(1.0, max(canvas_width, canvas_height))
                # ellipse.angle 可保持原样或做 wrap

            # ---- Polygon / Polyline ----
            elif isinstance(shape, (pydiffvg.Polygon, pydiffvg.Polyline)):
                shape.points.data[:, 0].clamp_(0.0, canvas_width)
                shape.points.data[:, 1].clamp_(0.0, canvas_height)

            # ---- Stroke 宽度统一约束 ----
            if hasattr(shape, "stroke_width") and shape.stroke_width is not None:
                shape.stroke_width.data.clamp_(0.1, 10.0)  # 根据需求调整上限

        # ========= 颜色 / 偏移量约束 =========
        for group in shape_groups:
            fill = group.fill_color
            if isinstance(fill, (pydiffvg.RadialGradient, pydiffvg.LinearGradient)):
                fill.stop_colors.data.clamp_(0.0, 1.0)
                if offsets_vars and offsets_enable and clamp_enable:
                    fill.offsets.data.clamp_(0.0,1.0)
            else:
                fill.data.clamp_(0.0, 1.0)

        # 在每次优化步之后，也把归一化变量做一次 clamp（防止优化器把它们推到奇异值）
        with torch.no_grad():
            for p, _ in begin_norm_params:
                p.clamp_(0.0, 1.0)
            for p, _ in end_norm_params:
                p.clamp_(0.0, 1.0)
            for p, _ in center_norm_params:
                p.clamp_(0.0, 1.0)
            for p, _ in radius_norm_params:
                p.clamp_(0.0, 1.0)

        if save["intermediate_svg"]:
            if t % 20 == 0 or t == num_iters - 1:
                pydiffvg.save_svg(f"{output_dir}/svg/{t:03d}.svg",
                                  canvas_width, canvas_height, shapes, shape_groups)
        if t == 0:
            pydiffvg.save_svg(f"{output_dir}/init.svg",
                              canvas_width, canvas_height, shapes, shape_groups)
        if t == num_iters - 1:
            if background and not save_background:
                shapes, shape_groups = opt_utils.remove_white_background(shapes, shape_groups)

            pydiffvg.save_svg(f"{output_dir}/final.svg",
                              canvas_width, canvas_height, shapes, shape_groups)
    opt_time_e = time.time()
    opt_time = opt_time_e - opt_time_s

    if save["opt_mp4"]:
        save_mp4(num_iters, f"{output_dir}/png", f"{output_dir}/opt_video.avi", w, h)
    if not save["intermediate_png"]:
        delete_folder(f"{output_dir}/png")
    # print("✅ Final optimized SVG saved!")
    if not save["res"]:
        delete_files(output_dir, ".svg")
        delete_files(output_dir, ".png")
        delete_if_empty(output_dir)

    print("🎉 Optimization Complete!")
    return canvas_width, canvas_height, shapes, shape_groups, opt_time


# 优化一组结构svgs
def optimize_struct_svgs(structure_svgs, image_names, cfg, flag_ctl=False, remove=False):
    print("🚀 Start optimizing structure SVG...")
    optimization_init(cfg)
    basic = cfg["basic"]
    optimization = cfg["optimization"]
    save = cfg["save"]
    output_dir = basic["output_dir"]
    output_top_dir = output_dir
    bg = basic["bg"]
    num_iters = optimization["num_iters"]
    targets_path = batch_input(basic["input_images_dir"], basic["input_image"])
    json_save = cfg["save"]["json"]
    total_opt_time = 0
    total_struct_shape_num = 0
    top_json_info, top_json_path = file_utils.read_top_json(cfg)
    # 处理输入图像
    RGBA_flag = False
    loss_type = optimization["loss"]["base"]
    if "rgba" in loss_type.lower():
        RGBA_flag = True
    targets = []
    for index, target_path in enumerate(targets_path):
        if save["struc_opt"]["opt_input"]:
            target_save_path = f"{output_dir}/{image_names[index]}/structure/struc_opt_input.png"
            target = load_and_process_image(target_path, bg, RGBA_flag, gamma, target_save_path)
        else:
            target = load_and_process_image(target_path, bg, RGBA_flag, gamma)

        targets.append(target)

    if structure_svgs is not None:
        # 如果传来了需要优化的svgs 不需要重新读取svgs
        if len(structure_svgs) != len(targets) or len(structure_svgs) != len(image_names):
            raise ValueError("Wrong number of structure_svgs and targets")
        for idx, (shapes, shape_groups, canvas_width, canvas_height) in enumerate(structure_svgs):
            # 创建中间结果的文件夹
            intermediate_path = f"{output_dir}/{image_names[idx]}/structure/intermediate"
            ensure_clean_dir(intermediate_path)
            print(f"{idx + 1}/{len(structure_svgs)}:{image_names[idx]}")
            now_json_file_name = f"{output_top_dir}/{image_names[idx]}/info.json"
            # now json
            if os.path.exists(now_json_file_name):
                with open(now_json_file_name, 'r', encoding='utf-8') as f:
                    now_json_info = json.load(f)
            else:
                raise FileNotFoundError(now_json_file_name)
            if now_json_info.get('optimization') is not None and flag_ctl:
                if now_json_info['optimization'].get("struct_opt_time") is not None:
                    continue
            canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups, struct_opt_time = optimize_svg(
                shapes, shape_groups, canvas_width,
                canvas_height, targets[idx],
                optimization["offsets_enable"],
                optimization["stroke_enable"], num_iters,
                intermediate_path,
                optimization["decay_ratio"],
                optimization["lr"], optimization["loss"],
                save["struc_opt"],clamp_enable=optimization["clamp_enable"])
            if remove:
                pydiffvg.save_svg(f"{output_dir}/{image_names[idx]}/structure/structure_opt_output_no_remove.svg",
                                  canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)

                canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups, loss_shape_num = init_utils.remove_high_alpha_primitives(
                    None, [canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups], 0.5)
            else:
                loss_shape_num = 0
            now_json_info["optimization"] = {
                "struct_opt_time": struct_opt_time,
                "struct_shape_num": len(opt_structure_shapes),
                "struct_shape_remove": loss_shape_num
            }

            json_save_info(json_save, now_json_file_name, now_json_info)
            total_opt_time += struct_opt_time
            pydiffvg.save_svg(f"{output_dir}/{image_names[idx]}/structure/structure_opt_output.svg",
                              canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)
            total_struct_shape_num += len(opt_structure_shapes)
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
            pydiffvg.imwrite(img.cpu(), f"{output_dir}/{image_names[idx]}/structure/structure_opt_output.png",
                             gamma=gamma)
    else:
        # 如果没有就要重新读取，请确保文件夹下有structure_init.svg

        for idx, image_name in enumerate(image_names):
            now_svg_path = f"{output_dir}/{image_name}/structure/structure_init.svg"
            if not os.path.exists(f"{output_dir}/{image_name}/structure") or not os.path.isfile(now_svg_path):
                raise ValueError(f"Don't find the structure init svg of {image_name}.")
            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(now_svg_path)
            intermediate_path = f"{output_dir}/{image_name}/structure/intermediate"
            ensure_clean_dir(intermediate_path)
            print(f"{idx + 1}/{len(image_names)}:{image_name}")
            now_json_file_name = f"{output_top_dir}/{image_names[idx]}/info.json"
            # now json
            if os.path.exists(now_json_file_name):
                with open(now_json_file_name, 'r', encoding='utf-8') as f:
                    now_json_info = json.load(f)
            else:
                raise FileNotFoundError(now_json_file_name)
            if now_json_info.get('optimization') is not None and flag_ctl:
                if now_json_info['optimization'].get("struct_opt_time") is not None:
                    continue
            canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups, struct_opt_time = optimize_svg(
                shapes, shape_groups, canvas_width,
                canvas_height, targets[idx],
                optimization["offsets_enable"],
                optimization["stroke_enable"], num_iters,
                intermediate_path,
                optimization["decay_ratio"],
                optimization["lr"], optimization["loss"],
                save["struc_opt"],clamp_enable=optimization["clamp_enable"])
            if remove:
                pydiffvg.save_svg(f"{output_dir}/{image_name}/structure/structure_opt_output_no_remove.svg",
                                  canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)

                canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups, loss_shape_num = init_utils.remove_high_alpha_primitives(
                    None, [canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups], 0.5)
            else:
                loss_shape_num = 0
            now_json_info["optimization"] = {
                "struct_opt_time": struct_opt_time,
                "struct_shape_num": len(opt_structure_shapes),
                "struct_shape_remove": loss_shape_num
            }
            json_save_info(json_save, now_json_file_name, now_json_info)
            total_opt_time += struct_opt_time
            pydiffvg.save_svg(f"{output_dir}/{image_name}/structure/structure_opt_output.svg",
                              canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)
            total_struct_shape_num += len(opt_structure_shapes)

            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
            pydiffvg.imwrite(img.cpu(), f"{output_dir}/{image_name}/structure/structure_opt_output.png", gamma=gamma)
    if structure_svgs is not None:
        total_len = len(structure_svgs)
    else:
        total_len = len(image_names)
    top_json_info["optimization"] = {
        "struct_total_opt_time": total_opt_time,
        "struct_avg_opt_time": total_opt_time / total_len,
        "total_struct_shape_num": total_struct_shape_num,
        "avg_struct_shape_num": total_struct_shape_num / total_len,

    }

    json_save_info(json_save, top_json_path, top_json_info)


# 优化细节加结构 svgs
def optimize_detail_and_struct_svgs(detail_and_structure_svgs, image_names, id_offsets, cfg, frozen=False,
                                    remove=False):
    print("🚀 Start optimizing detail&structure SVG...")
    optimization_init(cfg)
    basic = cfg["basic"]
    optimization = cfg["optimization"]
    # optimization["lr"]["points_lr"] /=10
    # optimization["lr"]["color_lr"] /=10

    save = cfg["save"]
    output_dir = basic["output_dir"]
    output_top_dir = output_dir
    bg = basic["bg"]
    num_iters = optimization["num_iters"]
    targets_path = batch_input(basic["input_images_dir"], basic["input_image"])
    json_save = cfg["save"]["json"]
    total_opt_time = 0
    total_final_shape_num = 0

    top_json_info, top_json_path = file_utils.read_top_json(cfg)

    # 处理输入图像
    RGBA_flag = False
    loss_type = optimization["loss"]["base"]
    if "rgba" in loss_type.lower():
        RGBA_flag = True
    targets = []
    for index, target_path in enumerate(targets_path):
        if save["detail"]["opt_input"]:
            target_save_path = f"{output_dir}/{image_names[index]}/detail/detail_opt_input.png"
            target = load_and_process_image(target_path, bg, RGBA_flag, gamma, target_save_path)
        else:
            target = load_and_process_image(target_path, bg, RGBA_flag, gamma)

        targets.append(target)

    if id_offsets is None and frozen:
        id_offsets = []
        for idx, image_name in enumerate(image_names):
            now_struct_path = f"{output_dir}/{image_name}/structure/structure_opt_output.svg"
            canvas_width, canvas_height, structure_shapes, structure_shape_groups = pydiffvg.svg_to_scene(
                now_struct_path)
            id_offsets.append(len(structure_shapes))
    elif not frozen:
        id_offsets = []
        for idx, image_name in enumerate(image_names):
            id_offsets.append(-1)

    if detail_and_structure_svgs is not None:
        # 如果传来了需要优化的svgs 不需要重新读取svgs
        if len(detail_and_structure_svgs) != len(targets) or len(detail_and_structure_svgs) != len(image_names):
            raise ValueError("Wrong number of detail_and_structure_svgs and targets")
        for idx, d_s_svg in enumerate(detail_and_structure_svgs):
            if len(d_s_svg) == 0:
                continue
            else:
                shapes, shape_groups, canvas_width, canvas_height = d_s_svg

            # 创建中间结果的文件夹
            intermediate_path = f"{output_dir}/{image_names[idx]}/detail/intermediate"
            ensure_clean_dir(intermediate_path)
            print(f"{idx + 1}/{len(detail_and_structure_svgs)}:{image_names[idx]}")
            now_json_file_name = f"{output_top_dir}/{image_names[idx]}/info.json"
            # now json
            if os.path.exists(now_json_file_name):
                with open(now_json_file_name, 'r', encoding='utf-8') as f:
                    now_json_info = json.load(f)
            else:
                raise FileNotFoundError(now_json_file_name)

            canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups, detail_opt_time = optimize_svg(
                shapes, shape_groups, canvas_width,
                canvas_height, targets[idx],
                optimization["offsets_enable"],
                optimization["stroke_enable"], num_iters,
                intermediate_path,
                optimization["decay_ratio"],
                optimization["lr"], optimization["loss"],
                save["detail"], frozen_id=id_offsets[idx],clamp_enable=optimization["clamp_enable"])
            if remove:
                pydiffvg.save_svg(f"{output_dir}/{image_names[idx]}/detail/detail_opt_output_no_remove.svg",
                                  canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)

                canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups, loss_shape_num = init_utils.remove_high_alpha_primitives(
                    None, [canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups], 0.5)
            else:
                loss_shape_num = 0
            now_json_info["optimization"]["detail_opt_time"] = detail_opt_time
            now_json_info["optimization"]["final_shape_num"] = len(opt_structure_shapes)
            now_json_info["optimization"]["final_shape_remove"] = loss_shape_num
            now_json_info["optimization"]["detail_shape_num"] = \
                len(opt_structure_shapes) - now_json_info["optimization"]["struct_shape_num"]

            total_final_shape_num += len(opt_structure_shapes)
            json_save_info(json_save, now_json_file_name, now_json_info)
            total_opt_time += detail_opt_time
            opt_structure_shapes, opt_structure_shape_groups = opt_utils.add_white_background(opt_structure_shapes, opt_structure_shape_groups,canvas_width, canvas_height)
            pydiffvg.save_svg(f"{output_dir}/{image_names[idx]}/detail/detail_opt_output.svg",
                              canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
            pydiffvg.imwrite(img.cpu(), f"{output_dir}/{image_names[idx]}/detail/detail_opt_output.png", gamma=gamma)
    else:
        # 如果没有就要重新读取，请确保文件夹下有structure_init.svg

        # 上一个步骤是合并结构优化后矢量和细节矢量
        for idx, image_name in enumerate(image_names):
            now_json_file_name = f"{output_top_dir}/{image_names[idx]}/info.json"
            # now json
            if os.path.exists(now_json_file_name):
                with open(now_json_file_name, 'r', encoding='utf-8') as f:
                    now_json_info = json.load(f)
            else:
                raise FileNotFoundError(now_json_file_name)

            if now_json_info["detail"]["initial_shapes_len"] == 0:
                continue
            now_svg_path = f"{output_dir}/{image_name}/detail/detail_opt_input.svg"
            if not os.path.exists(f"{output_dir}/{image_name}/structure") or not os.path.isfile(now_svg_path):
                raise ValueError(f"Don't find the detail_opt_init svg of {image_name}.")
            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(now_svg_path)
            intermediate_path = f"{output_dir}/{image_name}/structure/intermediate"
            ensure_clean_dir(intermediate_path)
            print(f"{idx + 1}/{len(image_names)}:{image_name}")

            canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups, detail_opt_time = optimize_svg(
                shapes, shape_groups, canvas_width,
                canvas_height, targets[idx],
                optimization["offsets_enable"],
                optimization["stroke_enable"], num_iters,
                intermediate_path,
                optimization["decay_ratio"],
                optimization["lr"], optimization["loss"],
                save["detail"], frozen_id=id_offsets[idx],clamp_enable=optimization["clamp_enable"])
            if remove:
                pydiffvg.save_svg(f"{output_dir}/{image_name}/detail/detail_opt_output_no_remove.svg",
                                  canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)

                canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups, loss_shape_num = init_utils.remove_high_alpha_primitives(
                    None, [canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups], 0.5)
            else:
                loss_shape_num = 0
            now_json_info["optimization"]["detail_opt_time"] = detail_opt_time
            now_json_info["optimization"]["final_shape_num"] = len(opt_structure_shapes)
            now_json_info["optimization"]["final_shape_remove"] = loss_shape_num
            now_json_info["optimization"]["detail_shape_num"] = \
                len(opt_structure_shapes) - now_json_info["optimization"]["struct_shape_num"]
            total_final_shape_num += len(opt_structure_shapes)
            json_save_info(json_save, now_json_file_name, now_json_info)
            total_opt_time += detail_opt_time
            opt_structure_shapes, opt_structure_shape_groups = opt_utils.add_white_background(opt_structure_shapes, opt_structure_shape_groups,canvas_width, canvas_height)
            pydiffvg.save_svg(f"{output_dir}/{image_name}/detail/detail_opt_output.svg",
                              canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                canvas_width, canvas_height, opt_structure_shapes, opt_structure_shape_groups)
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
            pydiffvg.imwrite(img.cpu(), f"{output_dir}/{image_name}/structure/structure_opt.png", gamma=gamma)
    if detail_and_structure_svgs is not None:
        total_len = len(detail_and_structure_svgs)
    else:
        total_len = len(image_names)

    top_json_info["optimization"]["detail_total_opt_time"] = total_opt_time
    top_json_info["optimization"]["detail_avg_opt_time"] = total_opt_time / total_len
    top_json_info["optimization"]["avg_detail_shape_num"] = \
        total_final_shape_num / total_len - top_json_info["optimization"]["avg_struct_shape_num"]
    top_json_info["optimization"]["total_final_shape_num"] = total_final_shape_num
    top_json_info["optimization"]["avg_final_shape_num"] = total_final_shape_num / total_len

    json_save_info(json_save, top_json_path, top_json_info)
