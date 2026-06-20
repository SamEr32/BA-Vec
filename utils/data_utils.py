import os
import glob
from utils import opt_utils, file_utils, init_utils
import json
import numpy as np
from typing import List, Dict, Tuple


# 就能用保存tangka的其他不好用
def convert_gt_to_png(dataset_name):
    base_dir = os.path.abspath("../data")

    if dataset_name == 'tangka':
        dataset_path = os.path.join(base_dir, "tangka/svg")
        save_png = True
    elif dataset_name == 'openclipart':
        dataset_path = os.path.join(base_dir, "openclipart/svg")
        save_png = True
    elif dataset_name == 'iconfont':
        dataset_path = os.path.join(base_dir, "iconfont/svg")
        save_png = False  # iconfont 不保存彩色png
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    print(f"[INFO] Using dataset path: {dataset_path}")
    assert os.path.isdir(dataset_path), f"Dataset directory does not exist: {dataset_path}"

    line_dir = os.path.join(os.path.dirname(dataset_path), "line_png")
    png_dir = os.path.join(os.path.dirname(dataset_path), "png")
    line_svg_dir = os.path.join(os.path.dirname(dataset_path), "line_svg")

    os.makedirs(line_dir, exist_ok=True)
    os.makedirs(line_svg_dir, exist_ok=True)
    if save_png:
        os.makedirs(png_dir, exist_ok=True)

    svg_files = glob.glob(os.path.join(dataset_path, "*.svg"))
    print(f"[INFO] Found {len(svg_files)} SVG files.")

    failed = []

    for svg_path in svg_files:
        filename = os.path.splitext(os.path.basename(svg_path))[0]
        line_output_path = os.path.join(line_dir, filename + ".png")
        line_svg_output_path = os.path.join(line_svg_dir, filename + ".svg")

        print(f"[INFO] Processing: {svg_path}")
        try:
            # 先生成线稿图和对应svg
            opt_utils.render_svg_outline(svg_path, line_output_path, line_svg_output_path)
            # 只有非iconfont才生成彩色png
            if save_png:
                png_output_path = os.path.join(png_dir, filename + ".png")
                opt_utils.render_svg(svg_path, png_output_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {svg_path}: {e}")
            failed.append((svg_path, str(e)))

    if failed:
        print("\n[SUMMARY] Some files failed:")
        for path, reason in failed:
            print(f" - {path} | Reason: {reason}")
    else:
        print("[SUMMARY] All SVGs processed successfully!")


def average_file_size(folder_path, extension):
    """
    计算一个文件夹及其所有子文件夹中所有指定扩展名文件的平均大小（单位KB）
    """
    total_size = 0
    count = 0

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(extension.lower()):
                filepath = os.path.join(root, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
                    count += 1

    if count == 0:
        return 0.0
    return total_size / count / 1024  # 转换为KB


def average_png_size(folder_path):
    return average_file_size(folder_path, ".png")


def average_svg_size(folder_path):
    return average_file_size(folder_path, ".svg")


def get_last_folder_name(path):
    return os.path.basename(os.path.normpath(path))


def analyze_data_folder(root_folder="../data", output_file="avg_sizes.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Folder\tAverage PNG Size (KB)\tAverage SVG Size (KB)\n")

        for root, dirs, _ in os.walk(root_folder):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                avg_png = average_png_size(dir_path)
                avg_svg = average_svg_size(dir_path)
                relative_path = os.path.relpath(dir_path, root_folder)
                f.write(f"{relative_path}\t{avg_png:.2f}\t{avg_svg:.2f}\n")

    print(f"分析完成，结果保存在: {output_file}")


def compute_ours_metrics(cfg):
    json_info, json_file_name = file_utils.read_top_json(cfg)
    dataset_path = cfg.get("basic", {}).get("input_images_dir", "")
    output_dir = cfg.get("basic", {}).get("output_dir", "")
    dataset_image_paths = file_utils.get_all_png_images(dataset_path)

    json_info["metrics"] = {
        "dataset_path": dataset_path,
        "dataset_length": len(dataset_image_paths),
    }

    dataset_name = dataset_path.split("/")[2] if len(dataset_path.split("/")) > 2 else ""
    gt_dataset_name = ["iconfont/png", "openclipart/png", "tangka/png", "iconfont", "openclipart", "tangka"]

    total_mse = total_psnr = total_lpips = total_ssim = total_mse_rgba = 0
    total_svg_size = total_vb_mse = total_vbq = total_points_num = 0
    opt_time_total = pro_time_total = 0
    detail_ratio = final_shapes_num = 0
    N = 0
    wrong_info = []

    for index, image_path in enumerate(dataset_image_paths):
        print(image_path)
        image_name = file_utils.get_png_name_without_suffix(image_path)
        target_path = os.path.join(dataset_path, f"{image_name}.png")
        stru_svg_path = os.path.join(output_dir, image_name, "structure", "structure_opt_output.svg")
        svg_res_path = os.path.join(output_dir, image_name, "detail", "detail_opt_output.svg")
        png_res_path = os.path.join(output_dir, image_name, "detail", "detail_opt_output.png")
        # opt_utils.render_svg(svg_res_path, png_res_path)

        line_png_res_path = os.path.join(output_dir, image_name, "detail", "detail_opt_output_line.png")
        line_svg_res_path = os.path.join(output_dir, image_name, "detail", "detail_opt_output_line.svg")

        if os.path.exists(svg_res_path):
            if not os.path.exists(line_png_res_path) or not os.path.exists(line_svg_res_path):
                opt_utils.render_svg_outline(svg_res_path, line_png_res_path, line_svg_res_path)

            target = opt_utils.read_image(target_path)
            RGBA_target = opt_utils.read_image(target_path,to_rgb=False)

            png_res = opt_utils.read_image(png_res_path)
            RGBA_png_res = opt_utils.read_image(png_res_path,to_rgb=False)
            line_png_res = opt_utils.read_image(line_png_res_path)
            points_num, f_shapes_num = init_utils.count_points_and_shapes(svg_res_path)
            mse = opt_utils.compute_mse(target, png_res)
            mse_rgba = opt_utils.compute_mse_rgba(RGBA_target, RGBA_png_res)
            psnr = opt_utils.compute_psnr(target, png_res)
            lpips = opt_utils.compute_lpips(target, png_res)
            ssim = opt_utils.compute_ssim(target, png_res)
            svg_size = os.path.getsize(svg_res_path) / 1024
            vbq = opt_utils.compute_vbq(target_path, png_res_path, line_png_res_path, points_num)

            total_mse += mse
            total_points_num += points_num
            total_mse_rgba += mse_rgba
            total_psnr += psnr
            total_lpips += lpips
            total_ssim += ssim
            total_svg_size += svg_size
            total_vbq += vbq

            if dataset_name in gt_dataset_name:
                d = dataset_name.split("/")[0]
                gt_line_png_path = os.path.join("..", "data", d, "line_png", f"{image_name}.png")
                gt_line_png = opt_utils.read_image(gt_line_png_path)
                vb_mse = opt_utils.compute_mse(gt_line_png, line_png_res)
                total_vb_mse += vb_mse
        else:
            wrong_info.append(image_name)
            print(f"{image_name} no {svg_res_path}")
            f_shapes_num = 0

        now_json_path = os.path.join(output_dir, image_name, "info.json")
        print(now_json_path)

        if not os.path.exists(now_json_path):
            wrong_info.append(image_name)
            print(f"{image_name} no {now_json_path}")

        else:
            with open(now_json_path, 'r', encoding='utf-8') as f:
                now_json_info = json.load(f) or {}

            optimization_info = now_json_info.get("optimization", {})
            structure_info = now_json_info.get("structure", {})
            detail_info = now_json_info.get("detail", {})

            if optimization_info.get("struct_opt_time") is not None:
                opt_time_total += optimization_info.get("detail_opt_time", 0.0) + optimization_info.get(
                    "struct_opt_time", 0.0)
                pro_time_total += structure_info.get("structure_time", 0.0) + structure_info.get("merge_time", 0.0) + \
                                  detail_info.get("detail_capture_time", 0.0) + detail_info.get("detail_merge_time",
                                                                                                0.0)

                _, s_shapes_num = init_utils.count_points_and_shapes(stru_svg_path)
                if f_shapes_num != 0:
                    detail_ratio += 1 - float(s_shapes_num) / float(f_shapes_num)
                    final_shapes_num += f_shapes_num

                    N += 1
                else:
                    wrong_info.append(image_name)
                    print(f"{image_name} no f_shapes_num")

            else:
                wrong_info.append(image_name)
                print(f"{image_name} no struct_opt_time")

    if N == 0:
        print(wrong_info)
        raise FileNotFoundError("No right files!")

    json_info["metrics"]["mse"] = total_mse / N
    json_info["metrics"]["mse_rgba"] = total_mse_rgba / N
    json_info["metrics"]["psnr"] = total_psnr / N
    json_info["metrics"]["lpips"] = total_lpips / N
    json_info["metrics"]["ssim"] = total_ssim / N
    json_info["metrics"]["svg_size"] = total_svg_size / N
    json_info["metrics"]["vbq"] = total_vbq / N
    json_info["metrics"]["points_num"] = total_points_num / N



    # 总时间和比率
    json_info["metrics"]["opt_time"] = optimization_info.get("struct_avg_opt_time",
                                                             opt_time_total / N) + optimization_info.get(
        "detail_avg_opt_time", 0.0)
    json_info["metrics"]["pro_time"] = structure_info.get("sam_avg_time", pro_time_total / N) + \
                                       structure_info.get("avg_merge_time", 0.0) + \
                                       detail_info.get("avg_detail_capture_time", 0.0) + \
                                       detail_info.get("avg_merger_time", 0.0)

    avg_final_shape = optimization_info.get("avg_final_shape_num", final_shapes_num / N)
    avg_detail_shape = optimization_info.get("avg_detail_shape_num", detail_ratio / N * avg_final_shape)

    if avg_final_shape > 0:
        json_info["metrics"]["detail_ratio"] = avg_detail_shape / avg_final_shape
        json_info["metrics"]["final_shapes_num"] = avg_final_shape
    else:
        json_info["metrics"]["detail_ratio"] = 0
        json_info["metrics"]["final_shapes_num"] = 0

    if dataset_name in gt_dataset_name:
        json_info["metrics"]["vb_mse"] = total_vb_mse / N

    if wrong_info:
        json_info["metrics"]["wrong_info"] = wrong_info

    file_utils.json_save_info(True, json_file_name, json_info)
    print(json_file_name)


def record_mask_info(output_dir: str, image_name: str,
                     sam_masks: List[Dict],
                     unsam_masks: List[Dict],
                     unsam_masks_bg: List[Dict] = None):
    """
    记录每个 mask 的复杂度、是否经过二级分割、以及对应分割结果情况。

    参数：
        output_dir: 输出目录（建议是 structure/{image_name}/ ）
        image_name: 当前图像名
        sam_masks: 原始 SAM 掩码列表（每个包含 segmentation、area、complexity）
        unsam_masks: SAM 掩码的二级分割结果列表
        unsam_masks_bg: 空白区域 UnSAM 分割结果（可选）
    """
    record = {
        "image_name": image_name,
        "mask_records": []
    }

    # --- 建立一个简单索引 ---
    for i, m in enumerate(sam_masks):
        rec = {
            "mask_id": f"sam_{i}",
            "type": "sam",
            "area": int(m.get("area", 0)),
            "complexity": float(m.get("complexity", 0)),
            "second_stage": False,
            "child_masks": []
        }

        # 检查是否在 unsam_masks 中有对应区域
        related_unsam = []
        for j, um in enumerate(unsam_masks):
            inter = np.logical_and(m["segmentation"], um["segmentation"]).sum()
            union = np.logical_or(m["segmentation"], um["segmentation"]).sum()
            iou = inter / union if union > 0 else 0
            if iou > 0.2:  # IOU > 0.2 认为来自该 SAM 区域的二级分割
                related_unsam.append(j)

        if related_unsam:
            rec["second_stage"] = True
            for idx in related_unsam:
                um = unsam_masks[idx]
                rec["child_masks"].append({
                    "mask_id": f"unsam_{idx}",
                    "area": int(um.get("area", 0)),
                    "complexity": float(um.get("complexity", 0))
                })

        record["mask_records"].append(rec)

    # --- 若存在空白区域分割结果 ---
    if unsam_masks_bg:
        for i, m in enumerate(unsam_masks_bg):
            record["mask_records"].append({
                "mask_id": f"unsam_bg_{i}",
                "type": "unsam_bg",
                "area": int(m.get("area", 0)),
                "complexity": float(m.get("complexity", 0)),
                "second_stage": False,
                "child_masks": []
            })

    # --- 保存 JSON ---
    save_path = os.path.join(output_dir, f"{image_name}_mask_info.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"✅ Mask 信息已保存至: {save_path}")


if __name__ == "__main__":
    input_images_dir = "../data/emoji/Y"
    shape_num = 8
    output_dir = f"../res/ours/emoji/Y/{shape_num}_2_200_MSE_RGBA"
    cfg = {
        "basic":
            {
                "shape_num": shape_num,
                "exp_name": "2_200_MSE_RGBA",
                "modern": "exp",
                "input_images_dir": input_images_dir,
                "output_dir": output_dir
            },
        "save": {
            "json": True,
        }

    }
    compute_ours_metrics(cfg)
