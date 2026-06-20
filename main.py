import os.path
import torch, numpy as np, random

from utils.file_utils import read_config, get_input_image_names,get_png_name_without_suffix
from init.initialize import structure_init, detail_init, merge_structure_and_detail_svgs
from opt.optimize import optimize_struct_svgs, optimize_detail_and_struct_svgs
from utils.opt_utils import parse_args
from utils import data_utils, file_utils
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = parse_args()
    # cfg = read_config("config/config.yaml")
    image_names = get_input_image_names(cfg)
    if cfg.get("metrics"):
        print("✅ 开始计算指标")
        json_info, json_file_name = file_utils.read_top_json(cfg)
        # print(json_file_name)
        data_utils.compute_ours_metrics(cfg)
    else:
        # 阶段执行控制标志
        structure_init_flag = True
        optimize_struct_svgs_flag = True
        detail_init_flag = True
        merge_structure_and_detail_svgs_flag = True
        optimize_detail_and_struct_svgs_flag = True

        detail_opt_init_svgs = None
        structure_svgs = None


        # 跑一整个数据集 或者跑一张图片
        if cfg.get("basic").get("input_images_dir") is None:

            flag = False
        else:
            flag = True
        # print(flag)
        if flag:
            # print(cfg["basic"]["input_images_dir"])
            # 读取已有 json 结果
            json_info, json_file_name = file_utils.read_top_json(cfg)
            if json_info is not None:

                # 判断结构部分是否已存在
                if json_info.get("structure", {}).get("avg_merge_time") is not None:
                    structure_init_flag = False
                if json_info.get("optimization", {}).get("struct_avg_opt_time") is not None:
                    optimize_struct_svgs_flag = False

                # 判断细节部分是否已存在
                if json_info.get("detail", {}).get("avg_detail_capture_time") is not None:
                    detail_init_flag = False
                if json_info.get("detail", {}).get("avg_merger_time") is not None:
                    merge_structure_and_detail_svgs_flag = False
                if json_info.get("optimization", {}).get("detail_avg_opt_time") is not None:
                    optimize_detail_and_struct_svgs_flag = False



            # 执行各阶段流程
            if structure_init_flag:
                print("🔹 执行结构矢量初始化")
                structure_svgs, image_names = structure_init(cfg, True)

            if optimize_struct_svgs_flag:
                print("🔹 执行结构矢量优化")
                optimize_struct_svgs(structure_svgs, image_names, cfg, True)

            if detail_init_flag:
                print("🔹 执行细节矢量初始化")
                detail_svgs, image_names = detail_init(cfg)

            if merge_structure_and_detail_svgs_flag:
                print("🔹 执行结构与细节矢量合并")
                detail_opt_init_svgs, id_offsets = merge_structure_and_detail_svgs(image_names, cfg)

            if optimize_detail_and_struct_svgs_flag:
                print("🔹 执行最终矢量优化")
                optimize_detail_and_struct_svgs(detail_opt_init_svgs, image_names, id_offsets, cfg)

            # 最后统一评估
            print("✅ 开始计算指标")
            data_utils.compute_ours_metrics(cfg)
        else:
            # print(cfg["basic"]["input_image"])
            image_name = get_png_name_without_suffix(cfg["basic"]["input_image"])
            top_out_dir = cfg["basic"]["output_dir"]
            base = f"{top_out_dir}/{image_name}"
            if not os.path.exists(f"{base}/structure/structure_init.svg"):
                print("🔹 执行结构矢量初始化")
                structure_svgs, image_names = structure_init(cfg)

            if not os.path.exists(f"{base}/structure/structure_opt_output.svg"):
                print("🔹 执行结构矢量优化")
                optimize_struct_svgs(structure_svgs, image_names, cfg)


            print("🔹 执行细节矢量初始化")
            detail_svgs, image_names = detail_init(cfg)


            print("🔹 执行结构与细节矢量合并")
            detail_opt_init_svgs, id_offsets = merge_structure_and_detail_svgs(image_names, cfg)


            print("🔹 执行最终矢量优化")
            optimize_detail_and_struct_svgs(detail_opt_init_svgs, image_names,id_offsets, cfg)


