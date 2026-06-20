# BA-Vec: Budget-aware image vectorization


BA-Vec是一个利用分割大模型与可微渲染框架实现高效利用路径预算的通用矢量化框架，可以实现全图像类型矢量化，兼容渐变与复杂光效，对路径预算具有强鲁棒性。



## Installation

项目的依赖如：requrements.txt 和environment.yml 所示 请依据机器配置进行调整

也可以先安装以下使用到的项目及模型：
SGLIVE-DiffVG: https://github.com/Rhacoal/diffvg/tree/54cb406ac8bf11394c275236c7f3b496500e3459
SAM: https://github.com/facebookresearch/segment-anything
UnSAM: https://github.com/frank-xwang/UnSAM
vtracer(package) :pip install vtracer 

### 注意：
1. diffvg下载后将文件夹中的pydiffvg/save_svg.py和parse_svg.py使用提供的同名文件覆盖后再进行本地编译
2. 本项目安装的sam模型为vit_h版本 如需更改 请修改配置文件config.yaml
3. 本项目安装unsam模型为unsam_sa1b_4perc_ckpt_200k.pth 如需更改 请修改配置文件config.yaml
4. 建议每个项目安装完成后进行对应的测试以确保该部分无误，并修改程序和配置中的路径


## Getting Started
创建完整的conda虚拟环境 激活conda环境 运行主程序 运行时请关注显存是否充足

                 python main.py \
                --exp_name test \
                --config config.yml \
                --color_type 2 \
                --stop_number 2 \
                --target "1.png" \
                --output_dir "res\" \
                --detail_ratio "0.25" \
                --num_iters "200" \
                --shape_num "256" \
                --no_use_hq_sam \
                --device "cuda"\
                --loss_type "MSE_RGB"

运行结果会存储在指定文件夹下，最终的矢量图在：图像名称/detail/detail_opt_output.svg

## QQ
辅助安装请联系：1225257645
