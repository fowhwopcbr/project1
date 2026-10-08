# OmniStream + TMTB 视频人群计数基础版

独立的研究集成工程：冻结 OmniStream，使用最后一层空间特征训练单个 TMTB 密度头。本仓库不是 OmniStream 或 TMTB 的官方实现仓库，不包含新增的 A/B 适配器或双分支模块。

```text
当前及过去的视频帧 → OmniStream → 最后一层 patch 特征
→ 逐帧空间特征图 → TMTB 密度头 → ROI 掩膜 → 空间求和
```

源码包不含预训练权重、数据集、训练结果、IDE 设置及个人路径。权重单独下载，数据集由使用者自行准备。本次打包仅进行静态与文件核对，没有运行 Python、训练或测试。

## 1. 安装环境

以下为 Linux / RTX 5090 的建议环境安装命令，由使用者执行。Python 3.10 或 3.12 均可作为环境版本；完整组合尚未在目标服务器实测。宿主机需要支持该 GPU 和 CUDA 构建的 NVIDIA 驱动。

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-bus.txt
python -m pip check
```

若服务器已经提供匹配的 PyTorch / torchvision，可跳过第一条。环境信息检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
```

PyTorch 2.8.0 的 cu128 安装包及 torchvision 配对见 [官方版本列表](https://pytorch.org/get-started/previous-versions/)。实际训练入口无需额外安装 FlashAttention、xFormers 或官方仓库的完整研究环境。

## 2. 下载预训练权重

从项目根目录运行以下命令。固定 revision 与原基础实验使用的权重一致：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='StreamFormer/OmniStream', revision='d40aa14c851f6932b39df83a4fa20173c106651b', local_dir='checkpoints/OmniStream', allow_patterns=['config.json','model.safetensors','preprocessor_config.json','README.md'])"
```

模型加载默认使用本地文件，不会在权重缺失时自动改为随机初始化骨干。权重文件约 1.21 GB，SHA-256：

```text
f9da399c61360115d956466fd1dda29f457ae4a37ffb0ff9b3a9a357bc0033d2
```

已有完整权重目录时也可以直接复制到 `checkpoints/OmniStream/`，无需重复下载。官方模型来源：[StreamFormer/OmniStream](https://huggingface.co/StreamFormer/OmniStream)。

## 3. 准备 Bus 数据

数据来源和获取入口见 [MFA 项目](https://github.com/tianhangpan/MFA)。本仓库不分发数据；当前加载器要求已准备好的 H5 `density` 标签与 ROI，不能直接将点坐标文件作为密度图使用。

```text
/your/path/DSTI/bus/
├── bus_roi.npy
├── train/
│   ├── images/bus_*.jpg
│   └── ground_truth/bus_*.h5
└── test/
    ├── images/bus_*.jpg
    └── ground_truth/bus_*.h5
```

每个 H5 必须含与对应图像同尺寸的 `density` 数组。原 Bus 图像为 576×704，ROI 为同尺寸二值数组。

**复用本基础实验的划分（建议）**：复制示例清单，然后只修改 `root` 为本机数据目录：

```bash
cp configs/bus_manifest.example.json configs/bus_manifest.json
```

Windows PowerShell 可使用 `Copy-Item configs/bus_manifest.example.json configs/bus_manifest.json`。示例清单保留原来的 frame ID 列表：train 1114、隔离区 16、val 283、test 1413。不得为了迁移服务器重新混洗这些分区。

**首次生成划分或审核新数据副本**：可执行以下准备脚本；此方式与上面的复制方式二选一。脚本拒绝覆盖已有清单，审核时可用 `--output` 指定另一个文件。

```bash
python prepare_bus.py --root /your/path/DSTI/bus --full
```

划分方式为原训练序列尾部留出验证集，并在训练和验证间留出隔离帧；这是内部验证协议，不声称与原论文的验证协议相同。

## 4. 检查与训练

所有命令均在项目根目录运行。先检查小尺寸接口和一批真实原图数据：

```bash
python check_crowd_model.py --device cuda --frames 2 --height 128 --width 192 --backward
python train_bus.py check --device cuda --frames 2 --full-frame
```

这些检查包含前向、反向和参数更新，但不保存训练结果，输出误差不能作为模型准确率。确认成功后再进行一轮试训：

```bash
python train_bus.py train --device cuda --frames 2 --full-frame --epochs 1 --run-dir runs/bus_baseline_trial --diagnostics
```

正式训练示例（30 轮只是初始预算，是否收敛需看验证曲线）：

```bash
python train_bus.py train --device cuda --frames 2 --full-frame --epochs 30 --seed 42 --run-dir runs/bus_baseline_30 --diagnostics
```

- 默认冻结主干，只训练密度头；batch=1、FP32、密度 MSE，人数 L1 损失权重为 0。
- `--full-frame` 不裁剪、不缩放，训练保留片段同步水平翻转；不加该参数则默认裁剪训练。
- 输入 `[B,T,3,H,W]`，密度输出 `[B,T,1,H/4,W/4]`。每个片段只监督最后一帧。
- 片段只取当前分区内的当前和过去帧，开头不足时重复首帧；片段之间不保存 KV cache。
- 训练入口不支持断点续训；非空 run 目录拒绝覆盖。每个 run 的 best/last 完整检查点合计约 2.4 GB，另需日志及诊断文件空间。
- `--diagnostic-features` 可增加选定样本的中间特征输出，详见 [诊断说明](DIAGNOSTICS_README.md)。

## 5. 验证与测试

```bash
python train_bus.py evaluate --device cuda --checkpoint runs/bus_baseline_30/best.pt --split val --diagnostics
python train_bus.py evaluate --device cuda --checkpoint runs/bus_baseline_30/best.pt --split test --diagnostics
```

best 仅由验证 MAE 选择；设置确定后再进行最终测试。评估读取检查点旁的 settings 和 manifest。当前人数真值使用 H5 密度积分、ROI 掩膜和保总量降采样；与其他论文比较前需统一点标注、ROI 和预处理协议。

## 6. 上传 GitHub

将本目录的内容作为仓库根目录上传，不要把本地完整训练工程或此目录外的权重文件拖进去。打包时没有创建远程仓库或发布任何内容。

`.gitignore` 已排除 checkpoints、runs、数据、环境、缓存、实际数据清单及常见凭据文件。示例清单可以提交。上传前用 `git status --short` 检查待提交内容；GitHub 网页上传时也请保持该目录结构。

## 来源与许可

- [OmniStream](https://github.com/Go2Heart/OmniStream)：视频骨干，保留上游 MIT LICENSE。
- [TMTB](https://github.com/syhien/taste_more_taste_better)：CountingHead，保留原头文件和 Apache-2.0 许可。
- 随骨干附带的 DINOv3 代码保留 Meta 版权及 DINOv3 License；部分 DINOv2 来源文件保留其原始版权和许可标记。

各组件分别适用其原有许可，根目录 LICENSE 不替代第三方组件条款。详见 [third_party/NOTICE.md](third_party/NOTICE.md)。
