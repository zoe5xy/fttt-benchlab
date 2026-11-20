# 按数据集管理 FTTT 扩散先验

本目录供三种公开算法中的 `dps` 选择预训练先验。`dip` 和 `inr` 不依赖扩散权重。
每个数据集使用独立的 manifest、权重和训练记录，避免将合成示例误用为真实数据先验。

## 当前已有内容

```text
diffusion/
  README.md
  weights/
    synthetic_ellipses/
      manifest.json
      reference.pt
      reference.pt.json
    openbreastus/
      manifest.json
    nbpslice2d/
      manifest.json
```

| 数据集 ID | 状态 | 后端 | 当前用途 |
| --- | --- | --- | --- |
| `synthetic_ellipses` | `ready` | `native_tiny_unet` | 16×16 合成椭圆先验，接口及合成实验 |
| `openbreastus` | `untrained` | `score_sde_pytorch` | 预留位置，无该数据集的预训练权重 |
| `nbpslice2d` | `untrained` | `score_sde_pytorch` | 预留位置，无该数据集的预训练权重 |

后两个 manifest 中的 `training_config.py` 是待准备的配置名称，当前没有相应文件或
checkpoints 目录。合成参考权重源自既有训练记录，不能作为真实乳腺组织先验或重建
质量结论。

## 查看与调用

从仓库根目录执行；查看登记信息不需要安装 PyTorch：

```bash
python -m ftttbench.algorithms.fttt.weight_bank
python -m ftttbench.algorithms.fttt.weight_bank --dataset openbreastus
```

安装 `fttt-benchlab[neural]` 后，可用实际存在的合成配置运行 DPS：

```bash
fttt simulate --out data/dps_demo.h5
fttt run dps --case data/dps_demo.h5 --config configs/algorithms/fttt_dps.yaml --out runs/dps_demo
```

配置选择 `prior_manifest`，也可显式提供 `checkpoint`，二者不能同时设置。
算法 YAML 中的路径相对 YAML 文件目录解析，当前合成配置为：

```yaml
name: dps
parameters:
  prior_manifest: ../../diffusion/weights/synthetic_ellipses/manifest.json
  sampling_steps: 64
  guidance_scale: 0.1
  eta: 0.0
  seed: 0
deployment:
  neural_device: cpu
```

manifest 中的 checkpoint 路径相对 manifest 所在数据集目录，且不得越出该目录。
加载时检查文件、SHA256、图像尺寸、模型结构、归一化和噪声日程。
缺失、未训练或尚未适配的权重会明确失败，不会使用随机网络替代。

## manifest 状态

| 状态 | 含义 |
| --- | --- |
| `untrained` | 尚无可用权重；checkpoint 和 SHA256 可以为空 |
| `registered` | 已登记权重文件和 SHA256，尚未完成当前 DPS 运行适配 |
| `ready` | 已符合当前原生 TinyUNet 加载器支持的格式 |

`ready` 表示格式可运行，不代表图像质量达标。当前外部 Score SDE 后端只能登记为
`registered`，完成明确的运行适配之前不能作为 `ready` 调用。
manifest 格式标识保留 `usct.diffusion.weights.v1`，原生 checkpoint 保留
`usct.fttt.prior.v1`，用于兼容来源实现的已有文件。

## 原生训练入口

使用本库的 TinyUNet 和离散 VP 日程可训练一个匹配当前 DPS 接口的先验。
输出路径必须不存在；下面命令不会覆盖附带的参考权重：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m ftttbench.algorithms.fttt.train_prior --output checkpoints/my_synthetic_prior.pt --steps 5000 --shape 16 16 --seed 70
```

不提供 `--images` 时，训练器生成 512 张训练图、另一种子生成 32 张验证图。
图像是 1500 m/s 背景中的随机椭圆，训练目标是归一化慢度上的噪声预测。
训练元数据、图像哈希及固定验证噪声下的 epsilon MSE 随权重保存。
该验证指标用于检查噪声预测，不能直接解释为重建图像精度。

自有数据使用 NPZ 中的 `train_speed_mps`、`validation_speed_mps`，
形状均为 `[image, ny, nx]`，单位 m/s，当前工具要求有限且处于 `[1300,1700]`：

```bash
python -m ftttbench.algorithms.fttt.train_prior --images data/prior_images.npz --shape 128 128 --steps 5000 --output checkpoints/my_dataset_prior.pt
```

训练、验证和重建测试必须先按独立患者或病例划分。工具检查跨集合重复图像，
但无法自动证明病例级独立性。训练尺寸必须匹配后续重建网格，当前加载器不会
自动缩放不同大小的图像。

需要通过权重库选择新原生先验时，在 `weights/<dataset_id>/` 保存 checkpoint、
创建 manifest 并填写实际的相对路径与 SHA256：

```json
{
  "schema_version": "usct.diffusion.weights.v1",
  "dataset_id": "my_dataset",
  "backend": "native_tiny_unet",
  "status": "ready",
  "checkpoint": "prior.pt",
  "sha256": "replace_with_actual_64_character_sha256",
  "training": {
    "repository": "local:ftttbench.algorithms.fttt.train_prior",
    "image_shape": [128, 128]
  },
  "notes": "Replace with actual data source, split, training and validation records."
}
```

上面是填写模板，SHA256 占位值必须替换后才能通过校验。
不要覆盖其他数据集的 manifest，也不要将合成先验登记为真实数据集权重。

## 使用外部 Score SDE 训练

官方实现为 [yang-song/score_sde](https://github.com/yang-song/score_sde)（JAX）及
[yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)。
这些仓库的 checkpoint 与当前原生 TinyUNet 格式并不直接兼容。

1. 准备独立的训练、验证和测试病例，明确物理变量、单位、尺寸、轴方向、归一化
   和 centered 缩放。仅使用训练数据确定训练所需统计量。
2. 在单独环境安装官方代码依赖，完成自己的声速/慢度图像加载器与训练配置。
   本库 HDF5/NPZ 不会自动成为官方训练格式。
3. 在对应数据集目录保存实际的 `training_config.py`、上游提交、数据划分和
   预处理记录，然后在官方仓库运行其训练命令。
4. 保存选定 checkpoint、EMA 参数、配置和评估记录；填写 manifest 的相对路径和
   SHA256，状态设为 `registered`。
5. 编写并验证运行适配，核对结构、时间标签、EMA、缩放、输出变量和噪声日程。

当前 DPS 使用离散 VP epsilon 预测器。只有在匹配的 VP 扰动与坐标缩放下，
才可使用 `epsilon = -sigma(t) * score` 转换。
VE、subVP、连续时间模型及 NCSN++/DDPM++ 不能仅改文件名后套用当前调度器。
真实数据权重与适配准备完成后，再用独立重建病例评价观测残差和成像质量。
