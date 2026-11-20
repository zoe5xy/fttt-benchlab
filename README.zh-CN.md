# FTTT BenchLab

[English](README.md) | 简体中文

独立的二维首达走时层析成像（First-arrival Travel-time Tomography，FTTT）代码库。
输入换能器位置和首达走时，输出声速图；包含病例、配置、运行和结果管理。
Python 包名为 `ftttbench`，命令行为 `fttt`。
收到完整的 `fttt-benchlab` 目录后即可安装运行，无需其他代码仓库。

公开重建算法只有三种：

| ID | 方法 | 图像表示 | 是否需要预训练权重 |
| --- | --- | --- | --- |
| `dip` | Deep Image Prior | 固定随机输入的 SkipUNet | 否 |
| `inr` | Implicit Neural Representation | 坐标输入的 SIREN | 否 |
| `dps` | Diffusion Posterior Sampling | VP 扩散先验及走时观测引导 | 是 |

三种方法共用可微 Eikonal 首达正演，分别通过 `dip`、`inr`、`dps` 调用。

## 安装

要求 Python 3.12 或更高版本。在收到的 `fttt-benchlab` 目录中执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,viz]'
fttt list-algorithms --json
fttt describe-algorithm inr
```

基础安装 `python -m pip install -e .` 包含 NumPy、SciPy、h5py、Pydantic、
PyYAML 和 scikit-image，可查看算法、生成和读取病例。实际运行三种神经方法还需
`neural`（PyTorch）。`performance` 提供 Numba，`viz` 提供 Matplotlib/Pillow，
`dev` 提供 pytest、build 和 Ruff，参与开发或运行测试时可另行安装
`python -m pip install -e '.[dev]'`。CUDA 版 PyTorch 请按自己的驱动环境安装；
数值 Eikonal 求解目前仍在 CPU 上执行。

## 从合成病例开始

```bash
fttt simulate --out data/demo.h5
fttt inspect --case data/demo.h5
fttt run inr --case data/demo.h5 --config configs/algorithms/fttt_inr.yaml --out runs/demo_inr
fttt evaluate --run runs/demo_inr
```

也可使用 `python -m ftttbench` 替代 `fttt`。运行命令要求空输出目录；
请为新实验选择新的目录。

短流程可用以下 Python 调用，避免一开始运行默认的长迭代配置：

```python
from ftttbench import run, write_case_hdf5
from ftttbench.data.synthetic import make_first_arrival_case

case = make_first_arrival_case(shape=(16, 16), n_transducers=12)
write_case_hdf5(case, "data/python_demo.h5")
result = run(
    "inr", case,
    {"parameters": {"seed": 0}, "run_controls": {"max_iterations": 30}},
    out="runs/python_inr", preview=False,
)
if str(result.status) == "failed":
    raise RuntimeError(result.failure_reason)
print(result.sound_speed_mps.shape, result.metrics["data_relative_residual"])
```

三方法演示和批量比较：

```bash
python examples/fttt_demo.py --out runs/three_methods --iterations 30 --sampling-steps 16
fttt bench --suite configs/benchmarks/synthetic.yaml
```

批量配置默认读取 `data/` 中的合成病例，先执行上面的 `simulate`。
YAML 中的文件路径相对该 YAML 文件所在目录解析。
例如 DPS 算法配置使用
`../../diffusion/weights/synthetic_ellipses/manifest.json`。

每次运行保存 `result.h5`、`run.json` 和 `metrics.json`；成功保存数值结果后还生成
`index.json`，启用预览时生成 `preview.png`。批量评估汇总已有运行记录，
不会重新求解。查看 `status`、`failure_reason`、`stages` 和 `stop_reason`，
并结合正演/伴随次数及耗时比较方法；三种算法的单次迭代成本不同。

## 数据约定

`FTTTCase` 由 `grid`、`geometry`、`measurement`、可选 `ground_truth` 和
`metadata` 组成。数组与坐标采用 `[y, x]`，网格形状为 `(ny, nx)`；
`origin_m` 是第一个像素的边界坐标，像素中心需加半个网格间距。
所有长度用 m、走时用 s、声速用 m/s。

`measurement` 限于 `domain="features"`、`observable="first_arrival"`。
必须且只能提供 `tof_s`（绝对首达走时）和 `delta_tof_s`（差分首达走时）之一，
同时提供同形状 `(n_tx, n_rx)` 的 `valid_mask`。
有效通道必须有限，绝对走时须非负；无效通道可以含 NaN。
差分走时还必须声明：

```python
metadata = {
    "reference_sound_speed_mps": 1500.0,
    "reference_tof_definition": "distance_over_c0",
}
```

这里差分定义为 `首达走时 - 发射接收距离 / c0`。
原始压力波形、互相关延迟、相位斜率延迟不能直接当作首达走时输入。
真实数据构建示例见 [开发与数据接入](docs/development.md)。

## DPS 权重范围

附带的 `synthetic_ellipses` 是 **16×16 合成图像**先验，仅用于接口和合成实验。
`openbreastus`、`nbpslice2d` 的 manifest 是 `untrained` 占位，尚无相应真实数据先验。
网格尺寸、归一化范围和模型结构须与 checkpoint 一致；加载器不会自动缩放或在缺失时
使用随机权重。DIP、INR 不依赖这些权重。

原生训练入口如下，输出位置须不存在：

```bash
python -m ftttbench.algorithms.fttt.train_prior --output checkpoints/my_prior.pt --steps 5000 --shape 16 16
```

完整算法、权重要求和适用范围见 [算法说明](docs/algorithm.md)。本库中的合成演示
不能证明实验数据上的成像效果。

## 目录与验证

```text
configs/algorithms/       三种算法的 YAML 配置
configs/benchmarks/       病例 × 算法的批量配置
src/ftttbench/core/       类型、配置、注册、HDF5、预算与来源记录
src/ftttbench/operators/  Eikonal 正演、线性化和 PyTorch 梯度桥接
src/ftttbench/models/     SkipUNet、SIREN、TinyUNet
src/ftttbench/solvers/    DIP、INR、DPS 的数值循环
src/ftttbench/algorithms/ FTTT 物理适配、损失、权重与训练入口
src/ftttbench/benchmark/  运行、保存和汇总
examples/                最小可复跑演示
tests/                   数据契约、物理、算法与命令行验证
diffusion/weights/       数据集 manifest 与合成先验
```

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

测试包含数值梯度/伴随一致性、掩模及留出通道、停止预算和权重验证。
测试通过表示对应接口与数值检查通过，不等于真实数据重建质量已达标。

第三方代码出处及许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

数据字段与文件结构见 [数据格式](docs/data_format.md)，已有 CPU 合成流程实测记录见 [验证记录](docs/validation.md)。
