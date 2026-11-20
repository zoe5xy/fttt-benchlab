# 开发与数据接入

## 模块职责

| 目录 | 职责 |
| --- | --- |
| `core/` | 病例及结果 schema、HDF5、配置、算法注册、预算和来源记录 |
| `data/` | 合成病例；后续在此添加实验数据转换 |
| `operators/` | NumPy Eikonal、离散线性化与伴随、PyTorch 桥接 |
| `models/` | 仅负责网络结构 |
| `solvers/` | 与具体数据格式无关的 DIP、INR、DPS 数值循环 |
| `algorithms/fttt/` | 观测筛选、物理变量转换、损失、停止策略、权重和训练入口 |
| `evaluation/`、`metrics.py` | 通道划分、残差和可选真值指标 |
| `benchmark/` | 运行、保存、批量实验及汇总 |
| `cli.py` | 参数解析与 Python API 调用 |

模型与求解器无需导入病例 HDF5；数据转换也无需依赖神经模型。
公开算法 ID 维持 `dip/inr/dps`。注册信息应可在未安装 PyTorch 时查询，
只在实际运行神经算法时导入它。保留的 `fttt` 家族入口负责兼容旧调用。

## 开发环境与验证

本目录包含所需源码，不需要任何兄弟仓库。使用 Python 3.12 或更高版本，
在收到的 `fttt-benchlab` 目录中建立环境并安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,viz]'
```

参与开发或运行测试时，再安装开发依赖：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

纯 Python 数值循环可以执行，安装 `.[performance]` 可启用 Numba 加速；
首次编译会增加首轮耗时。小型 CPU 演示可限制线程以减少线程启动开销：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/fttt_demo.py --out runs/dev_demo --iterations 30 --sampling-steps 16
```

应优先检查：数据契约及 HDF5 往返；均匀介质与网格加密；Jacobian 有限差分、
伴随内积及 PyTorch 链式梯度；掩模与留出通道；预算和失败保存；三算法小型执行；
DPS 权重格式与缺失权重。测试数据和演示均为合成数据，不能替代实验验证。

## 构建真实数据病例

先在采集侧完成首达拾取、系统延迟校正、单位转换和坏通道标注。
本库读取首达走时，不从原始 RF 波形自动推断物理单位或拾取定义。
以下例子假设 `data/acquisition.npz` 已包含三个数组：

- `tx_pos_m`、`rx_pos_m`：`[n,2]`，各行是 `[y,x]` 米坐标。
- `tof_s`：`[n_tx,n_rx]`，绝对首达走时，单位秒。
- `valid_mask`：与 `tof_s` 同形状，布尔或 0/1。

```python
from pathlib import Path
import numpy as np
from ftttbench import (
    FTTTCase, GridSpec, GeometrySpec, MeasurementSpec, write_case_hdf5,
)

with np.load("data/acquisition.npz", allow_pickle=False) as source:
    case = FTTTCase(
        case_id="subject001_scan01",
        grid=GridSpec(
            shape=(128, 128),
            spacing_m=(0.0005, 0.0005),
            origin_m=(-0.032, -0.032),
        ),
        geometry=GeometrySpec(
            type="custom",
            tx_pos_m=source["tx_pos_m"], rx_pos_m=source["rx_pos_m"],
        ),
        measurement=MeasurementSpec(
            domain="features", observable="first_arrival",
            tof_s=source["tof_s"], valid_mask=source["valid_mask"],
        ),
        metadata={
            "reference_sound_speed_mps": 1500.0,
            "measurement_provenance": "external_measurement",
            "source_file": "acquisition.npz",
            "picker_method": "replace_with_actual_method_and_version",
            "time_zero_correction": "replace_with_actual_calibration",
        },
    )
Path("data").mkdir(exist_ok=True)
write_case_hdf5(case, "data/subject001_scan01.h5")
```

这里的网格仅是例子，需按真实视野、坐标原点和所需分辨率设置；
参考声速应使用实际标定值。没有真值时省略 `ground_truth`，运行仍可保存图像和
残差，图像真值指标保持缺失。

差分数据要用 `delta_tof_s` 替换 `tof_s`，并在 metadata 中额外写入
`reference_tof_definition="distance_over_c0"`。
必须确认输入数值确实为 `绝对首达 - 收发距离 / reference_sound_speed_mps`。
与实验水参考相减所得数值不一定符合该定义，应先完成对应校准转换。
交叉相关和相位延迟也不能仅改字段名就作为首达数据使用。

## 接入一个数据集

1. 在 `data/` 下添加明确的转换函数或脚本，记录输入格式、轴顺序、单位、
   坐标、首达拾取方法、系统延迟和有效通道规则。
2. 导出每个独立病例的 HDF5，检查通道数、位置、有效比例、走时范围和均匀水校准。
3. 先用 DIP/INR 小预算运行，再根据合适的网格和参数扩大实验。
4. DPS 需用独立训练病例训练匹配尺寸的先验，再登记 checkpoint 与哈希；
   真实数据 manifest 在此之前保持 `untrained`。
5. 固定病例划分、参数和预算后批量运行，报告失败及预算耗尽的实验；
   同时保存重建图、数据残差、运行耗时、工作量与来源信息。

训练集、验证集、测试集应按患者或独立病例划分，避免同一病例相邻切片跨集合。
不要用重建测试图训练 DPS，也不要用测试真值决定迭代停止。

## 配置和复跑

单算法配置集中放在 `configs/algorithms/`。可复制模板并修改迭代预算、学习率和
设备设置。配置中的相对文件路径按配置文件所在目录解析，批量配置中的病例通配符、
算法配置和输出根路径也相对批量 YAML 所在目录。

```yaml
name: my_dataset
case_glob: ../../data/my_dataset/*.h5
algorithms:
  - id: dip_default
    name: dip
    config: ../algorithms/fttt_dip.yaml
  - id: inr_default
    name: inr
    config: ../algorithms/fttt_inr.yaml
outputs:
  root: ../../runs
```

各实验 `id` 需唯一。同一方法不同超参数可使用不同 `id`；这不会增加算法种类。
新数据集未训练先验时，先只配置 DIP/INR；准备好真实先验后再加入 DPS。

```bash
fttt bench --suite configs/benchmarks/my_dataset.yaml
fttt evaluate --run runs/my_dataset_run
```

实际运行根目录由命令返回。保存原始采集和处理脚本版本、配置、随机种子、
checkpoint 哈希及运行记录，以便后续排查模型误差和数据误差。
