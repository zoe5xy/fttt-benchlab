# 病例与结果数据格式

本库的 `FTTTCase` 是二维首达走时专用契约。Python 模型使用 Pydantic 校验，
未知字段被拒绝；数组输入转换为实数 NumPy `float64`，掩模转换为布尔值。
建议通过 `write_case_hdf5` 写文件、`read_case_hdf5` 读取，避免手工写入缺失语义的 HDF5。

## 坐标与单位

| 数据 | 轴或分量 | 单位 |
| --- | --- | --- |
| 图像与 ROI | `[y, x]`，形状 `(ny, nx)` | 声速 m/s；ROI 无量纲 |
| 网格间距 | `(dy, dx)` | m |
| 网格原点 | `(y0, x0)`，第一个像素的边界 | m |
| 发射/接收位置 | `[transducer, coordinate]`，每行 `[y, x]` | m |
| 走时、掩模、权重 | `[tx, rx]` | s 或无量纲 |

像素 `(iy, ix)` 的中心为
`(y0 + (iy + 0.5) * dy, x0 + (ix + 0.5) * dx)`。
Eikonal 内部自变量为慢度 s/m，但输入真值和输出图像均为声速 m/s。
库不会自动把 mm、µs 或其他单位转换为 SI 单位。

## Python schema

`FTTTCase` 包含以下字段：

| 字段 | 要求 |
| --- | --- |
| `case_id` | 非空字符串 |
| `grid` | `GridSpec`，描述重建图像网格 |
| `geometry` | `GeometrySpec`，描述发射、接收位置 |
| `measurement` | `MeasurementSpec`，描述唯一一种首达走时表示与有效通道 |
| `ground_truth` | 可省略，默认没有真值图 |
| `metadata` | 可省略，默认空字典；差分走时必须填写参考定义 |

`GridSpec.shape` 两轴均至少为 2，`spacing_m` 必须有限且为正，`origin_m` 必须有限。
`roi_mask` 可省略；提供时形状须与网格一致，元素只能是布尔或 0/1。
ROI 不是有效通道掩模。是否只更新 ROI 内部由算法参数 `roi_update_only` 控制。

`GeometrySpec.type` 为 `ring`、`linear` 或 `custom`。
`tx_pos_m`、`rx_pos_m` 为非空有限实数数组，形状分别为 `(n_tx,2)` 和 `(n_rx,2)`。
可选 `radius_m` 必须有限且大于零。换能器不必位于重建图像内，正演可扩展背景网格。

`GroundTruthSpec.sound_speed_mps` 可省略；提供时须与 `grid.shape` 相同，有限且为正。
真值仅用于事后图像指标，运行器会在进入求解器前移除真值。

### 首达观测

```python
MeasurementSpec(
    domain="features",
    observable="first_arrival",
    tof_s=absolute_first_arrival_seconds,
    valid_mask=available_channels,
    ray_weights=optional_pick_confidence,
)
```

- `domain` 只接受 `"features"`，`observable` 只接受 `"first_arrival"`。
- 必须且只能提供 `tof_s`、`delta_tof_s` 之一，数组形状为 `(n_tx,n_rx)`。
- `valid_mask` 必填，形状相同，至少含一个真值；元素为布尔或 0/1。
- 有效通道的走时须有限；`tof_s` 有效值还须非负。差分走时允许负数。
- 无效通道的走时允许 NaN。算法先筛选通道再计算残差，不会用无效 NaN 参与训练。
- `ray_weights` 可省略；提供时形状相同，**全部元素**必须有限且处于 `[0,1]`，
  即使对应走时通道无效，权重本身也不能为 NaN。

`ray_weights` 是无量纲的首达拾取置信度，不是走时或声速。
算法结合 `min_ray_weight` 和 `ray_weight_power` 使用这些权重，零权重不能贡献损失。
不要把修改权重当作修正采集系统偏差。有效掩模通过 schema 校验后，仍可能因
收发重合或全部有效权重为零而没有可求解观测，算法会报告失败。

### 差分参考定义

差分病例必须提供：

```python
metadata = {
    "reference_sound_speed_mps": 1500.0,
    "reference_tof_definition": "distance_over_c0",
}
```

定义严格为：

```text
distance[tx,rx] = ||tx_pos_m[tx] - rx_pos_m[rx]||₂
reference_tof_s = distance / reference_sound_speed_mps
delta_tof_s = absolute_first_arrival_s - reference_tof_s
```

参考声速必须有限且为正，且不能与算法配置的参考声速冲突。
当前只支持 `"distance_over_c0"`，不支持把任意实验水参考曲线直接声明为该定义。
如果采集数据定义为 `样本首达 - 实测水首达`，需先根据系统标定转换为当前定义，
并保留转换过程。不要通过补一个 metadata 字段来改变已有数值的含义。

## 病例 HDF5 布局

当前 `schema_version` 为 `"0.1"`。文件写入结构如下，`@` 表示属性：

```text
/
  @schema_version = "0.1"
  @record_type = "FTTTCase"
  @case_id
  @metadata_json
  grid/
    @shape
    @spacing_m
    @origin_m
    roi_mask                       可选
  geometry/
    @type
    @radius_m                      可选
    tx_pos_m
    rx_pos_m
  measurement/
    @domain = "features"
    @observable = "first_arrival"
    tof_s 或 delta_tof_s            恰好一个
      @axes = "tx,rx"
      @units = "s"
    valid_mask
      @axes = "tx,rx"
      @units = "1"
    ray_weights                    可选
      @axes = "tx,rx"
      @units = "1"
  ground_truth/
    sound_speed_mps                可选
```

`metadata_json` 是序列化 JSON 字典，差分参考定义保存在这里。
`ground_truth` 组即使没有真值也由写入器创建；当前读取器要求该组存在。
`grid`、`geometry` 和真值的数据单位及轴由字段语义规定，当前写入器没有为这些
数据集另外写 `axes` / `units` 属性。measurement 的四类数据集则带上述属性。

读取器检查版本、所需组和属性，并按严格模型校验数组。
measurement 数据集的 `axes` / `units` 属性如果存在就校验其值；读取器当前允许
省略这两个属性，但新写文件应使用标准写入器保留它们。
`measurement.observable` 属性必须存在，不能只在 `metadata_json` 中声明。

最小读写：

```python
from ftttbench import read_case_hdf5, write_case_hdf5
from ftttbench.data.synthetic import make_first_arrival_case

case = make_first_arrival_case()
write_case_hdf5(case, "data/demo.h5")
restored = read_case_hdf5("data/demo.h5")
assert restored.measurement.valid_mask.shape == (12, 12)
```

Python 写入函数会创建父目录，并以写模式打开目标文件；它可以覆盖已有文件。
CLI 的 `simulate` 额外拒绝覆盖既有病例。
真实数据的 NPZ 转换示例见 [开发与数据接入](development.md)。

## 结果 HDF5 与运行文件

`ReconstructionResult` 包含 `algorithm`、`case_id`、`status`，以及可选的
`experiment_id`、`variant`、`sound_speed_mps`、`runtime_s`、`failure_reason`，
并包含 `metrics`、`artifacts` 字典。

`status` 为 `completed`、`budget_exhausted` 或 `failed`。
非失败结果必须有有限、为正的二维声速图；失败结果必须有 `failure_reason`，
并可能没有图像。达到预算时保留的图像不能据此解释为已收敛。

`result.h5` 根属性包含版本、记录类型 `ReconstructionResult`、算法与病例标识、
状态、耗时及 JSON 格式的指标/产物字典；可选属性包含实验 ID、variant 和失败原因。
根数据集 `sound_speed_mps` 可省略，存在时轴为 `[y,x]`、单位 m/s。
结果文件本身不嵌入完整病例网格，应与原病例和运行记录一起保留。

运行器同时保存：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 解析后的配置、状态、停止原因、阶段结果、工作量、采集来源与产物路径 |
| `metrics.json` | 数据残差、停止记录及有真值时可获得的图像指标 |
| `index.json` | 已保存可用数值结果的索引 |
| `preview.png` | 启用可视化时生成的预览 |

## 与 USCT 文件的关系

本库沿用 USCT BenchLab 的分组结构和运行组织方式，测量契约则收窄为首达走时。
**共享结构不保证旧 USCT HDF5 可以直接读取。** 原始宽泛 USCT 文件可能含频域压力、
RF 波形、多个延迟定义或缺失 `measurement.observable` 属性；这些内容不能自动
证明是本库要求的首达观测。

导入旧文件时，应明确挑选真正的首达字段，转换轴和单位、校正延迟、设置有效掩模，
按需要重新计算当前定义的差分参考，再构建 `FTTTCase` 并用本库写入器导出新文件。
不要把互相关延迟、相位斜率或原始压力数组仅改名为 `tof_s`。
