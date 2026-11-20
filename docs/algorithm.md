# 算法与物理适配

本库公开算法为 `dip`、`inr`、`dps`，共用同一套首达走时数据和 Eikonal 物理。
`run("fttt", ..., {"parameters": {"variant": "inr"}})` 是兼容调用；新代码优先
使用 `run("inr", ...)`，并在独立输出目录保存配置和结果。

## 正演与导数

待重建量是声速 `c(y, x)`，正演自变量为慢度 `s = 1 / c`。
对每个发射位置求解：

```text
|∇T(y, x)| = s(y, x),   T(source) = 0
```

`EikonalForward` 使用 Cartesian fast marching，默认一阶迎风，
`eikonal_order=2` 可选择混合二阶离散。图像外的换能器通过固定背景水层扩展纳入
计算网格，在接收位置双线性采样。默认用相同离散网格的均匀水正演校准离散误差。
这是首达走时模型，不包含波形、振幅、衍射或多次到达的信息。

正演保存已接受节点的因果依赖。`EikonalJacobian.forward` 沿该依赖传播慢度
扰动，`adjoint` 逆序累积其转置。`TorchEikonalOperator` 用同一个离散导数完成
PyTorch 反向传播；Eikonal 求解在 CPU 上进行，网络可按配置运行在 CPU 或 CUDA。
路径依赖切换处只能按当前离散分支解释局部导数；网格和点源误差仍需通过加密网格
与数值检查评估。

差分观测严格采用 `delta_tof_s = tof_s - distance / c0`。
因此必须声明 `metadata.reference_sound_speed_mps` 及
`reference_tof_definition="distance_over_c0"`，并保证与配置中的参考声速一致。
任意实验水参考不能未经转换就套用这一差分定义。

## 有界变量与损失

网络估计归一化慢度 `u`，映射为：

```text
s_min = 1 / c_max
s_max = 1 / c_min
s(u) = s_min + clip(u, 0, 1) · (s_max - s_min)
c(u) = 1 / s(u)
```

DIP、INR 使用 sigmoid 输出；DPS 的干净图估计按上述区间截断。
默认声速界 `[1300, 1700]` m/s 是数值限制，不代表对特定组织的生理范围验证。
`roi_update_only=true` 时 ROI 外固定为参考介质；否则全图可更新。

训练损失为：

```text
L = 1/2 · sum_i [w_i · ((F_i(s(u)) - d_i) / residual_scale_s)^2] / sum_i w_i
```

求和只覆盖训练通道。`residual_scale_s` 是固定秒单位标度，默认 `1e-6`，
不从留出集估计。无效与留出通道先筛选，再计算残差，避免无效 NaN 进入梯度。
收发位置重合的通道不参与反演。留出通道用于独立记录残差和可选停止判断，
不进入训练损失；真值图也不进入求解器或停止规则。

## 三种求解方法

| 方法 | 更新内容 | 主要参数 | 一次迭代 |
| --- | --- | --- | --- |
| DIP | 固定随机输入下的 SkipUNet 参数 | `learning_rate`、`input_channels`、`base_channels` | 一次 Adam 更新 |
| INR | 坐标 SIREN 参数 | `learning_rate`、`hidden_features`、`hidden_layers` | 一次 Adam 更新 |
| DPS | 扩散样本 | `sampling_steps`、`guidance_scale`、`eta`、权重路径 | 一次反向采样更新 |

DIP、INR 针对单个病例从随机初始化优化，不需要预训练图像。
DPS 需要预训练的噪声预测器，使用 VP 日程和 DDIM 更新，再用训练通道的加权
归一化残差范数梯度引导采样。`guidance_scale` 作用于该范数的梯度，不能直接当作
DIP/INR 的学习率比较。`eta=0` 表示 DDIM 更新不额外注入噪声，初始样本仍由种子决定。

`sampling_steps` 指完整采样日程长度；`run_controls.max_iterations` 可以提前截断
该日程，但不会把剩余日程自动重排。DPS 不支持目标函数平台停止
`objective_rtol`。三方法耗时和算子次数不同，比较时同时报告工作量与时间。

## 配置和执行控制

```yaml
name: inr
parameters:
  learning_rate: 0.0001
  seed: 0
  residual_scale_s: 1.0e-6
  sound_speed_bounds_mps: [1300.0, 1700.0]
run_controls:
  max_iterations: 30
deployment:
  neural_device: cpu
```

`parameters` 是科学参数；`run_controls` 是迭代、时间、正演及伴随预算与停止规则；
`deployment` 是运行设备等宿主设置。实际可用参数及约束可查询：

```bash
fttt describe-algorithm dip
fttt describe-algorithm inr
fttt describe-algorithm dps
```

耗时预算在工作单元之间协作检查，不是操作系统级硬超时。
达到预算可保存已有可用状态；应结合 `status`、`stop_reason`、`completed_iterations`
和实际工作量判断运行是否满足实验设计，不能只看是否生成了图片。

## DPS 权重与训练

`checkpoint` 和 `prior_manifest` 二选一。manifest 相对路径在 YAML 中相对配置
文件解析，checkpoint 在 manifest 内相对 manifest 解析。
加载时检查 SHA256、模型结构、图像尺寸、慢度归一化、有限数值与噪声日程。
原生格式为 `usct.fttt.prior.v1`，保留该名称以兼容已有权重。

现有 `synthetic_ellipses` 是 16×16 合成椭圆先验，仅供合成接口实验。
其他数据集的 `untrained` manifest 不会被当作可用权重。
`registered` 表示已登记但尚未完成运行适配，`ready` 仅表示当前加载器支持格式。
外部 Score SDE 的 checkpoint 还需适配结构、时间标签、缩放和 score/epsilon
预测约定，不能仅重命名后使用。

可用原生工具训练自有声速图像集：

```bash
python -m ftttbench.algorithms.fttt.train_prior --images data/prior_images.npz --shape 128 128 --steps 5000 --output checkpoints/my_prior.pt
```

NPZ 包含 `train_speed_mps` 和 `validation_speed_mps`，均为 `[image, ny, nx]`，
单位 m/s，当前工具要求值处于 `[1300,1700]`。事先按病例划分训练、验证、测试集；
工具检查形状、数值和跨集合重复图像并记录哈希，但不能自动证明病例级独立性。
训练后的噪声预测误差和重建后的成像指标是不同证据，应分别报告。

## 结果解释

有真值时可计算 RMSE、PSNR、SSIM；无真值时只报告可获得的观测、执行和来源信息。
真实声速图不存在时，不应将观测拟合误差称为图像精度。
`evaluate` 汇总已保存的 `run.json` 和 `metrics.json`，不补造缺失分数。
同物理模型生成的合成病例主要验证执行流程，真实数据效果还需独立校准、数据接入和
实验比较。
