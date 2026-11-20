# 示例

所有命令从 `fttt-benchlab` 仓库根目录执行，先安装：

```bash
python -m pip install -e '.[neural,viz]'
```

## 一个病例、一个方法

```bash
fttt simulate --out data/demo.h5
fttt inspect --case data/demo.h5
fttt run inr --case data/demo.h5 --config configs/algorithms/fttt_inr.yaml --out runs/demo_inr
fttt evaluate --run runs/demo_inr
```

把 `inr` 和配置替换为 `dip` / `fttt_dip.yaml` 即可运行 DIP。
DPS 使用 `dps` / `fttt_dps.yaml`，默认先验仅匹配 16×16 网格。
命令拒绝覆盖已有病例或非空运行目录，新实验需另选输出路径。

## 三种算法的短演示

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/fttt_demo.py --out runs/three_methods --iterations 30 --sampling-steps 16
```

脚本生成一个带非中心夹杂的 16×16 Eikonal 合成病例，以固定种子运行 DIP、INR、DPS。
`--iterations` 控制 DIP/INR 的 Adam 更新数，`--sampling-steps` 控制 DPS 采样数。
可用 `--checkpoint /absolute/path/to/prior.pt` 显式指定匹配的原生 DPS 权重。

输出目录包含 `case.h5`、各方法的标准结果目录和 `summary.json`。
这是同模型合成执行演示，短预算的结果主要用于验证接入和比较实际工作量，
不代表实验数据上的性能排序。

## 批量实验与真实数据

```bash
fttt bench --suite configs/benchmarks/synthetic.yaml
```

先生成 `data/demo.h5`，批量配置再对匹配病例运行三种方法。
真实数据的坐标、单位、拾取和 HDF5 构建方法见
[开发与数据接入](../docs/development.md)；物理变量、损失与权重要求见
[算法说明](../docs/algorithm.md)。
