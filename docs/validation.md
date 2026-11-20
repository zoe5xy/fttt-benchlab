# 验证记录

本次源码交付检查：完整目录复制到新位置，在 Python 隔离模式下禁止导入其他项目，
**62 项测试通过（19.84 秒）**。目录内没有符号链接、分发压缩包或本机绝对路径依赖。

以下记录来自 2026-11-20 的既有 CPU 合成流程验证，使用 Python 3.12.14、
PyTorch 2.5.1、CPU 单线程。数值是该次运行的观测结果；接收源码目录后可按下方命令复跑。

- 完整测试：61 项通过（19.80 秒）。
- Ruff 静态检查与格式检查通过。
- 独立安装后的导入与三方法执行检查通过，DIP、INR、DPS 均能保存结果。
- 无 PyTorch 环境可发现算法、生成病例；执行神经重建时返回明确的依赖错误。

同一组 16×16、12 阵元的合成 Eikonal 观测，固定随机种子：

| 方法 | 更新数 | 走时相对残差 | 声速 RMSE (m/s) | 耗时 (s) |
| --- | ---: | ---: | ---: | ---: |
| DIP | 30 | 0.6394 | 44.18 | 11.31 |
| INR | 30 | 0.4552 | 13.68 | 8.91 |
| DPS | 16 | 1.8998 | 24.62 | 5.00 |

三个结果均为 `budget_exhausted / max_iterations`，表示按预算完成执行，不能解释为收敛。
DPS 使用随库附带的合成椭圆先验。这些数值仅用于合成流程核验，不构成真实乳腺数据效果结论或方法排名。

原始指标见 [validation.json](validation.json)。运行输出不随源码目录提供，复跑时会生成新的结果。

无需其他代码仓库。使用 Python 3.12 或更高版本，在收到的完整 `fttt-benchlab`
目录中安装运行依赖；测试依赖单独选择。复跑时选择新的输出目录：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,viz]'
python -m pip install -e '.[dev]'
python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python examples/fttt_demo.py --out runs/reproduce --iterations 30 --sampling-steps 16
```
