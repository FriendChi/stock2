# factor_analysis 流程说明

`tradition/factor_analysis/` 目录承载一条按阶段推进的研究流水线。当前共包含 6 个主流程，前一阶段的输出作为后一阶段的输入，目标是从数据预处理与单因子筛选逐步收敛到最终可执行的连续仓位策略回测结果。

## 流程 0：数据预处理

入口函数：

- `run_data_preprocess_single_fund`

所属模块：

- `selection.py`

作用：

- 下载或读取缓存基金数据
- 执行标准化、单基金过滤和价格序列适配
- 输出后续流程统一消费的预处理结果文件

输入：

- 基金代码
- 是否强制刷新下载（`--force-refresh`）

输出：

- `data_preprocess_*.csv`

说明：

- 流程 1-5 强制依赖该输出
- 只有流程 0 允许使用 `--force-refresh`

对应指令：

```bash
python -m tradition.runner research data-preprocess \
  --fund-code 007301 --force-refresh
```

## 流程 1：因子筛选

入口函数：

- `run_factor_selection_single_fund`

所属模块：

- `selection.py`

作用：

- 针对单只基金和给定因子族，生成参数化因子候选集合
- 在训练集和验证集上做单因子 IC / ICIR 评估
- 依据阈值规则筛出首轮可用因子

输入：

- 流程 0 输出路径（`--preprocess-path`）
- 因子族列表
- 单因子筛选阈值

输出：

- `factor_selection_*.json`

说明：

- 该阶段只解决“哪些单因子值得进入下一步”
- 不处理稳定性、不处理相关性冗余、不处理组合权重
- 禁止 `--force-refresh`，数据来源必须是流程 0

对应指令：

- 基于流程 0 输出执行：

```bash
python -m tradition.runner research factor-select \
  --preprocess-path /home/chi/snap/stock2/tradition/outputs/data_preprocess_007301_2026-04-12_abc12.csv \
  --factor-groups "均线趋势,趋势强度,波动调整趋势" \
  --train-min-spearman-ic 0.0 \
  --train-min-spearman-icir 0.0 --ic-exp-weighted
```

## 流程 2：单因子稳定性分析

入口函数：

- `run_single_factor_stability_analysis`

所属模块：

- `stability.py`

作用：

- 读取流程 1 的筛选结果
- 对入选单因子做稳定性评估
- 根据训练/验证一致性、翻转情况、尾部剔除规则，进一步过滤不稳定因子

输入：

- `factor_selection_*.json`

输出：

- `single_factor_stability_*.json`

说明：

- 该阶段只解决“筛出来的单因子是否稳定”
- 输出结果用于后续去冗余和组合搜索

对应指令：

- 基于流程 1 输出继续执行：

```bash
python -m tradition.runner research stability \
  --factor-selection-path /home/chi/snap/stock2/tradition/outputs/factor_selection_007301_2026-04-11_z0000.json  --ic-exp-weighted
```

## 流程 3：去冗余与组合选择

入口函数：

- `run_single_factor_dedup_selection`

所属模块：

- `dedup.py`

作用：

- 读取流程 2 的稳定性分析结果
- 对高相关因子做去冗余处理
- 在训练集上执行树形前向搜索
- 对前向选择结果做 Optuna 扩展搜索
- 在验证集上确定最终组合来源和最终因子集合

输入：

- `single_factor_stability_*.json`

输出：

- `single_factor_dedup_*.json`

说明：

- 该阶段输出的是“最终因子集合”
- 但还没有确定组合权重，也没有进入最终策略回测

对应指令：

- 基于流程 2 输出继续执行：

```bash
python -m tradition.runner research dedup \
  --stability-analysis-path /home/chi/snap/stock2/tradition/outputs/single_factor_stability_007301_2026-04-11_zl000.json \
  --dedup-root-topk 3  --ic-exp-weighted
```

## 流程 4：因子组合与权重微调

入口函数：

- `run_factor_combination`

所属模块：

- `combination.py`

作用：

- 读取流程 3 的最终因子集合
- 比较不同组合方式
- 在选定方法基础上对组合权重做 Optuna 微调
- 在验证集上选出最终权重方案

输入：

- `single_factor_dedup_*.json`

输出：

- `factor_combination_*.json`

说明：

- 该阶段固定因子集合，只优化组合权重
- 输出结果是后续策略回测使用的最终因子权重配置

对应指令：

- 基于流程 3 输出继续执行：

```bash
python -m tradition.runner research combination \
  --dedup-selection-path /home/chi/snap/stock2/tradition/outputs/single_factor_dedup_007301_2026-04-11_zlX00.json  --ic-exp-weighted
```

## 流程 5：连续仓位策略回测

入口函数：

- `run_strategy_backtest`

所属模块：

- `backtest.py`

作用：

- 读取流程 4 的最终因子权重方案
- 构建组合得分序列
- 针对多个连续仓位函数分别进行参数搜索
- 在训练集搜索、验证集定型、测试集比较
- 输出最终策略回测结果和权益曲线图

输入：

- `factor_combination_*.json`

输出：

- `strategy_backtest_*.json`
- `strategy_backtest_*.png`

说明：

- 该阶段是整条研究链路的最终落点
- 输出的是最终连续仓位策略在 train / valid / test 上的回测结果

对应指令：

- 基于流程 4 输出继续执行：

```bash
python -m tradition.runner research strategy-backtest \
  --factor-combination-path /home/chi/snap/stock2/tradition/outputs/factor_combination_007301_2026-04-11_zlXL0.json
```

- `2026-04-09` 对应流程 5 结果文件：

```text
/home/chi/snap/stock2/tradition/outputs/strategy_backtest_007301_2026-04-09.json
```

## 流程关系

六个流程的依赖顺序如下：

1. 数据预处理
2. 因子筛选
3. 单因子稳定性分析
4. 去冗余与组合选择
5. 因子组合与权重微调
6. 连续仓位策略回测

对应的输入输出链路如下：

1. `data_preprocess_*.csv`
2. `factor_selection_*.json`
3. `single_factor_stability_*.json`
4. `single_factor_dedup_*.json`
5. `factor_combination_*.json`
6. `strategy_backtest_*.json`

按当前命令行入口，对应子命令如下：

1. `python -m tradition.runner research data-preprocess`
2. `python -m tradition.runner research factor-select`
3. `python -m tradition.runner research stability`
4. `python -m tradition.runner research dedup`
5. `python -m tradition.runner research combination`
6. `python -m tradition.runner research strategy-backtest`

## 使用建议

- 若只想完成数据准备，运行到流程 0
- 若只想研究单因子质量，运行到流程 2 或流程 3 即可
- 若想得到最终因子集合，运行到流程 4
- 若想得到最终权重方案，运行到流程 5
- 若想得到最终策略结果，运行到流程 6

## 模块划分

当前目录中的模块职责如下：

- `common.py`
  - 跨流程复用的公共工具函数
- `selection.py`
  - 流程 1
- `stability.py`
  - 流程 2
- `dedup.py`
  - 流程 3
- `combination.py`
  - 流程 4
- `backtest.py`
  - 流程 5
- `io.py`
  - 各流程的输入输出与摘要打印
