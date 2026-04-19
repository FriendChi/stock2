# tradition/factor_analysis 功能介绍

`tradition/factor_analysis/` 目录实现了一条面向单基金研究的分阶段因子分析流水线。它的主要目标不是直接产出单个技术指标，而是把“原始基金及关联标的数据”逐步收敛成“可回测的连续仓位策略方案”。

这条流水线的核心思路是：

1. 先把输入数据整理成统一的特征矩阵。
2. 再从大量候选因子中筛出有预测能力的单因子。
3. 然后检验这些单因子的稳定性，剔除不稳定项。
4. 再做去冗余和组合选择，得到更精炼的候选集合。
5. 最后确定组合权重，并进入连续仓位策略回测。

## 目录主要作用

这个目录主要解决以下问题：

- 把基金净值及关联标的价格序列转成统一可复用的研究输入。
- 用统一的 walk-forward 切分方式评估因子，而不是只看单段样本。
- 在进入最终策略前，分阶段完成筛选、稳定性检验、去冗余、组合和权重优化。
- 将每个阶段的结果固化为文件，供下一阶段继续消费，避免重复计算。

从工程视角看，这个目录承担的是“研究流程编排层”的角色。它把因子构造、评价指标、组合逻辑、文件读写和最终回测串成一条可重复执行的研究链路。

## 总体流程

当前实际流程共分 6 个阶段：

1. 流程 0：特征预处理
2. 流程 1：因子筛选
3. 流程 2：单因子稳定性分析
4. 流程 3：去冗余与前向选择
5. 流程 4：因子组合与权重微调
6. 流程 5：连续仓位策略回测

它们的依赖关系是严格串联的：

```text
流程0 -> 流程1 -> 流程2 -> 流程3 -> 流程4 -> 流程5
```

每一阶段都会把结果落盘，下一阶段只读取上游文件，不直接复算整条链路。

## 流程 0：特征预处理

入口函数：

- `run_feature_preprocess_single_fund`

对应命令：

```bash
python -m tradition.runner research data-preprocess --fund-code 007301 --force-refresh
```

主要功能：

- 读取目标基金及其关联标的数据。
- 构建宽表特征矩阵。
- 对原始列做检查、补齐和因子扩展。
- 输出后续流程统一使用的特征 CSV 和元信息 JSON。

输入：

- 基金代码
- 是否强制刷新数据
- 配置中的关联标映射和多因子参数

输出：

- `feature_preprocess_*.csv`
- `feature_preprocess_*.json`

这一阶段是整条链路的基础。它不直接判断因子好坏，而是负责把后续研究所需的数据形态准备好。JSON 元信息中会显式记录：

- 特征 CSV 路径
- 目标价格列
- 目标净值列
- 可用因子列
- 原始特征列
- 数据质量摘要

因此从流程 1 开始，后续阶段主要消费的是“流程 0 产出的特征结果”，而不是重新从原始行情侧回溯生成整套输入。

## 流程 1：因子筛选

入口函数：

- `run_factor_selection_single_fund`

对应命令：

```bash
python -m tradition.runner research factor-select \
  --preprocess-metadata-path /path/to/feature_preprocess_xxx.json \
  --preprocess-path /path/to/feature_preprocess_xxx.csv \
  --factor-groups "均线趋势,趋势强度,波动调整趋势" \
  --train-min-spearman-ic 0.0 \
  --train-min-spearman-icir 0.0 \
  --ic-exp-weighted
```

主要功能：

- 从流程 0 的特征矩阵中读取候选因子列。
- 基于 walk-forward 切分，对每个候选因子分别计算训练集和验证集的 IC / ICIR。
- 根据阈值规则筛出首轮可用因子。

输入：

- 流程 0 特征 CSV
- 流程 0 元信息 JSON
- 因子组配置
- 单因子阈值配置

输出：

- `factor_selection_*.json`

这一阶段的核心问题是：“哪些单因子值得进入下一步”。  
它只做单因子质量筛选，不负责：

- 稳定性判断
- 相关性去冗余
- 因子组合权重
- 最终策略回测

## 流程 2：单因子稳定性分析

入口函数：

- `run_single_factor_stability_analysis`

对应命令：

```bash
python -m tradition.runner research stability \
  --factor-selection-path /path/to/factor_selection_xxx.json \
  --ic-exp-weighted
```

主要功能：

- 读取流程 1 的入选单因子。
- 继续在 walk-forward 结构下检查训练集和验证集的表现一致性。
- 识别翻转、不稳定和尾部失真的因子。
- 剔除不稳定因子，保留更稳健的单因子候选。

输入：

- `factor_selection_*.json`

输出：

- `single_factor_stability_*.json`

这一阶段解决的问题是：“这些已经通过首轮筛选的因子，是否足够稳定，值得继续投入到组合搜索中”。

## 流程 3：去冗余与前向选择

入口函数：

- `run_single_factor_dedup_selection`

对应命令：

```bash
python -m tradition.runner research dedup \
  --stability-analysis-path /path/to/single_factor_stability_xxx.json \
  --dedup-root-topk 3 \
  --ic-exp-weighted
```

主要功能：

- 读取流程 2 留下的稳定因子。
- 先按相关性做去冗余，去掉高度重复的信息源。
- 再执行树形前向搜索，寻找更优的因子子集。
- 对前向选择结果做 Optuna 扩展搜索。
- 在验证集上确定最终因子组合来源。

输入：

- `single_factor_stability_*.json`

输出：

- `single_factor_dedup_*.json`

这一阶段的重点不是调权重，而是确定“最终应该保留哪些因子”。  
换句话说，到这一阶段结束时，因子集合基本收敛，但组合权重还未最终定型。

## 流程 4：因子组合与权重微调

入口函数：

- `run_factor_combination`

对应命令：

```bash
python -m tradition.runner research combination \
  --dedup-selection-path /path/to/single_factor_dedup_xxx.json \
  --ic-exp-weighted
```

主要功能：

- 读取流程 3 的最终因子集合。
- 比较不同组合方式。
- 在固定因子集合的前提下，对组合权重进行微调。
- 在验证集上选出最终权重方案。

输入：

- `single_factor_dedup_*.json`

输出：

- `factor_combination_*.json`

这一阶段解决的是“同一组因子如何组合、如何赋权”的问题。  
因此它的重点已经从“选因子”转为“定权重”。

## 流程 5：连续仓位策略回测

入口函数：

- `run_strategy_backtest`

对应命令：

```bash
python -m tradition.runner research strategy-backtest \
  --factor-combination-path /path/to/factor_combination_xxx.json
```

主要功能：

- 读取流程 4 的最终因子权重方案。
- 生成组合得分序列。
- 针对多个连续仓位函数进行搜索和比较。
- 在训练集搜索、验证集定型、测试集比较。
- 输出最终策略回测结果和权益曲线。

输入：

- `factor_combination_*.json`

输出：

- `strategy_backtest_*.json`
- `strategy_backtest_*.png`

这是整条研究链路的最终落点。  
它把上游各阶段沉淀下来的因子集合与权重方案，最终转成可以直接比较的策略收益、风险和权益曲线结果。

## 文件流转关系

从文件产物角度看，整条链路的流转关系如下：

1. `feature_preprocess_*.csv`
2. `feature_preprocess_*.json`
3. `factor_selection_*.json`
4. `single_factor_stability_*.json`
5. `single_factor_dedup_*.json`
6. `factor_combination_*.json`
7. `strategy_backtest_*.json`
8. `strategy_backtest_*.png`

其中最关键的设计点是：

- 流程 0 输出的是“特征数据 + 元信息”双文件结构。
- 从流程 1 开始，后续阶段通过 JSON 结果文件逐步传递上游产物路径、基金代码、候选因子、筛选摘要和最终方案。
- 各阶段不是共享内存对象，而是通过文件接口解耦。

这种设计的收益是：

- 便于中途停止后续跑
- 便于回溯某一阶段的输入和输出
- 便于比较不同日期、不同基金、不同参数的阶段性结果

## 命令入口关系

当前命令行入口统一由 `tradition.runner` 提供，研究流程命令组为：

```bash
python -m tradition.runner research data-preprocess
python -m tradition.runner research factor-select
python -m tradition.runner research stability
python -m tradition.runner research dedup
python -m tradition.runner research combination
python -m tradition.runner research strategy-backtest
```

需要注意的是：

- `research data-preprocess` 现在实际对应的是“特征预处理”流程，而不是旧版本中那种仅输出价格预处理 CSV 的流程。
- `factor-select` 当前已经支持基于流程 0 元信息 JSON 恢复特征 CSV 路径，因此流程接口比旧版更偏向“特征矩阵驱动”。

## 模块职责

目录中的主要模块职责如下：

- `feature_preprocess.py`
  - 流程 0 的主入口
  - 负责把原始输入整理成特征矩阵，并输出元信息
  - 内部承载原始特征抓取、宽表检查、缺失值修补和因子扩展逻辑

- `selection.py`
  - 流程 1 的主入口
  - 负责单因子筛选与首轮候选确定

- `stability.py`
  - 流程 2 的主入口
  - 负责稳定性检验与尾部剔除

- `dedup.py`
  - 流程 3 的主入口
  - 负责相关性去冗余、前向搜索和扩展搜索

- `combination.py`
  - 流程 4 的主入口
  - 负责组合方式比较与权重微调

- `backtest.py`
  - 流程 5 的主入口
  - 负责连续仓位函数搜索与最终策略回测

- `common.py`
  - 跨流程公共函数
  - 包括 IC 聚合、forward return 构造、实例组合得分构造等复用逻辑

- `io.py`
  - 统一处理阶段结果文件的保存、读取、命名和摘要打印

## 推荐理解顺序

如果你第一次阅读这个目录，建议按下面顺序理解：

1. 先看 `feature_preprocess.py`，理解流程 0 到底产出了什么。
2. 再看 `selection.py`，理解候选因子如何被评估。
3. 然后看 `stability.py` 和 `dedup.py`，理解因子是如何从“可用”收敛到“最终集合”的。
4. 再看 `combination.py`，理解固定因子集合后如何比较组合方式和优化权重。
5. 最后看 `backtest.py`，理解最终组合如何映射为连续仓位策略并完成回测。

## 使用建议

按使用目标划分，可以这样理解各阶段的价值：

- 如果只想准备研究输入，运行到流程 0 即可。
- 如果只想知道哪些单因子值得保留，运行到流程 1。
- 如果只想得到更稳健的单因子候选，运行到流程 2。
- 如果想得到最终因子集合，运行到流程 3。
- 如果想得到最终因子权重方案，运行到流程 4。
- 如果想得到最终策略表现，运行到流程 5。

## 总结

`tradition/factor_analysis/` 的核心不是单点算法，而是一条可复用、可追踪、可分阶段落盘的研究流水线。

它把量化研究过程拆成了几个边界清晰的阶段：

- 流程 0 负责准备特征输入。
- 流程 1 到流程 3 负责筛因子。
- 流程 4 负责定权重。
- 流程 5 负责做最终策略回测。

因此，这个目录更适合作为“研究生产线”来理解，而不是作为零散指标脚本集合来理解。
