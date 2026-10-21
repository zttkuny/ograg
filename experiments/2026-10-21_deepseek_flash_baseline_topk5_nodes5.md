# DeepSeek-V4.1-Flash 基线 top_k=5 / nodes_top_k=5 实验记录

- 日期：2026-10-21（Asia/Shanghai）
- 数据集：Soybean Reasoning，97 题
- 方法：`ontohypergraph-rag`（基线）
- 模型：`deepseek-flash`（DeepSeek-V4.1-Flash-1017）
- Embedding：DashScope `text-embedding-v4`
- 配置文件：`configs/hypergrag/config_soybean.yaml`
- 代码版本：`master` @ `9f9631d`
- 结果目录：`results/deepseek_flash_baseline_topk5_nodes5/`

## 1. 摘要

本次把基线方法 `ontohypergraph-rag` 的上下文预算固定为 `top_k=5 / nodes_top_k=5`，在 Soybean Reasoning 97 题上重新跑了一遍生成和 RAGAS 评测，用于和以下两个结果做对照：

- `experiments/2026-10-20_deepseek_flash_baseline.md`：旧基线，`top_k=2 / nodes_top_k=5`，每题 2 条上下文
- `experiments/2026-10-20_deepseek_flash_de_soybean.md`：DE，`top_k=5 / nodes_top_k=20`，平均 5.46 条上下文

核心结论：

1. 基线从 2 条上下文增加到 5 条后，`context_recall` 从 0.5069 提升到 **0.5739**（+0.0670，p≈0.012），是唯一统计显著的提升。
2. `answer_correctness`、`faithfulness`、`answer_similarity` 也有小幅上升，但 95% CI 均跨 0，不能认为稳定改善。
3. 与 DE 相比，在上下文数量基本对齐后，DE 只剩 `answer_correctness +0.0177`、`answer_similarity +0.0247` 的小幅差异，且全部不显著。
4. 因此，DE 之前相对旧基线的提升，主要来自上下文数量从 2 条增加到 5 条，而不是辩论机制本身。
5. 新基线平均 2.76 秒/题，DE 为 49.75 秒/题，DE 的额外时间成本约 18 倍。

## 2. 实验目的

1. 固定上下文预算，隔离“上下文数量”对基线指标的影响。
2. 为 DE 提供一个上下文数量更接近的对照基线。
3. 判断 DE 相对基线的小幅提升是否来自检索辩论机制。

## 3. 实验配置

### 3.1 模型与 embedding

| 项目 | 配置 |
|---|---|
| LLM | `deepseek-flash` |
| LLM API | `https://api.deepseek.com`，OpenAI 兼容 |
| `reasoning_effort` | `high` |
| `enable_thinking` | `False`（DeepSeek 实际未完全关闭隐藏推理） |
| Embedding | `text-embedding-v4` |
| Embedding API | DashScope OpenAI 兼容接口 |

### 3.2 检索配置

配置文件：`configs/hypergrag/config_soybean.yaml`

```yaml
query:
  method: ontohypergraph-rag
  hyperparams:
    vector_index: False
    top_k: 5
    nodes_top_k: 5
```

含义：

- `nodes_top_k=5`：按 query embedding 分别对 hypernode 的 key 和 value 排序，各取前 5 个，最多得到 10 个候选节点。
- `top_k=5`：从候选节点覆盖的 hyperedge/chunk 中选前 5 个，作为最终 context。
- 本次 97 题每题固定得到 **5 条上下文**。

### 3.3 数据

- 问题文件：`data/kg/soybean/questions/ontodoc_ragas/testset_reasoning.csv`
- 问题数：97
- 问题类型：soybean reasoning

## 4. 运行方式

生成：

```bash
conda run -n ograg python query_llm.py \
  --config_file configs/hypergrag/config_soybean.yaml \
  --query.hyperparams.top_k 5 \
  --query.hyperparams.nodes_top_k 5 \
  --results_dir results/deepseek_flash_baseline_topk5_nodes5
```

评测：

```bash
conda run -n ograg python test_answers.py \
  --config_file configs/hypergrag/config_soybean.yaml \
  --query.hyperparams.top_k 5 \
  --query.hyperparams.nodes_top_k 5 \
  --query.answers_file results/deepseek_flash_baseline_topk5_nodes5/soybean/ontodocragas_reasoning_answers_ontohypergraphrag_ontohypergraph_rag.json
```

说明：

- 生成脚本会自动追加方法名和超参后缀，实际文件为
  `ontodocragas_reasoning_answers_ontohypergraphrag_ontohypergraph_rag_vector_indexFalse_top_k5_nodes_top_k5.json`
- 评测阶段使用临时脚本把 `qna.tester.run_config.max_workers` 从默认的 1 改成 4；仓库默认值未修改。
- `test_answers.py` 会自动追加 `_vector_indexFalse_top_k5_nodes_top_k5` 后缀并找到对应 JSON。

## 5. 生成结果

| 项目 | 结果 |
|---|---:|
| 完成题数 | 97 / 97 |
| 报错数 | 0 |
| 空答案 | 0 |
| 总耗时 | 267.27 秒（约 4.45 分钟） |
| 平均耗时 | 2.76 秒/题 |
| 中位耗时 | 2.09 秒/题 |
| 最短耗时 | 0.99 秒 |
| 最长耗时 | 13.71 秒 |
| 每题 context 数 | 5（固定） |
| 平均答案长度 | 262.0 字符 |

与旧基线和 DE 的运行对比：

| 项目 | 旧基线 top_k2/nodes5 | 新基线 top_k5/nodes5 | DE top_k5/nodes20 |
|---|---:|---:|---:|
| 平均耗时 | 4.17 秒/题 | **2.76 秒/题** | 49.75 秒/题 |
| 中位耗时 | 2.17 秒 | 2.09 秒 | 44.28 秒 |
| 总耗时 | 404.5 秒 | **267.3 秒** | 4,825.6 秒 |
| 每题 context 数 | 2 | 5 | 5.46（范围 5–12） |
| 平均答案长度 | 242.2 字符 | 262.0 字符 | 259.8 字符 |

新基线比旧基线更快，但上下文更多。这更可能是 API 延迟、缓存命中和服务端波动造成的，不建议把速度差异当作方法差异。

## 6. 评测指标

评测共计算 582 个指标（97 题 × 6 指标），耗时 **53 分 22 秒**。

| 指标 | Mean | Std | Min | Max | 非空 |
|---|---:|---:|---:|---:|---:|
| answer_correctness | 0.5595 | 0.3313 | 0.0274 | 1.0000 | 97/97 |
| faithfulness | 0.7402 | 0.3801 | 0.0000 | 1.0000 | 97/97 |
| answer_similarity | 0.7643 | 0.2291 | 0.1096 | 1.0000 | 97/97 |
| answer_relevancy | 0.6158 | 0.3272 | 0.0000 | 1.0000 | 97/97 |
| context_precision | 0.5270 | 0.4463 | 0.0000 | 1.0000 | 97/97 |
| context_recall | 0.5739 | 0.4834 | 0.0000 | 1.0000 | 97/97 |

说明：

- `answer_relevancy` 因 DeepSeek 不支持 `n>1`，使用 `strictness=1`，与之前两次 DeepSeek 实验一致。
- `context_relevancy` 和 `context_entity_recall` 不在当前 RAGAS metric map 中，未计算。
- `faithfulness` 有部分题目为 0 分，标准差较大，解释均值时需要谨慎。

## 7. 与旧基线对比

旧基线结果来自 `experiments/2026-10-20_deepseek_flash_baseline.md`，模型、embedding、数据集相同，但上下文为每题 2 条。

| 指标 | 新基线 | 旧基线 | 差值 | 95% CI | p（配对 t） | 逐题 新胜/平/旧胜 |
|---|---:|---:|---:|---|---:|---:|
| answer_correctness | 0.5595 | 0.5196 | +0.0399 | [-0.0054, +0.0917] | 0.1137 | 32 / 21 / 44 |
| faithfulness | 0.7402 | 0.6915 | +0.0487 | [-0.0101, +0.1114] | 0.1161 | 24 / 57 / 16 |
| answer_similarity | 0.7643 | 0.7330 | +0.0313 | [-0.0026, +0.0675] | 0.0865 | 38 / 21 / 38 |
| answer_relevancy | 0.6158 | 0.6293 | -0.0135 | [-0.0681, +0.0394] | 0.6241 | 41 / 13 / 43 |
| context_precision | 0.5270 | 0.5464 | -0.0194 | [-0.0620, +0.0219] | 0.3709 | 11 / 57 / 29 |
| context_recall | 0.5739 | 0.5069 | **+0.0670** | [+0.0206, +0.1220] | **0.0120** | 8 / 88 / 1 |

关键观察：

- `context_recall` 是唯一 CI 不跨 0 的指标，8 胜 1 负 88 平，说明增加上下文主要补到了部分原先缺失的证据。
- 答案质量指标均值都有上升，但逐题胜负并不占优，且置信区间跨 0。
- `context_precision` 小幅下降，但同样不显著。
- 从 2 条增加到 5 条上下文，是本组变化中最明确的收益来源。

## 8. 与 DE 对比

DE 结果来自 `experiments/2026-10-20_deepseek_flash_de_soybean.md`，同为 97 题、`deepseek-flash`、`text-embedding-v4`，但 DE 使用 `top_k=5 / nodes_top_k=20`，平均 5.46 条上下文。

| 指标 | 新基线 | DE | DE - 新基线 | 95% CI | p（配对 t） | 逐题 DE胜/平/新基线胜 |
|---|---:|---:|---:|---|---:|---:|
| answer_correctness | 0.5595 | 0.5772 | +0.0177 | [-0.0299, +0.0662] | 0.4743 | 38 / 23 / 36 |
| faithfulness | 0.7402 | 0.7411 | +0.0009 | [-0.0632, +0.0641] | 0.9770 | 18 / 59 / 20 |
| answer_similarity | 0.7643 | 0.7890 | +0.0247 | [-0.0083, +0.0598] | 0.1633 | 36 / 24 / 37 |
| answer_relevancy | 0.6158 | 0.6342 | +0.0184 | [-0.0366, +0.0748] | 0.5165 | 44 / 10 / 43 |
| context_precision | 0.5270 | 0.5200 | -0.0070 | [-0.0523, +0.0390] | 0.7679 | 13 / 67 / 17 |
| context_recall | 0.5739 | 0.5747 | +0.0009 | [-0.0533, +0.0541] | 0.9750 | 6 / 86 / 5 |

关键观察：

- 在上下文数量基本对齐后，DE 相对新基线的优势非常小。
- `faithfulness` 和 `context_recall` 几乎完全相同。
- `answer_correctness` 和 `answer_similarity` 的差值也不显著，逐题胜负接近。
- DE 的主要代价仍然是时间：49.75 秒/题，约为新基线的 18 倍。

## 9. 结论

1. **上下文数量是旧基线到 DE 提升的主要解释变量**：把基线从 2 条上下文提升到 5 条后，`context_recall` 基本追平 DE。
2. **DE 的增量收益不显著**：在上下文数量接近的情况下，DE 相对新基线的所有指标差异都无法通过统计检验。
3. **DE 不是稳定全面优于基线**：correctness 和 similarity 略高，precision 略低，faithfulness 和 recall 基本持平。
4. **DE 的代价很高**：约 18 倍生成耗时，且上下文数仍可能扩展到 5–12 条。
5. **当前对比仍不是完全受控实验**：DE 的 `nodes_top_k=20`，新基线是 `nodes_top_k=5`，两者最终上下文数量仍不完全一致。

## 10. 局限与下一步

### 局限

- DE 仍是 `top_k=5 / nodes_top_k=20`，平均 5.46 条上下文，而本次基线固定 5 条，尚未完全对齐。
- 只跑了一次随机性样本，DeepSeek 的 `reasoning_effort=high` 仍有随机性。
- `answer_relevancy` 使用 `strictness=1`，与 RAGAS 默认值不同。
- RAGAS 只计算了 6 个指标，缺少 `context_relevancy` 和 `context_entity_recall`。
- 新基线速度反而快于旧基线，说明单次运行的耗时波动较大，不宜过度解读。

### 建议的下一步

1. 把 DE 也固定为 `top_k=5 / nodes_top_k=5` 再跑一次，完成严格受控对比。
2. 对 97 题做 paired bootstrap 或置信区间分析，而不只看均值。
3. 增加重复运行，评估 DeepSeek 输出的稳定性。
4. 如需论文级指标，补做 `context_relevancy` / `context_entity_recall`，或自定义 RAGAS 指标。

## 11. 成本与产物

### 11.1 Token 用量

生成阶段：

| 项目 | 数值 |
|---|---:|
| 输入 tokens | 88,035 |
| 其中 cache hit | 52,096 |
| 其中 cache miss | 35,939 |
| 输出 tokens | 37,601 |
| 其中 reasoning tokens | 31,872 |
| 总 tokens | 125,636 |

评测阶段：

| 项目 | 数值 |
|---|---:|
| 调用次数 | 1,164 |
| 输入 tokens | 1,178,505 |
| 其中 cache hit | 827,570 |
| 其中 cache miss | 350,935 |
| 输出 tokens | 1,027,851 |
| 其中 reasoning tokens | 889,380 |
| 总 tokens | 2,206,356 |

按 `experiments/2026-10-20_deepseek_flash_baseline.md` 中记录的 2026-10-20 off-peak 价格粗估：

- 生成约 `$0.028`（约 0.20 CNY）
- 评测约 `$0.672`（约 4.8 CNY）
- 合计约 **$0.70（约 5.0 CNY）**

实际扣费以 DeepSeek 账单为准，该数字仅用于量级参考。

### 11.2 产物

```text
results/deepseek_flash_baseline_topk5_nodes5/
├── query.log
├── eval.log
└── soybean/
    ├── ontodocragas_reasoning_answers_ontohypergraphrag_ontohypergraph_rag_vector_indexFalse_top_k5_nodes_top_k5.json
    ├── ..._eval_ragas.csv
    └── ..._eval_token_usage.json
```

- JSON：97 题答案、检索上下文、逐题耗时。
- CSV：6 个 RAGAS 指标的逐题分数。
- `query.log`：逐题回答、token usage、reasoning tokens、耗时。
- `eval.log`：RAGAS 评测过程和 warning。
- `*_eval_token_usage.json`：评测阶段按调用记录的 token 用量。
