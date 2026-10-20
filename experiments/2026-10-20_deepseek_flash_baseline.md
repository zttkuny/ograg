# DeepSeek-V4.1-Flash 基线实验记录

- 日期：2026-10-20（Asia/Shanghai）
- 数据集：Soybean Reasoning，97 题
- 方法：`ontohypergraph-rag`（基线）
- 模型：`deepseek-flash`（DeepSeek-V4.1-Flash-1017）
- Embedding：DashScope `text-embedding-v4`
- 代码版本：`master` @ `58e5886`（DeepSeek 适配提交；实验文档尚未提交）

## 1. 摘要

本次用 DeepSeek-V4.1-Flash 跑完了 Soybean Reasoning 的 97 题基线，生成和 RAGAS 评测均成功，无报错。生成阶段平均 4.17 秒/题，评测阶段约 43 分钟。

本次配置的 `context_recall` 为 0.5069，明显低于旧超图基线的 0.8038。主要原因是本次只检索 2 条上下文（`top_k=2` / `nodes_top_k=5`），而旧基线是 5 条上下文（`top_k=5` / `nodes_top_k=20`）；同时模型和 embedding 也不同，因此不能把差异直接归因于 DeepSeek Flash。

## 2. 实验目的

1. 验证 DeepSeek-V4.1-Flash 是否能完整跑通 `ontohypergraph-rag` 基线。
2. 记录生成质量、运行时间、token 消耗和费用。
3. 与仓库中已有的 97 题基线结果做初步对比，判断后续实验方向。

## 3. 实验配置

### 3.1 模型与 embedding

| 项目 | 配置 |
|---|---|
| LLM | `deepseek-flash` |
| LLM API | `https://api.deepseek.com` |
| LLM API 类型 | OpenAI 兼容 |
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
    top_k: 2
    nodes_top_k: 5
```

两个超参的含义：

- `nodes_top_k=5`：按 query embedding 分别对 hypernode 的 key 和 value 排序，各取前 5 个，最多得到 10 个候选节点。
- `top_k=2`：从候选节点覆盖的 hyperedge/chunk 中选前 2 个，作为最终 context。

因此本次每题最终只有 **2 条上下文**。旧超图基线结果是每题 5 条上下文（对应配置记录为 `top_k=5` / `nodes_top_k=20`）。

### 3.3 数据

- 问题文件：`data/kg/soybean/questions/ontodoc_ragas/testset_reasoning.csv`
- 问题数：97
- 问题类型：soybean reasoning

## 4. 运行方式

生成：

```bash
conda run -n ograg python query_llm.py \
  --config_file configs/hypergrag/config_soybean.yaml \
  --results_dir results/deepseek_flash_baseline
```

评测：

```bash
conda run -n ograg python test_answers.py \
  --config_file configs/hypergrag/config_soybean.yaml \
  --query.answers_file results/deepseek_flash_baseline/soybean/ontodocragas_reasoning_answers_ontohypergraphrag_ontohypergraph_rag.json
```

`test_answers.py` 会自动追加 hyperparams 后缀 `_vector_indexFalse_top_k2_nodes_top_k5`，从而找到实际生成的 JSON。

评测阶段为了缩短时间，使用临时脚本把 `qna.tester.run_config.max_workers` 从默认的 1 改成 4。仓库默认值没有修改；如果直接按上面的评测命令运行，耗时会明显更长。

## 5. 运行结果

### 5.1 生成阶段

| 项目 | 结果 |
|---|---:|
| 完成题数 | 97 / 97 |
| 报错数 | 0 |
| 总耗时 | 404.5 秒 |
| 平均耗时 | 4.17 秒/题 |
| 最短耗时 | 0.82 秒 |
| 最长耗时 | 131.12 秒 |
| 空答案 | 0 |
| 每题上下文数 | 2 |
| 平均答案长度 | 242.2 字符 |

### 5.2 评测指标

| 指标 | Mean | Std | Min | Max | 非空 |
|---|---:|---:|---:|---:|---:|
| answer_correctness | 0.5196 | 0.3324 | 0.0274 | 1.0000 | 97/97 |
| faithfulness | 0.6915 | 0.4061 | 0.0000 | 1.0000 | 97/97 |
| answer_similarity | 0.7330 | 0.2341 | 0.1096 | 1.0000 | 97/97 |
| answer_relevancy | 0.6293 | 0.3176 | 0.0000 | 0.9971 | 97/97 |
| context_precision | 0.5464 | 0.4792 | 0.0000 | 1.0000 | 97/97 |
| context_recall | 0.5069 | 0.4910 | 0.0000 | 1.0000 | 97/97 |

说明：

- `answer_relevancy` 因 DeepSeek API 不支持 `n>1`，本次使用 `strictness=1`；旧结果使用 RAGAS 默认 `strictness=3`，该指标不能直接横比。
- `context_relevancy` 和 `context_entity_recall` 在当前项目的 RAGAS metric map 中不存在，未计算，只打印 warning。

## 6. 与旧结果对比

### 6.1 与旧超图基线对比

旧超图基线：按仓库旧配置记录为 gpt-4o + text-embedding-3-small，每题 5 条上下文。旧结果文件与当前配置存在命名/版本歧义，精确来源应以当时的生成日志为准。

| 指标 | DeepSeek 本次 | 旧超图基线 | 差值 |
|---|---:|---:|---:|
| answer_correctness | 0.5196 | 0.5844 | -0.0648 |
| faithfulness | 0.6915 | 0.8205 | -0.1290 |
| answer_similarity | 0.7330 | 0.7586 | -0.0256 |
| answer_relevancy | 0.6293 | 0.5814 | +0.0480 |
| context_precision | 0.5464 | 0.6877 | -0.1414 |
| context_recall | 0.5069 | 0.8038 | -0.2969 |

逐题胜负：

| 指标 | DeepSeek 胜 | 旧基线胜 | 平局 |
|---|---:|---:|---:|
| answer_correctness | 38.1% | 49.5% | 11.3% |
| faithfulness | 21.6% | 32.0% | 46.4% |
| answer_similarity | 50.5% | 37.1% | 12.4% |
| answer_relevancy | 48.5% | 39.2% | 4.1% |
| context_precision | 35.1% | 38.1% | 26.8% |
| context_recall | 2.1% | 41.2% | 56.7% |

### 6.2 与旧 Ontology-RAG 基线对比

旧 Ontology-RAG：按仓库旧配置记录为 gpt-4o + text-embedding-3-small。它的 `retrieved_context` 是长字符串（平均约 1.97 万字符），与超图基线保存的 5 条结构化 context 不是同一种格式，不能按“上下文条数”直接对齐。

| 指标 | DeepSeek 本次 | 旧 Ontology-RAG | 差值 |
|---|---:|---:|---:|
| answer_correctness | 0.5196 | 0.3298 | +0.1898 |
| faithfulness | 0.6915 | 0.1979 | +0.4936 |
| answer_similarity | 0.7330 | 0.6332 | +0.0999 |
| answer_relevancy | 0.6293 | 0.7369 | -0.1076 |
| context_precision | 0.5464 | 0.9897 | -0.4433 |
| context_recall | 0.5069 | 0.9656 | -0.4588 |

结论与上一组类似：答案质量指标更高，但检索指标明显更低。这是“跨模型 + 跨 embedding + 跨上下文数量”的比较，不能用于证明 DeepSeek 基线优于或劣于旧基线。

## 7. 答案质量初步分析

对答案做简单的字符串层面统计：

| 项目 | DeepSeek 本次 | 旧超图基线 |
|---|---:|---:|
| 答案与 ground truth 规范化后完全一致 | 12/97 | 17/97 |
| 答案包含 ground truth | 20/97 | 29/97 |
| 含 "no/not ... context/information" 的拒答式回答 | 11/97 | 21/97 |
| 平均答案长度 | 242.2 字符 | 438.1 字符 |

可以看出：

- 本次答案更短、更简洁，拒答更少。
- 但完全匹配和包含 ground truth 的比例都低于旧基线。
- 与 `context_recall` 下降一致，说明 2 条上下文可能不足以覆盖多跳问题所需证据。

典型例子：

| 问题 | Ground Truth | DeepSeek 答案 | 判断 |
|---|---|---|---|
| Which pest can be controlled with Imidacloprid 48 FS pesticide in soybean crops? | YELLOW MOSAIC VIRUS (YMV) | White Grub | 错误 |
| What type of soil should be avoided for growing soybeans ...? | Sandy and heavy black soils | Sandy and heavy black soils should be avoided ... | 正确 |
| Which soybean crop is recommended for the Central Zone and has an early sowing time on January 1st, 2016? | JS 20-69 | No soybean crop in the provided context matches ... RVS 2002RVS -4 ... | 错误/拒答 |

第二个例子说明模型可以正确使用上下文；第一和第三个例子更可能是检索证据不足或排序错误，而不是模型完全不会回答。

## 8. 成本

### 8.1 生成阶段

| 项目 | 数值 |
|---|---:|
| 输入 tokens | 48,798 |
| 其中 cache hit | 25,464 |
| 其中 cache miss | 23,334 |
| 输出 tokens | 38,487 |
| 其中 reasoning tokens | 33,283 |
| 总 tokens | 87,285 |

按 DeepSeek 官方 off-peak 价格计算：

```text
(25,464 / 1M) * $0.003
+ (23,334 / 1M) * $0.15
+ (38,487 / 1M) * $0.6
= $0.0267
≈ 0.19 CNY
```

如果全部按 peak 价格计算，约为 `$0.0533 ≈ 0.38 CNY`。本次运行时间处于 off-peak。

### 8.2 评测阶段

评测由 DeepSeek Flash 作为 RAGAS judge，共 582 个指标计算（97 题 × 6 指标），耗时约 43 分钟。

DeepSeek 账户余额：

- 评测前：52.18 CNY
- 评测后：48.47 CNY
- 差额：约 **3.71 CNY**

### 8.3 总成本

| 阶段 | 费用 |
|---|---:|
| 生成 | 约 0.19 CNY |
| 评测 | 约 3.71 CNY |
| 合计 | 约 **3.90 CNY**（约 $0.55） |

折合约 **0.04 元/题**（含评测）。评测费用远高于生成费用，主要因为 RAGAS judge 需要多次调用模型并生成长推理。

价格来源：<https://api-docs.deepseek.com/quick_start/pricing>（2026-10-20 查询）。

## 9. 分析结论

1. **DeepSeek-V4.1-Flash 可以稳定跑通完整基线**：97 题全部完成，无 API 错误、无空答案，生成速度约 4.17 秒/题。
2. **本次基线质量低于旧超图基线**：`answer_correctness` -0.065，`faithfulness` -0.129，`context_recall` -0.297。
3. **最大问题在检索，不在生成**：本次每题只有 2 条上下文，`context_recall` 仅 0.507；旧基线有 5 条上下文，`context_recall` 0.804。上下文数量差异是最直接的解释。
4. **不能直接得出“DeepSeek 比 gpt-4o 差”的结论**：模型、embedding、上下文数量三个变量同时变化，属于混杂比较。
5. **成本很低**：完整生成约 0.19 元，含 RAGAS 评测约 3.9 元；适合先做小规模探索和超参筛选。

## 10. 局限与下一步

### 局限

- 旧基线和新基线的模型、embedding、上下文数量均不同，不能做因果归因。
- `answer_relevancy` 的 `strictness` 不同，不能直接横比。
- RAGAS 只计算了 6 个指标，`context_relevancy` 和 `context_entity_recall` 缺失。
- 本次只跑了一个随机性样本；DeepSeek 的 `reasoning_effort=high` 仍有随机性，单次结果不足以判断稳定性。
- 字符串层面的“完全匹配/包含 ground truth”只是粗粒度统计，不能替代人工评估。

### 建议的下一步

1. **公平复现旧基线超参**：用 `top_k=5` / `nodes_top_k=20` 再跑一次 DeepSeek 基线，隔离上下文数量的影响。
2. **固定 embedding 后再比较模型**：同一 retrieval 配置下分别跑 DeepSeek Flash 和旧模型，才能判断模型差异。
3. **补全 DE 的 DeepSeek 全量实验**：目前 DE 只有单题 smoke test；需要 97 题完整跑分后再与基线比较。
4. **增加配对显著性分析**：对 97 题做 paired bootstrap 或置信区间，而不是只看均值。
5. **如需论文级指标**：补做 `context_relevancy` / `context_entity_recall`，或升级/自定义 RAGAS 指标。

## 11. 产物与日志

```text
results/deepseek_flash_baseline/
├── query.log
├── eval.log
└── soybean/
    ├── ontodocragas_reasoning_answers_ontohypergraphrag_ontohypergraph_rag_vector_indexFalse_top_k2_nodes_top_k5.json
    └── ontodocragas_reasoning_answers_ontohypergraphrag_ontohypergraph_rag_vector_indexFalse_top_k2_nodes_top_k5_eval_ragas.csv
```

- `query.log`：逐题回答、token usage、reasoning tokens、耗时。
- `eval.log`：RAGAS 评测过程和 warning。
- JSON：97 题答案、检索上下文、逐题耗时。
- CSV：6 个 RAGAS 指标的逐题分数。

本次结果和文档均未提交到 git。
