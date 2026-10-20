# DeepSeek-V4.1-Flash DE 全量实验记录

- 日期：2026-10-20（Asia/Shanghai）
- 数据集：Soybean Reasoning，97 题
- 方法：`de-ontohypergraph-rag`（DE，检索辩论增强）
- 模型：`deepseek-flash`（DeepSeek-V4.1-Flash-1017）
- Embedding：DashScope `text-embedding-v4`
- 配置文件：`configs/de/config_soybean_deepseek.yaml`
- 代码版本：`master` @ `58e5886`

## 1. 摘要

本次在 **Soybean Reasoning 97 题**上跑完了 DeepSeek-V4.1-Flash 的 DE 全量实验，生成和 RAGAS 评测均成功，无 API 报错、无空答案。

与 `2026-10-20_deepseek_flash_baseline.md` 中的同模型基线相比，DE 的答案质量指标全面小幅提升：

- `answer_correctness`：0.5196 → **0.5772**（+0.0576）
- `faithfulness`：0.6915 → **0.7411**（+0.0496）
- `answer_similarity`：0.7330 → **0.7890**（+0.0560）
- `context_recall`：0.5069 → **0.5747**（+0.0678）

但 `context_precision` 下降（0.5464 → 0.5200），且生成耗时从 4.17 秒/题增加到 **49.75 秒/题**，约 **11.9 倍**。因此 DE 的提升是“有但不算压倒性”，没有形成数量级上的鲜明优势。

## 2. 实验目的

1. 补全 DE 在 DeepSeek-V4.1-Flash 上的 97 题全量实验。
2. 与同一数据集、同一模型、同一 embedding 的 `ontohypergraph-rag` 基线做直接对比。
3. 判断 DE 的检索辩论是否值得其额外调用成本。

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

### 3.2 DE 检索配置

配置文件：`configs/de/config_soybean_deepseek.yaml`

```yaml
query:
  method: de-ontohypergraph-rag
  hyperparams: {}
```

`hyperparams: {}` 表示使用 DE 运行器默认值：

- `max_query_debate_rounds = 3`
- `top_k = 5`
- `nodes_top_k = 20`
- `vector_index = False`

DE 每题先做一次初始检索，然后最多进行 3 轮 Proponent / Opponent / Moderator 检索辩论；如果 Moderator 判定证据充分，则提前结束，最后用合并后的 context 生成答案。

### 3.3 数据

- 问题文件：`data/kg/soybean/questions/ontodoc_ragas/testset_reasoning.csv`
- 问题数：97
- 问题类型：soybean reasoning

## 4. 运行方式

生成：

```bash
conda run -n ograg python rag_engine_de_og_rag.py \
  --config_file configs/de/config_soybean_deepseek.yaml
```

评测：

```bash
conda run -n ograg python test_answers.py \
  --config_file configs/de/config_soybean_deepseek.yaml
```

评测阶段使用临时脚本把 `qna.tester.run_config.max_workers` 从默认的 1 改为 4；仓库默认值未修改。

## 5. 生成结果

| 项目 | 结果 |
|---|---:|
| 完成题数 | 97 / 97 |
| 报错数 | 0 |
| 空答案 | 0 |
| 总耗时 | 4,825.63 秒（约 80.4 分钟） |
| 平均耗时 | 49.75 秒/题 |
| 最短耗时 | 13.07 秒 |
| 最长耗时 | 193.84 秒 |
| 平均 context 数 | 5.46 |
| context 数范围 | 5–12 |
| 平均答案长度 | 259.8 字符 |

作为对比，同模型基线为：

| 项目 | DE | 基线 | 变化 |
|---|---:|---:|---:|
| 平均耗时 | 49.75 秒/题 | 4.17 秒/题 | 约 11.9 倍 |
| 平均 context 数 | 5.46 | 2 | +3.46 |
| 平均答案长度 | 259.8 字符 | 242.2 字符 | +17.6 |

## 6. 评测指标

评测共计算 582 个指标（97 题 × 6 指标），耗时约 50 分 12 秒。

| 指标 | Mean | Std | Min | Max | 非空 |
|---|---:|---:|---:|---:|---:|
| answer_correctness | 0.5772 | 0.3318 | 0.0461 | 1.0000 | 97/97 |
| faithfulness | 0.7411 | 0.3652 | 0.0000 | 1.0000 | 97/97 |
| answer_similarity | 0.7890 | 0.2040 | 0.1436 | 1.0000 | 97/97 |
| answer_relevancy | 0.6342 | 0.3053 | 0.0000 | 1.0000 | 97/97 |
| context_precision | 0.5200 | 0.4194 | 0.0000 | 1.0000 | 97/97 |
| context_recall | 0.5747 | 0.4849 | 0.0000 | 1.0000 | 97/97 |

说明：

- `answer_relevancy` 因 DeepSeek 不支持 `n>1`，使用 `strictness=1`，与基线文档一致。
- `context_relevancy` 和 `context_entity_recall` 不在当前 RAGAS metric map 中，未计算。

## 7. 与 DeepSeek 基线对比

基线结果来自 `experiments/2026-10-20_deepseek_flash_baseline.md`，同为 97 题、`deepseek-flash`、`text-embedding-v4`。

| 指标 | DE | 基线 | 差值 |
|---|---:|---:|---:|
| answer_correctness | 0.5772 | 0.5196 | **+0.0576** |
| faithfulness | 0.7411 | 0.6915 | **+0.0496** |
| answer_similarity | 0.7890 | 0.7330 | **+0.0560** |
| answer_relevancy | 0.6342 | 0.6293 | +0.0049 |
| context_precision | 0.5200 | 0.5464 | **-0.0264** |
| context_recall | 0.5747 | 0.5069 | **+0.0678** |

逐题胜负：

| 指标 | DE 胜 | 基线胜 | 平局 |
|---|---:|---:|---:|
| answer_correctness | 37 | 40 | 20 |
| faithfulness | 26 | 16 | 55 |
| answer_similarity | 38 | 39 | 20 |
| answer_relevancy | 45 | 43 | 9 |
| context_precision | 14 | 33 | 50 |
| context_recall | 10 | 2 | 85 |

关键观察：

- DE 的均值提升主要来自少数题目的明显改善，逐题胜负并不占压倒优势。
- `context_recall` 均值提升 0.0678，逐题是 10 胜 2 负 85 平，说明 DE 主要补到了部分原先完全缺失的证据。
- `context_precision` 反而下降，且 14 胜 33 负，说明辩论扩展增加了上下文数量，也引入了更多噪声。
- DE 的 context 数从 2 增加到 5.46，因此这组对比同时改变了“检索方法”和“上下文预算”，不能完全归因于辩论机制本身。

## 8. 与仓库旧 DE 结果对比

仓库已有 `results/soybean/ragas_answers_graphrag_de_ontohypergraph_rag_eval_ragas.csv`，同为 97 题，但其模型和 embedding 版本未在文件中记录，不能做严格归因。

| 指标 | DeepSeek DE | 旧 DE | 差值 |
|---|---:|---:|---:|
| answer_correctness | 0.5772 | 0.5707 | +0.0065 |
| faithfulness | 0.7411 | 0.8407 | -0.0996 |
| answer_similarity | 0.7890 | 0.7558 | +0.0332 |
| answer_relevancy | 0.6342 | 0.6976 | -0.0634 |
| context_precision | 0.5200 | 0.6415 | -0.1215 |
| context_recall | 0.5747 | 0.8131 | -0.2383 |

DeepSeek DE 的 `answer_correctness` 和 `answer_similarity` 略高，但 faithfulness、relevancy、context precision/recall 均低于旧 DE。由于旧 DE 的模型、embedding、上下文数量未记录，这只能作为参考。

## 9. 答案变化案例

### 9.1 DE 改善的题目

| 问题 | Ground Truth | 基线 | DE |
|---|---|---|---|
| Which disease causes chlorotic gray brown spots on leaves, leading to early defoliation and reduced photosynthesis? | RUST | Anthracnose | RUST |
| What is the name of the crop with a minimum germination of 70% ...? | Soybean | 拒答，称上下文没有相关信息 | Soybean |
| Which pesticide, for YMV control in Soybean crops, requires 25 ml/kg of seed? | Imidacloprid 48 FS | 拒答，错误引用 Thiamethoxam 30 FS | Imidacloprid 48 FS |
| Which pests cause drying of plants in linear patches ...? | White Grub, Gram Pod Borer, Tobacco Caterpillar, Green Semilooper, Girdle Beetle | Anthracnose 和 YMV | 正确列出 5 种害虫 |

### 9.2 DE 变差的题目

| 问题 | 基线 | DE |
|---|---|---|
| What are the stages of weed management in soybean cultivation and the recommended herbicides for each stage? | 列出 pre-plant / pre-emergence / post-emergence 及对应除草剂 | 上下文组织更零散，答案不完整 |
| What are the guidelines for harvesting soybean crops in Central India? | 明确给出 Central India 的 85–90% 豆荚变褐标准 | 称上下文没有 Central India 专门指南，只给通用建议 |

这说明 DE 的查询扩展既可能补到缺失证据，也可能把问题带偏或引入噪声。

## 10. 结论

1. **DE 可以稳定跑通完整 97 题**：无报错、无空答案，但平均耗时约 49.75 秒/题。
2. **DE 对答案质量有小幅提升**：correctness +0.058、faithfulness +0.050、similarity +0.056，context_recall +0.068。
3. **提升不是压倒性的**：逐题胜负接近，context_precision 反而下降；DE 不是稳定全面优于基线。
4. **主要收益来自补 recall**：context_recall 10 胜 2 负，说明辩论机制能补到部分原先缺失的证据。
5. **主要代价是时间和噪声**：约 11.9 倍生成耗时，context 数从 2 增加到 5.46，precision 下降。
6. **当前结果不能单独证明辩论机制有效**：上下文预算同时变化；需要固定 `top_k=2` / `nodes_top_k=5` 再跑一次 DE，才能隔离辩论本身的影响。

## 11. 产物与日志

```text
results/deepseek_flash_de_soybean/
├── answers_de_ontohypergraph_rag.json
├── answers_de_ontohypergraph_rag_eval_ragas.csv
├── query.log
└── eval.log
```

- JSON：97 题答案、检索上下文、逐题耗时。
- CSV：6 个 RAGAS 指标的逐题分数。
- `query.log`：DE 生成进度。
- `eval.log`：RAGAS 评测过程和 warning。

配置与文档尚未提交到 git。
