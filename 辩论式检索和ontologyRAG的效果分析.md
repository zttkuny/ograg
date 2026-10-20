# DEOntoHyperGraphRAG 与 ontology-rag 评测结果原因分析

## 1. 对比对象

本文对比两组 RAGAS 评测结果：

- `ontology-rag`（对应用户当前提供的 `ragas_answers_graphrag_eval_ragas` 结果）
- `DEOntoHyperGraphRAGQueryEngine`

两组均在 soybean reasoning 问题集上进行了评估，但需要注意，它们**并不是严格受控的单变量实验**，因为配置中的模型、embedding 和检索超参数并不一致。因此，下述分析应理解为“当前实验设置下的原因解释”，而不是对算法本体的完全隔离结论。

## 2. 结果现象

### 2.1 指标对比

| 指标 | ontology-rag | DEOntoHyperGraphRAG | 差值（DE - ontology） |
| --- | ---: | ---: | ---: |
| answer_correctness | 0.3264 | 0.5844 | +0.2580 |
| faithfulness | 0.1351 | 0.8205 | +0.6854 |
| answer_similarity | 0.6332 | 0.7586 | +0.1255 |
| answer_relevancy | 0.7369 | 0.5814 | -0.1556 |
| context_precision | 0.9897 | 0.6877 | -0.3019 |
| context_recall | 0.9688 | 0.8038 | -0.1650 |

### 2.2 直接结论

从结果上看，`DEOntoHyperGraphRAG` 在以下方面明显更强：

- 最终答案正确性更高
- 回答与证据的一致性更强
- 回答与参考答案更接近

而 `ontology-rag` 在以下方面更强：

- 检索到的上下文更精准
- 检索到的上下文覆盖更高
- 回答与问题表面语义更贴近

这说明两者的差异不在于“谁绝对更好”，而在于：

- `ontology-rag` 更像是**检索强、生成弱**
- `DEOntoHyperGraphRAG` 更像是**检索更保守但生成更稳**

## 3. 为什么会出现这种结果

## 3.1 第一层原因：DE 不是简单检索，而是“检索辩论 + 证据补全”

基础 `OntoHyperGraphQueryEngine` 的流程比较直接：

1. 根据 query 做一次超图检索
2. 取回 `top_k` 个相关超边
3. 直接把这些结构化事实送给 LLM 生成答案

对应实现见：

- [query_engine/ontograph_query_engine.py](file:///home/liqiankun/data/ograg/query_engine/ontograph_query_engine.py#L358-L392)

而 `DEOntoHyperGraphRAGQueryEngine` 在此基础上增加了一个 retrieval debate 阶段：

1. 先执行一次检索
2. 由正反方围绕“证据是否足够”进行辩论
3. 反方可以提出 query optimization 或 query expansion
4. moderator 决定是否继续检索
5. 多轮检索结果合并后，再交给 LLM 回答

对应实现见：

- [rag_engine_de_og_rag.py](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L161-L284)
- [retrieve_debate.py](file:///home/liqiankun/data/ograg/retrieve_debate.py#L17-L117)

这会直接带来两个效果：

- 更容易补到回答所需的关键属性和多跳事实
- 回答更容易围绕证据展开，而不是只靠模型自行补全

这就是 `faithfulness` 和 `answer_correctness` 明显提升的第一原因。

## 3.2 第二层原因：DE 更强调 ontology grounding，而不仅仅是关键词相关性

在 DE 版本中，辩论阶段关心的不只是“当前内容像不像问题”，而是：

- concept coverage
- concept-type consistency
- multi-hop evidence sufficiency
- minimality

也就是说，它在判断检索结果是否足够时，会优先考虑：

- 是否覆盖了问题中的关键本体概念
- 概念类型是否匹配
- 是否具备完成推理所需的多跳证据
- 是否存在明显冗余或噪声

对应实现见：

- [rag_engine_de_og_rag.py](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L173-L220)

因此，DE 的优势主要不在“召回更多文本”，而在：

- 更偏向补齐结构化事实
- 更偏向确保回答可由当前证据支撑

这类机制天然有利于提升：

- `answer_correctness`
- `faithfulness`
- `answer_similarity`

## 3.3 第三层原因：ontology-rag 的上下文往往是对的，但回答容易超出证据发挥

从评测结果样例看，`ontology-rag` 的 `context_precision` 和 `context_recall` 很高，说明它经常能取回正确的结构化事实。

但是，它的 `faithfulness` 却很低，说明一个典型现象：

> 检索到的证据是对的，但模型生成答案时加入了额外推断、扩展说明或与当前证据无关的内容。

例如，在 `ragas_answers_graphrag_eval_ragas.csv` 的首个样例中，上下文本身已经包含 `YELLOW MOSAIC VIRUS (YMV)`，但回答中又引入了 Rust disease、区域种植说明、综合病虫害策略等额外信息。这会导致：

- 检索指标仍然很高
- 但回答与上下文的逐项对齐程度下降
- 最终 `faithfulness` 被显著拉低

参考样例：

- [results/soybean/ragas_answers_graphrag_eval_ragas.csv:L1-L8](file:///home/liqiankun/data/ograg/results/soybean/ragas_answers_graphrag_eval_ragas.csv#L1-L8)

因此，`ontology-rag` 当前更像是：

- 上下文找得准
- 但生成阶段缺乏对“只依据证据回答”的约束

## 3.4 第四层原因：DE 的上下文更大、更杂，因此检索指标反而下降

虽然 DE 通过辩论补齐了证据，但它也把多个 query 的结果池合并后再回答。这样做会带来一个副作用：

- 上下文数量变多
- 候选边更多
- 冲突事实和噪声边也更容易混入

这会直接拉低：

- `context_precision`
- `answer_relevancy`

原因并不复杂。RAGAS 的这两个指标更偏向衡量：

- 取回的上下文是否紧贴当前问题
- 回答是否紧贴问题本身

但 DE 的目标更接近：

- 为了保证答案可证据化，允许多取一些相关边
- 为了支持多跳推理，允许一定冗余

因此，DE 在答案质量上提升的同时，检索指标下降是可以解释的，不是矛盾，而是典型的检索-生成权衡。

## 3.5 第五层原因：当前 ontology 本身存在重复、冲突和粒度不一致的问题

从样例可以看到，当前知识表示中存在以下现象：

- 同一属性存在多个不同版本，例如 `70% germination` 与 `80% germination`
- 同一概念有时被放在 pest treatment，有时又出现在 disease treatment
- 实体命名不一致，例如 `Imidacloprid` 与 `Imidachloprid`
- 同一 zone 或 variety 的事实边粒度不一致，容易混入近似但不相等的候选

这会导致两个后果：

1. 对于 `ontology-rag`：
   - 检索能取回某条“局部正确”的边
   - 但模型在生成时容易跨边整合，进而引入不被当前证据支持的内容

2. 对于 DE：
   - 辩论机制会进一步扩大证据池
   - 虽然更容易补到正确线索，但也更容易把冲突事实一起送入生成阶段

也就是说，**当前知识图谱质量本身就是影响结果分裂的重要原因**。

## 4. 一个更准确的解释框架

如果只看这组结果，最合理的解释不是：

> DE 的检索比 ontology-rag 差

而是：

> DE 的检索策略更像“为回答服务的证据组织”，ontology-rag 的检索策略更像“单次命中最相关事实”。

因此：

- `ontology-rag` 的强项是单次检索质量
- `DEOntoHyperGraphRAG` 的强项是证据驱动的最终回答质量

从用户视角看，后者通常更重要，因为用户更关心：

- 答案是不是对的
- 答案是不是能被证据支撑
- 答案是不是接近参考答案

而这三项恰好是 DE 当前明显占优的部分。

## 5. 当前实验的局限性

这组对比目前还不能直接写成“DE 优于 ontology-rag”的最终结论，因为存在以下实验控制问题。

### 5.1 模型不同

- DE 配置使用 `qwen3.5-plus`
- baseline 配置使用 `gpt-4o`

对应配置：

- [configs/config_soybean.yaml:L1-L15](file:///home/liqiankun/data/ograg/configs/config_soybean.yaml#L1-L15)
- [configs/hypergrag/config_soybean.yaml:L1-L29](file:///home/liqiankun/data/ograg/configs/hypergrag/config_soybean.yaml#L1-L29)

### 5.2 embedding 不同

- DE 使用 `text-embedding-v4`
- baseline 使用 `text-embedding-3-small`

embedding 差异会直接影响超图节点相似度排序，因此不能忽略。

### 5.3 检索超参数不同

baseline 显式设置：

- `top_k = 2`
- `nodes_top_k = 5`

而 DE 当前配置中未显式设置超参数，实际走默认值：

- `top_k = 5`
- `nodes_top_k = 20`
- `max_query_debate_rounds = 3`

对应配置与实现：

- [configs/hypergrag/config_soybean.yaml:L39-L44](file:///home/liqiankun/data/ograg/configs/hypergrag/config_soybean.yaml#L39-L44)
- [rag_engine_de_og_rag.py:L223-L256](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L223-L256)

这意味着当前对比同时改变了：

- LLM
- embedding
- 检索深度
- 是否启用 retrieval debate

因此，这是一组“系统级效果对比”，不是“算法单因素消融”。

### 5.4 文件命名存在歧义

当前配置中还有一个容易误判结果来源的问题：

- `configs/config_soybean.yaml` 中 `method` 是 `de-ontohypergraph-rag`
- 但 `answers_file` 却写成了 `ragas_answers_graphrag.json`

同时，`query_llm.py` 会自动在输出文件名后追加 method 后缀。

对应位置：

- [configs/config_soybean.yaml:L27-L40](file:///home/liqiankun/data/ograg/configs/config_soybean.yaml#L27-L40)
- [query_llm.py:L182-L197](file:///home/liqiankun/data/ograg/query_llm.py#L182-L197)

这意味着：

- 文件名里出现 `graphrag`，不一定真的对应 `GraphRAG`
- 结果归因前必须先核对真实 `method`

## 6. 可写入实验报告的总结表述

可以将本次结果总结为：

> 在当前 soybean reasoning 评测中，DEOntoHyperGraphRAG 在 answer correctness、faithfulness 和 answer similarity 上显著优于 ontology-rag，说明 retrieval debate 机制能够通过 query optimization 与 query expansion 补齐回答所需的 ontology-grounded 证据，并增强生成结果对上下文的依赖程度。  
> 与此同时，ontology-rag 在 context precision 和 context recall 上保持更高得分，说明其单次检索结果更加集中且贴近问题，但这些高质量上下文并未被稳定转化为高忠实度答案。  
> 该现象表明，两种方法的主要差异并不只体现在“是否检索到相关事实”，而更体现在“是否能够将结构化事实组织为可支撑最终回答的证据集合”。  
> 需要指出的是，当前实验尚未控制模型、embedding 与检索超参数，因此该结论更适合作为系统级现象分析，而非严格的算法消融结论。

## 7. 后续实验建议

为了把“DE 机制本身的收益”单独量出来，建议补做以下对照实验：

1. 固定同一个 LLM，只比较 `ontohypergraph-rag` 与 `de-ontohypergraph-rag`
2. 固定同一个 embedding 模型
3. 固定 `top_k`、`nodes_top_k`
4. 仅切换 `max_query_debate_rounds = 0 / 1 / 3`
5. 对错误样例做 case study，区分是：
   - 检索错
   - 检索对但生成错
   - 图谱事实冲突导致生成摇摆

如果完成上述控制实验，就可以进一步回答两个更关键的问题：

- DE 的收益到底来自 query debate，还是来自更大的检索池？
- 当前瓶颈到底在检索策略，还是在 ontology 数据质量？

## 8. 问题成因与改进建议

结合 `DEOntoHyperGraphRAGQueryEngine` 的实现细节，可以进一步解释为什么该方法会出现“答案质量提升，但检索指标下降、上下文变杂、结果更依赖知识质量”的现象。

### 8.1 问题成因

#### 8.1.1 多轮 query 结果会被直接合并，缺少后置裁剪

在 `DEOntoHyperGraphRAGQueryEngine.query()` 中，`run_query_stage(query_str)` 返回的是一个 `query_pool`，其中包含初始 query 以及辩论过程中扩展或改写后的 query 检索结果。当前实现会把 `query_pool` 中所有 query 的结果统一合并，只做字符串级去重：

- [rag_engine_de_og_rag.py:L267-L277](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L267-L277)

这意味着：

- 新增 query 带来的证据会直接进入最终上下文
- 不会再做相关性重排序
- 不会再做最小证据子集选择
- 不会主动剔除边际贡献很小的候选边

因此，该方法天然更容易补到多跳事实，但也更容易让上下文变大、变杂。

#### 8.1.2 辩论机制鼓励扩证据，但没有对应的压缩机制

在 Opponent prompt 中，系统明确鼓励：

- `Query Optimization`
- `Query Expansion`

对应实现：

- [rag_engine_de_og_rag.py:L184-L195](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L184-L195)
- [retrieve_debate.py:L131-L152](file:///home/liqiankun/data/ograg/retrieve_debate.py#L131-L152)

这套机制的优势是：当首轮检索缺少关键概念或多跳证据时，可以主动发起补检索。

但问题在于，系统当前只有“如何继续检索”的机制，没有“何时该收缩上下文”的机制。因此，它更像是一个会补证据的系统，而不是一个会主动压缩证据的系统。

#### 8.1.3 停止条件比较粗粒度，缺少“增益是否足够”的判断

当前辩论的停止条件主要有三类：

1. Moderator 判定 Proponent 胜出
2. 达到 `max_query_debate_rounds`
3. Opponent 输出无法解析

对应实现：

- [retrieve_debate.py:L39-L78](file:///home/liqiankun/data/ograg/retrieve_debate.py#L39-L78)

这套停止逻辑没有显式考虑：

- 新一轮 query 是否真的带来了新的关键概念
- 新增证据的边际收益是否已经很低
- 当前上下文是否已经过长
- 是否已经出现明显冲突事实

因此，它在一些样本上可能会停得太晚，把额外噪声一起带入最终生成。

#### 8.1.4 “minimality” 目前只存在于 prompt 中，没有变成算法约束

Moderator prompt 中明确写了：

- ontology grounding
- evidence sufficiency
- minimality

对应实现：

- [rag_engine_de_og_rag.py:L209-L218](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L209-L218)

但从代码执行流程上看，当前并没有一个真正的“最小证据集筛选”步骤。也就是说：

- LLM 被要求判断证据要尽量精简
- 但系统最终还是把所有保留下来的 query 结果一起送入生成

因此，`minimality` 目前更像是一个软提示，而不是硬性的系统约束。

#### 8.1.5 当前 ontology 的重复、冲突和粒度不一致会被 DE 放大

由于 DE 会把多轮检索结果汇聚起来，它比单次 `ontology-rag` 更依赖知识图谱本身的质量。

当 ontology 中存在：

- 实体别名不统一
- 属性拼写不一致
- 同一事实存在多个冲突版本
- 边的粒度不一致

DE 更容易把这些冲突事实一起送入生成阶段。当前实现对此几乎没有专门处理，只做了字符串级去重：

- [rag_engine_de_og_rag.py:L269-L277](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L269-L277)

因此，DE 的收益和风险都会随着知识图谱质量同步放大。

#### 8.1.6 最终回答阶段仍然是宽松生成，而不是证据约束生成

在所有辩论结束之后，最终答案生成仍然只是一次标准的 LLM 调用：

- [rag_engine_de_og_rag.py:L279-L282](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L279-L282)

当前 prompt 并没有强约束模型：

- 每个结论必须绑定哪条证据
- 遇到冲突证据时必须显式说明冲突
- 如果证据不足应拒答而不是补全

因此，只要最终 `contexts` 中混入了较多近似事实或冲突边，生成阶段仍然可能发生“跨边整合”和“超证据发挥”。

### 8.2 改进建议

#### 8.2.1 在辩论之后增加 evidence rerank / prune

当前最值得优先尝试的改进，是在 `query_pool` 合并之后、最终生成之前，再加一层候选证据重排序和裁剪。

可行做法包括：

- 按 query 相似度重新排序
- 按 ontology key 覆盖率排序
- 按是否新增关键概念排序
- 按冲突风险降权
- 限制最终上下文中保留的超边数和 vector 文档数

这样可以在尽量保留 DE 补证据优势的同时，降低 `context_precision` 下滑的幅度。

#### 8.2.2 将停止条件改为“轮数 + 增益”联合判断

除了保留 `max_query_debate_rounds` 外，建议再加入一些增益阈值：

- 新一轮新增 key 数低于阈值时停止
- 新增文档很多但新增有效概念很少时停止
- 上下文长度超过预算时停止
- 新 query 与旧 query 语义高度重复时停止

这样可以避免辩论在低收益阶段继续扩张检索池。

#### 8.2.3 引入最小证据子集选择

如果希望真正落实 prompt 中的 `minimality`，就需要把它从语言提示转成算法约束。

一个较自然的做法是：

1. 先识别当前问题涉及的关键 concept slots
2. 再从候选超边中选出能覆盖这些 slots 的最小证据集合

这样可以显著减少：

- 冗余边
- 相似边重复进入生成
- 上下文过大导致的回答发散

#### 8.2.4 加入 ontology 规范化和冲突消解

如果知识图谱本身存在较多命名不一致、重复边和冲突值，那么 DE 机制会持续放大这些问题。因此，建议在检索之前增加数据清洗与规范化步骤，例如：

- 实体别名归一化
- 属性名归一化
- 拼写纠错
- 重复边合并
- 冲突事实显式标记

这部分改进虽然不只服务 DE，但对 DE 的收益会更明显。

#### 8.2.5 将最终生成改为“证据约束回答”

当前回答阶段仍然太宽松。建议改成更强约束的生成方式，例如要求模型：

- 只能依据给定证据作答
- 每个结论必须可由至少一条证据支撑
- 若证据冲突，必须指出冲突
- 若证据不足，必须明确说明不足

这样可以进一步提高 `faithfulness`，同时降低“上下文一杂，回答就开始整合发挥”的风险。

#### 8.2.6 让 query optimization 更可控

当前 `maintain_query_pool()` 中对 `Query Optimization` 的替换逻辑，本质上是通过 Levenshtein 距离寻找最相近的旧 query：

- [retrieve_debate.py:L133-L145](file:///home/liqiankun/data/ograg/retrieve_debate.py#L133-L145)

这个机制在 query 数量增加时会变得脆弱，因为：

- 容易替错 query
- 无法显式区分“替换哪个旧 query”与“新增一个补充 query”

更稳妥的做法是：

- 给每个 query 分配 ID
- Opponent 在输出时显式指定要替换的 query ID
- 或者至少在维护 query pool 时加入语义相似度和轮次信息

#### 8.2.7 提高实验可观测性

当前实现中，向量检索分支如果异常会被直接吞掉：

- [rag_engine_de_og_rag.py:L142-L156](file:///home/liqiankun/data/ograg/rag_engine_de_og_rag.py#L142-L156)

这会导致实验在不知情的情况下退化成纯 ontology 检索。建议：

- 记录异常日志
- 在输出结果中标记是否使用了 vector evidence
- 在实验报告中区分“启用向量检索”和“实际成功使用向量检索”

否则，后续分析很容易误判真实收益来源。

### 8.3 小结

从系统设计上看，`DEOntoHyperGraphRAG` 当前更像是一个：

- 擅长补证据
- 擅长组织多跳事实
- 但不擅长强约束上下文规模与冲突信息的系统

因此，它自然会呈现出以下特征：

- 最终答案质量更高
- 忠实度更高
- 但检索上下文更容易变大、变杂
- 对 ontology 质量更敏感

换句话说，当前 DE 的主要问题并不是“不会检索”，而是：

> 会扩证据，但缺少足够强的证据压缩、冲突处理和最终回答约束机制。

如果后续改进能围绕“扩证据之后如何收敛证据”展开，那么 DE 很可能在保留 answer quality 优势的同时，进一步改善 context-level 指标表现。
