# Hypergraph Retrieval Notes

## 背景

项目中的 `DEOntoHyperGraphRAGQueryEngine` 位于 `rag_engine_de_og_rag.py`，底层仍然依赖
`query_engine/ontograph_query_engine.py` 里的 `OntoHyperGraphQueryEngine`。

它和普通向量 `RAG` 的主要区别不在最终生成，而在检索阶段：

- 使用本体结构化事实，而不是纯文本 chunk
- 以超边为检索目标，而不是单条文本片段
- 在 `DE` 版本中加入了 retrieval debate

## 什么是这里的超图检索

这里的超图检索，不是把知识拆成独立段落做相似度搜索，而是把一条事实表示成由多个属性共同组成的超边。

例如：

```python
{
  "crop": "soybean",
  "growing_zone": "North Eastern Hill zone",
  "seeding_rate": "55 kg per hectare",
  "sowing_time_start": "15th June",
  "sowing_time_end": "30th June"
}
```

在项目里，这类结构会建模成：

- `HyperNode`: 单个属性键值对
- `HyperEdge`: 一组共同组成事实的节点
- `OntoHyperGraph`: 由多个超边组成的超图

## 超图检索的特点

### 1. 检索单位是事实组合

普通向量 RAG 通常检索 chunk；这里检索的核心单位是一条超边，也就是一组相关属性组成的组合事实。

直接效果是：

- 返回上下文更完整
- 更适合多字段联合问答
- 不容易把关联信息拆散

### 2. 同时考虑 key 和 value 的相似度

`HyperNode` 会分别给属性名和属性值做 embedding。

因此查询不仅能匹配“这段文本像不像”，还可以分别匹配：

- 用户问的是什么属性
- 用户问的是什么属性值

这比普通文本向量匹配更适合结构化知识。

### 3. 选边时强调覆盖更多相关节点

流程不是直接返回 top-k 节点，而是先找相关节点，再找能覆盖这些节点的超边。

因此它更倾向于返回：

- 覆盖更高的事实
- 结构更完整的组合
- 冗余更少的上下文

### 4. 更适合 ontology 驱动的数据

这种方法依赖知识已经被整理成结构化、本体化的 `key-value` 事实，因此更适合：

- 农业知识
- 规则知识
- 本体知识
- 强字段关系的数据

### 5. 更适合多属性约束问题

普通图检索常见的是二元关系：

```text
subject - relation - object
```

而超图中的一条超边可以同时包含多个属性，因此更适合回答：

- 某地区某作物的播种时间是什么
- 某品种在什么区域适合种植
- 某个场景下施肥量、播种量和时间窗口分别是什么

## 它和普通 RAG 的差别

项目里的 `RAGQueryEngine` 流程很直接：

1. 向量召回文本
2. 拼接上下文
3. 一次性调用 LLM 生成答案

而 `OntoHyperGraphQueryEngine` 的差别在于：

1. 检索对象从文本 chunk 变成超边事实
2. 匹配依据包含 key/value 的结构化相似度
3. 返回的是更完整的组合事实

一句话概括：

> 普通 RAG 更像从文档里找相似段落，超图检索更像从结构化知识里找完整事实组合。

## `DEOntoHyperGraphRAG` 多了什么

`DEOntoHyperGraphRAGQueryEngine` 不是重写了超图结构，而是在超图检索前后加了一层 debate 机制。

它的流程是：

1. 先做一次初始超图检索
2. 让 Proponent 和 Opponent 围绕当前检索结果辩论
3. Opponent 可以提出：
   - `Query Optimization`
   - `Query Expansion`
4. Moderator 判断证据是否足够
5. 如果不够，就继续更新 query pool 并再次检索
6. 最终把多轮检索结果合并、去重，再交给 LLM 生成答案

所以它本质上是：

> `OntoHyperGraphRAG` + ontology-aware retrieval debate

## 为什么 `DE` 版本更强调 ontology grounding

在 `rag_engine_de_og_rag.py` 中，辩论阶段会特别关注：

- concept coverage
- concept-type consistency
- evidence sufficiency for multi-hop reasoning
- minimality

也就是说，它关心的不只是“信息够不够”，还关心：

- 本体概念覆盖够不够
- 概念类型是否一致
- 是否缺少关键属性
- 证据是否足够且不过多

## 总结

这个项目里的超图检索，优势不是“召回更多文本”，而是：

- 召回更完整的结构化事实
- 保留多属性之间的关联
- 更适合 ontology-grounded 的推理问题

三者可以简单理解为：

- `RAGQueryEngine`: 文本片段检索 + 生成
- `OntoHyperGraphQueryEngine`: 超图事实检索 + 生成
- `DEOntoHyperGraphRAGQueryEngine`: 超图事实检索 + 检索辩论 + 生成
