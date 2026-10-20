import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from llama_index.core.retrievers import VectorIndexRetriever
from tqdm import tqdm

from retrieve_debate import RetrievalDebate
from utils import (
    create_or_load_index,
    create_service_context,
    get_config,
    get_documents,
    load_llm_and_embeds,
)
from query_engine.ontograph_query_engine import OntoHyperGraphQueryEngine

METHOD_NAME = "de-ontohypergraph-rag"
MAX_TOKENS = 1024

RAG_QUERY_PROMPT = """Given the context below, answer the following question. 
Note that the context is provided as a list of valid facts in a dictionary format and an optional set of rules.

Context: {context}

Question: {query_str}

Answer:
"""


class QnA_IO:
    def __init__(self):
        self.data = defaultdict(lambda: [])

    def read(self, file_paths: Union[str, List[str]]):
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            with open(file_path, "r") as f:
                if ".json" in file_path:
                    data = json.load(f)
                elif ".csv" in file_path:
                    data = pd.read_csv(f).to_dict("records")
                else:
                    raise NotImplementedError(f"Input format for {file_path} is not implemented yet.")
                for datum in data:
                    for key in datum:
                        if key != "metadata":
                            self.data[key].append(datum[key])

    def write(self, file_path, **kwargs):
        for key in kwargs:
            self.data[key] = kwargs[key]
        if ".json" in file_path:
            new_data = [dict(zip(self.data, t)) for t in zip(*self.data.values())]
            with open(file_path, "w") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
        elif ".csv" in file_path:
            new_data = pd.DataFrame(self.data)
            new_data.to_csv(file_path, index=False)
        else:
            raise NotImplementedError(f"Output format for {file_path} is not implemented yet.")


class SimpleChatPromptTemplate:
    def get_string(self, *, messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if role == "system":
                parts.append(f"System:\n{content}")
            elif role == "user":
                parts.append(f"User:\n{content}")
            elif role == "assistant":
                parts.append(f"Assistant:\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts).strip()


class LangChainGeneratorAdapter:
    def __init__(self, llm):
        self._llm = llm

    def generate(self, prompt: str) -> List[str]:
        out = self._llm.invoke(prompt, max_tokens=MAX_TOKENS)
        content = getattr(out, "content", None)
        if content is None:
            return [str(out)]
        return [content]


class OntoHyperGraphRetrieverAdapter:
    def __init__(
        self,
        *,
        onto_engine: OntoHyperGraphQueryEngine,
        top_k: int = 5,
        nodes_top_k: int = 20,
    ):
        self._onto_engine = onto_engine
        self._top_k = top_k
        self._nodes_top_k = nodes_top_k

    def search(self, query: str) -> List[Dict[str, Any]]:
        _, relevant_context = self._onto_engine.retrieve_context(
            query,
            top_k=self._top_k,
            nodes_top_k=self._nodes_top_k,
        )

        docs: List[Dict[str, Any]] = []

        if isinstance(relevant_context, list):
            for i, item in enumerate(relevant_context):
                if isinstance(item, dict):
                    body = json.dumps(item, ensure_ascii=False)
                    keys = sorted(list(item.keys()))
                else:
                    body = str(item)
                    keys = []
                docs.append(
                    {
                        "contents": f"OntoHyperEdge {i+1}\n{body}",
                        "metadata": {"type": "ontohyperedge", "keys": keys},
                    }
                )
        else:
            docs.append(
                {
                    "contents": f"OntoHyperGraphContext\n{str(relevant_context)}",
                    "metadata": {"type": "ontohypergraph"},
                }
            )

        vector_retriever = getattr(self._onto_engine, "_vector_retriever", None)
        if vector_retriever is not None:
            try:
                nodes = vector_retriever.retrieve(query)
                for i, n in enumerate(nodes):
                    text = getattr(getattr(n, "node", None), "text", None)
                    if text:
                        docs.append(
                            {
                                "contents": f"VectorDoc {i+1}\n{text}",
                                "metadata": {"type": "vector"},
                            }
                        )
            except Exception:
                pass

        return docs


class OntologyGroundedRetrievalDebate(RetrievalDebate):
    def __init__(self, *args, min_new_keys_per_round: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_new_keys_per_round = min_new_keys_per_round

    def _ontology_signature(self, query_pool: Dict[str, List[Dict[str, Any]]]) -> Tuple[Set[str], int]:
        keys: Set[str] = set()
        doc_count = 0
        for _, docs in query_pool.items():
            doc_count += len(docs)
            for d in docs:
                meta = d.get("metadata") or {}
                for k in meta.get("keys") or []:
                    keys.add(str(k))
        return keys, doc_count

    def _query_stage_system_message(self, agent_name: str):
        if "Proponent" in agent_name:
            return {
                "role": "system",
                "content": (
                    "You are a debater. Argue that the current retrieved content is sufficient to answer the question.\n"
                    "Your argument must emphasize ontology grounding quality: concept coverage, concept-type consistency, "
                    "and that the evidence set is already minimal and non-redundant.\n"
                    "Do not suggest further retrieval. Output only the argument text. No extra explanations."
                ),
            }
        if "Opponent" in agent_name:
            return {
                "role": "system",
                "content": (
                    "You are a critical thinker and debater. Challenge the sufficiency of the current retrieved content.\n"
                    "You must focus on ontology grounding failures: concept mismatch, missing key concepts, noisy edges, "
                    "or insufficient multi-hop evidence.\n"
                    "Then choose exactly one action:\n"
                    "1) Query Optimization: Query Optimization: [Original Query] -> [New Query].\n"
                    "2) Query Expansion: Query Expansion: [New Query].\n"
                    "Output format: first a brief argument, then the action line in the exact format. No extra explanation."
                ),
            }
        raise ValueError("agent_name must contain Proponent or Opponent")

    def _query_stage_moderator_message(self, agents_messages, input_query: str, query_pool: dict):
        keys, doc_count = self._ontology_signature(query_pool)

        agents_arguments = ""
        for agent in agents_messages:
            agents_arguments += f"{agent}: {agents_messages[agent][1]}\n"

        return {
            "role": "system",
            "content": (
                "You are the judge in a debate. Decide whether the current retrieval is sufficient.\n"
                "Decision criteria (in order):\n"
                "1) Ontology grounding: concept coverage and concept consistency.\n"
                "2) Evidence sufficiency for multi-hop reasoning.\n"
                "3) Minimality: avoid redundant/noisy content.\n"
                f"Question: {input_query}\n"
                f"Ontology signature: unique_keys={sorted(list(keys))}, doc_count={doc_count}\n"
                f"{self.format_query_pool(query_pool)}\n"
                f"Agents Arguments:\n{agents_arguments}\n"
                "Output only the agent's name. No explanations."
            ),
        }

    def should_stop_after_round(self, previous_query_pool: dict, current_query_pool: dict) -> bool:
        if self.min_new_keys_per_round <= 0:
            return False

        previous_keys, _ = self._ontology_signature(previous_query_pool)
        current_keys, _ = self._ontology_signature(current_query_pool)
        new_keys_count = len(current_keys - previous_keys)
        return new_keys_count < self.min_new_keys_per_round


class DEOntoHyperGraphRAGQueryEngine:
    def __init__(
        self,
        *,
        llm,
        embed_model,
        ontology_nodes_path: str,
        vector_retriever=None,
        max_query_debate_rounds: int = 3,
        nodes_top_k: int = 20,
        top_k: int = 5,
        min_new_keys_per_round: int = 0,
        max_query_similarity: float = None,
    ):
        onto_engine = OntoHyperGraphQueryEngine.from_ontology_path(
            ontology_nodes_path=ontology_nodes_path,
            llm=llm,
            embed_model=embed_model,
            vector_retriever=vector_retriever,
        )

        self._llm = llm
        self._onto_engine = onto_engine
        self._retriever = OntoHyperGraphRetrieverAdapter(
            onto_engine=onto_engine,
            top_k=top_k,
            nodes_top_k=nodes_top_k,
        )
        self._debate = OntologyGroundedRetrievalDebate(
            retriever=self._retriever,
            generator=LangChainGeneratorAdapter(llm),
            prompt_template=SimpleChatPromptTemplate(),
            max_query_debate_rounds=max_query_debate_rounds,
            query_proponent_agent=1,
            query_opponent_agent=1,
            min_new_keys_per_round=min_new_keys_per_round,
            max_query_similarity=max_query_similarity,
            query_embedder=embed_model,
        )

    def query(
        self,
        *,
        query_str: str,
        return_context: bool = False,
        rules: Optional[List[str]] = None,
        **kwargs,
    ):
        rules = rules or []
        query_pool = self._debate.run_query_stage(query_str)

        contexts: List[str] = []
        seen: Set[str] = set()
        
        for q in query_pool:
            for d in query_pool[q]:
                c = str(d.get("contents", "")).strip()
                if c and c not in seen:
                    seen.add(c)
                    contexts.append(c)

        response = self._llm.invoke(
            RAG_QUERY_PROMPT.format(context=contexts + rules, query_str=query_str),
            max_tokens=MAX_TOKENS,
        )

        if return_context:
            return response, contexts
        return response


def _load_rules(config) -> List[str]:
    rules: List[str] = []
    if "rules_file" in config.data and config.data.rules_file:
        with open(config.data.rules_file, "r") as f:
            rules += [line.rstrip("\n") for line in f.readlines()]
    return rules


if __name__ == "__main__":
    config = get_config()
    if config.query.method != METHOD_NAME:
        raise ValueError(
            f"This runner only supports config.query.method == '{METHOD_NAME}', got '{config.query.method}'."
        )

    service_context = create_service_context(config.model, config.embedding_model)
    llm, embeddings = load_llm_and_embeds(config.model, config.embedding_model)

    # documents = get_documents(
    #     config.data.documents_dir,
    #     subdir=config.data.subdir,
    #     smart_pdf=config.data.smart_pdf,
    #     full_text=config.data.full_text,
    # )
    rules = _load_rules(config)

    vector_retriever = None
    if "hyperparams" in config.query and getattr(config.query.hyperparams, "vector_index", False):
        documents = get_documents(
            config.data.documents_dir,
            subdir=config.data.subdir,
            smart_pdf=config.data.smart_pdf,
            full_text=config.data.full_text,
        )
        vector_index = create_or_load_index(
            index_directory=config.data.index_dir,
            service_context=service_context,
            documents=documents,
        )
        vector_retriever = VectorIndexRetriever(index=vector_index)

    hp = config.query.hyperparams if "hyperparams" in config.query else {}
    max_rounds = int(getattr(hp, "max_query_debate_rounds", 3))
    nodes_top_k = int(getattr(hp, "nodes_top_k", 20))
    top_k = int(getattr(hp, "top_k", 5))
    min_new_keys_per_round = int(getattr(hp, "min_new_keys_per_round", 0))
    max_query_similarity = getattr(hp, "max_query_similarity", None)
    if max_query_similarity is not None:
        max_query_similarity = float(max_query_similarity)

    query_engine = DEOntoHyperGraphRAGQueryEngine(
        llm=llm,
        embed_model=embeddings,
        ontology_nodes_path=f"{config.data.kg_storage_path}",
        vector_retriever=vector_retriever,
        max_query_debate_rounds=max_rounds,
        nodes_top_k=nodes_top_k,
        top_k=top_k,
        min_new_keys_per_round=min_new_keys_per_round,
        max_query_similarity=max_query_similarity,
    )

    if len(config.query.questions_file) == 0:
        while True:
            query = input("Type your query (press Enter to stop):")
            if query == "":
                break
            response = query_engine.query(query_str=query, return_context=False, rules=rules)
            print(f"Response: {response}")
    else:
        answer_dir = os.path.dirname(config.query.answers_file) or "."
        os.makedirs(answer_dir, exist_ok=True)

        answer_file = os.path.basename(config.query.answers_file)
        if "." in answer_file:
            answer_fname, answer_ftype = answer_file.rsplit(".", 1)
        else:
            answer_fname, answer_ftype = answer_file, "json"

        method_suffix = str(config.query.method).replace("-", "_")
        if method_suffix and f"_{method_suffix}" not in answer_fname:
            answer_fname = f"{answer_fname}_{method_suffix}"

        if "hyperparams" in config.query and config.query.hyperparams:
            hyperparams_suffix = "_".join([f"{k}{v}" for k, v in config.query.hyperparams.items()])
            if hyperparams_suffix:
                answer_fname = f"{answer_fname}_{hyperparams_suffix}"

        config.query.answers_file = f"{answer_dir}/{answer_fname}.{answer_ftype}"

        if os.path.exists(config.query.answers_file) and not config.rewrite:
            raise RuntimeError(f"Answers file {config.query.answers_file} already exists.")

        evaluator = QnA_IO()
        evaluator.read(config.query.questions_file)
        questions = evaluator.data["question"]

        responses = []
        times = []
        retrieved_contexts = []

        # 逐个问题执行查询，并记录每条样本的耗时与检索上下文。
        for question in tqdm(questions):
            start_time = time.time()
            response, retrieved_context = query_engine.query(
                query_str=question, return_context=True, rules=rules
            )

            # 不同底层 LLM/QueryEngine 返回对象的字段不完全一致，
            # 这里按常见返回格式依次兼容提取最终答案文本。
            try:
                responses.append(response.response)
            except Exception:
                try:
                    responses.append(response.content)
                except Exception:
                    responses.append(response)

            # 保存该问题的总耗时，以及生成答案时使用到的上下文。
            times.append(time.time() - start_time)
            retrieved_contexts.append(retrieved_context)

        # 将答案、检索上下文和耗时统一写入结果文件，供后续评测使用。
        evaluator.write(
            config.query.answers_file,
            answer=responses,
            retrieved_context=retrieved_contexts,
            time=times,
        )
