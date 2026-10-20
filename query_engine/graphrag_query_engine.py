from pathlib import Path

try:
    from graphrag.cli.query import run_local_search, run_global_search
except ImportError:
    run_local_search = None
    run_global_search = None
from llama_index.core.retrievers import VectorIndexRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore
from typing import Any, List, Dict

QUERY_PROMPT = """Given the context below, answer the following question.
---------------------
Context:\n {context}\n\n
---------------------
Question: {query_str}
---------------------
Answer: """

MAX_TOKENS = 4096

class GraphRAGQueryEngine:
    def __init__(
        self,
        llm: BaseLanguageModel,
        data_config: Dict[str, Any],
        method: str = "local",
        community_level: int = 2,
        response_type: str = "Multiple Paragraphs",
        max_context_rows: int = 20,
        **kwargs: Any,
    ):
        self.data_config = data_config
        self.method = method
        self.community_level = community_level
        self.response_type = response_type
        self.llm = llm
        self.max_context_rows = max_context_rows

    def _context_records_to_text(self, context_records: Any) -> str:
        if context_records is None:
            return ""
        if isinstance(context_records, str):
            return context_records
        if isinstance(context_records, dict):
            parts: list[str] = []
            for name, value in context_records.items():
                if hasattr(value, "head") and hasattr(value, "to_csv"):
                    try:
                        table_txt = value.head(self.max_context_rows).to_csv(index=False)
                    except Exception:
                        table_txt = str(value)
                else:
                    table_txt = str(value)
                parts.append(f"-----{name}-----\n{table_txt}")
            return "\n\n".join(parts)
        return str(context_records)

    def _retrieve_nodes(self, query_str: str, return_context: bool = False, **kwargs):
        if run_local_search is None or run_global_search is None:
            raise ModuleNotFoundError(
                "graphrag CLI query module is unavailable. Install/upgrade graphrag so that graphrag.cli.query exists."
            )

        documents_dir = Path(str(self.data_config["documents_dir"]))
        documents_dir = documents_dir.expanduser()
        if documents_dir.is_absolute():
            root_dir = documents_dir
        else:
            repo_root = Path(__file__).resolve().parents[1]
            root_dir = (repo_root / documents_dir).resolve()

        artifacts_dir = root_dir / "output" / "final" / "artifacts"
        data_dir = artifacts_dir if artifacts_dir.exists() else None

        if self.method == "local":
            response_txt, context_data = run_local_search(
                data_dir=data_dir,
                root_dir=root_dir,
                community_level=self.community_level,
                response_type=self.response_type,
                streaming=False,
                query=query_str,
                verbose=False,
            )
        elif self.method == "global":
            response_txt, context_data = run_global_search(
                data_dir=data_dir,
                root_dir=root_dir,
                community_level=self.community_level,
                dynamic_community_selection=False,
                response_type=self.response_type,
                streaming=False,
                query=query_str,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown GraphRAG method: {self.method}")

        return str(response_txt), self._context_records_to_text(context_data)
    
    def query(self, query_str: str, return_context=False, rules=[], **kwargs):
        response_txt, retrieved_nodes_text = self._retrieve_nodes(query_str)
        response = Response(response_txt)
        if return_context:
            return response, retrieved_nodes_text
        return response
    

