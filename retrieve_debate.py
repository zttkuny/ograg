import Levenshtein
import numpy as np


class RetrievalDebate:
    def __init__(
        self,
        *,
        retriever,
        generator,
        prompt_template,
        max_query_debate_rounds: int = 3,
        query_proponent_agent: int = 1,
        query_opponent_agent: int = 1,
        max_query_similarity: float = None,
        query_embedder=None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.prompt_template = prompt_template
        self.max_query_debate_rounds = max_query_debate_rounds
        self.max_query_similarity = max_query_similarity
        self.query_embedder = query_embedder
        self.last_run_trace = {}

        self.agents_messages_query_stage = {}
        for i in range(query_proponent_agent):
            self.agents_messages_query_stage[f"Proponent Agent {i}"] = []
        for i in range(query_opponent_agent):
            self.agents_messages_query_stage[f"Opponent Agent {i}"] = []

    def _doc_keys(self, docs):
        keys = set()
        for doc in docs:
            metadata = doc.get("metadata") if isinstance(doc, dict) else None
            for key in (metadata or {}).get("keys") or []:
                keys.add(str(key))
        return keys

    def _pool_snapshot(self, query_pool):
        keys = set()
        docs_seen = set()
        doc_count = 0
        for docs in query_pool.values():
            doc_count += len(docs)
            keys.update(self._doc_keys(docs))
            docs_seen.update(str(doc.get("contents", "")) for doc in docs)
        return {
            "queries": list(query_pool.keys()),
            "doc_count": doc_count,
            "docs": docs_seen,
            "keys": sorted(keys),
        }

    def _initial_event(self, query, query_pool):
        snapshot = self._pool_snapshot(query_pool)
        return {
            "round": 0,
            "action": "initial",
            "query": query,
            "new_doc_count": len(snapshot["docs"]),
            "removed_doc_count": 0,
            "doc_delta": snapshot["doc_count"],
            "new_keys": snapshot["keys"],
            "doc_count": snapshot["doc_count"],
            "key_count": len(snapshot["keys"]),
        }

    def _query_event(self, round_id, action, query, before, after):
        before_keys = set(before["keys"])
        after_keys = set(after["keys"])
        before_docs = set(before["docs"])
        after_docs = set(after["docs"])
        return {
            "round": round_id,
            "action": action,
            "query": query,
            "new_doc_count": len(after_docs - before_docs),
            "removed_doc_count": len(before_docs - after_docs),
            "doc_delta": after["doc_count"] - before["doc_count"],
            "new_keys": sorted(after_keys - before_keys),
            "doc_count": after["doc_count"],
            "key_count": len(after["keys"]),
        }

    def _finish(self, query_pool, rounds, stop_reason, events):
        snapshot = self._pool_snapshot(query_pool)
        self.last_run_trace = {
            "rounds": rounds,
            "stop_reason": stop_reason,
            "events": events,
            "final_queries": snapshot["queries"],
            "final_doc_count": snapshot["doc_count"],
            "final_keys": snapshot["keys"],
        }
        return query_pool

    def run_query_stage(
        self,
        question: str,
        *,
        record=None,
        initial_query_pool=None,
        max_rounds=None,
        zero_round_reason: str = "max_rounds_zero",
    ):
        query_pool = {}
        self.last_run_trace = {}

        input_query = question
        if initial_query_pool is None:
            retrieval_results = self.retriever.search(input_query)
            query_pool[input_query.strip()] = retrieval_results
        else:
            query_pool = {
                query: list(docs) for query, docs in initial_query_pool.items()
            }
            if not query_pool:
                query_pool[input_query.strip()] = []

        events = [self._initial_event(input_query.strip(), query_pool)]
        rounds_limit = (
            self.max_query_debate_rounds if max_rounds is None else max(int(max_rounds), 0)
        )

        if rounds_limit == 0:
            return self._finish(query_pool, 0, zero_round_reason, events)

        for round_id in range(rounds_limit):
            agents_messages = {}

            for agent_name in self.agents_messages_query_stage:
                round_message = [
                    self._query_stage_system_message(agent_name),
                    {
                        "role": "user",
                        "content": f"Question: {question}\n{self.format_query_pool(query_pool)}",
                    },
                ]
                input_prompt = self.prompt_template.get_string(messages=round_message)
                output = self.generator.generate(input_prompt)[0]

                if record is not None:
                    record[f"QueryStage_{agent_name}_Round{round_id}_InputPrompt"] = input_prompt
                    record[f"QueryStage_{agent_name}_Round{round_id}_Output"] = output

                agents_messages[agent_name] = [input_prompt, output]

            moderator_message = [
                self._query_stage_moderator_message(agents_messages, input_query, query_pool)
            ]
            moderator_input_prompt = self.prompt_template.get_string(messages=moderator_message)
            moderator_output = self.generator.generate(moderator_input_prompt)[0]

            if record is not None:
                record[f"QueryStage_Moderator_Round{round_id}_InputPrompt"] = moderator_input_prompt
                record[f"QueryStage_Moderator_Round{round_id}_Output"] = moderator_output

            if "Proponent" in moderator_output:
                return self._finish(query_pool, round_id + 1, "proponent_win", events)

            opponent_output = agents_messages["Opponent Agent 0"][1]
            action = self.parse_opponent_action(opponent_output)
            if action is None:
                return self._finish(query_pool, round_id + 1, "unparsable_action", events)
            if self.should_stop_for_similar_query(query_pool, action["new_query"]):
                snapshot = self._pool_snapshot(query_pool)
                events.append(
                    {
                        "round": round_id + 1,
                        "action": "similar_query_rejected",
                        "query": action["new_query"],
                        "new_doc_count": 0,
                        "removed_doc_count": 0,
                        "doc_delta": 0,
                        "new_keys": [],
                        "doc_count": snapshot["doc_count"],
                        "key_count": len(snapshot["keys"]),
                    }
                )
                return self._finish(query_pool, round_id + 1, "similar_query", events)

            before = self._pool_snapshot(query_pool)
            query_pool_tmp = self.maintain_query_pool(query_pool, action)
            if query_pool_tmp is None:
                return self._finish(
                    query_pool, round_id + 1, "query_pool_update_failed", events
                )
            after = self._pool_snapshot(query_pool_tmp)
            events.append(
                self._query_event(
                    round_id + 1, action["type"], action["new_query"], before, after
                )
            )
            if self.should_stop_after_round(query_pool, query_pool_tmp):
                return self._finish(query_pool_tmp, round_id + 1, "min_new_keys", events)
            query_pool = query_pool_tmp

        return self._finish(query_pool, rounds_limit, "max_rounds", events)

    def _query_stage_system_message(self, agent_name: str):
        if "Proponent" in agent_name:
            return {
                "role": "system",
                "content": (
                    "You are a debater. Argue that the current retrieved content is sufficient "
                    "to answer the question and no further retrieval is needed. Deliver a brief, "
                    "strong argument with clear reasoning. Do not suggest further retrieval. "
                    "No extra explanations."
                ),
            }
        if "Opponent" in agent_name:
            return {
                "role": "system",
                "content": (
                    "You are a critical thinker and debater, and your task is to challenge the "
                    "sufficiency of the current retrieved content. Argue that the current information "
                    "is insufficient to generate a reliable answer and propose either query optimization "
                    "or query expansion.\n"
                    "The action you can choose:\n"
                    "1. Query Optimization: If the retrieved content is somewhat relevant but has expression "
                    "or scope issues, improve the query using this format: "
                    "Query Optimization: [Original Query] -> [New Query].\n"
                    "2. Query Expansion: If critical information is missing, propose a new query using this "
                    "format: Query Expansion: [New Query].\n"
                    "Deliver a brief, strong argument with clear reasoning, then you must choose only one action. "
                    "The output must be in the exact format after your reasoning, without additional explanation, "
                    "and keep the new query short and precise."
                ),
            }
        raise ValueError("agent_name must contain Proponent or Opponent")

    def _query_stage_moderator_message(self, agents_messages, input_query: str, query_pool: dict):
        agents_arguments = ""
        for agent in agents_messages:
            agents_arguments += f"{agent}: {agents_messages[agent][1]}\n"

        return {
            "role": "system",
            "content": (
                "You are the judge in a debate. Your task is to evaluate the arguments from agents. "
                "There are two types of agents:\n"
                "1. Proponent Agent: Argues that the current retrieved content is sufficient.\n"
                "2. Opponent Agent: Argues that the current retrieved content is insufficient and proposes query refinement.\n"
                f"Question: {input_query}\n"
                f"{self.format_query_pool(query_pool)}\n"
                f"Agents Arguments:\n{agents_arguments}\n"
                "Output only the agent's name. No explanations."
            ),
        }

    def parse_opponent_action(self, opponent_output: str):
        if "Query Optimization:" in opponent_output:
            optimization_instruction = opponent_output.split("Query Optimization:")[1].strip()
            if "->" in optimization_instruction:
                lhs, rhs = optimization_instruction.split("->", 1)
                original_query = lhs.strip()
                new_query = rhs.strip()
            else:
                original_query = optimization_instruction
                new_query = optimization_instruction
            if not new_query:
                return None
            return {
                "type": "optimization",
                "original_query": original_query,
                "new_query": new_query,
            }

        if "Query Expansion:" in opponent_output:
            new_query = opponent_output.split("Query Expansion:")[1].strip()
            if not new_query:
                return None
            return {
                "type": "expansion",
                "original_query": None,
                "new_query": new_query,
            }

        return None

    def maintain_query_pool(self, query_pool: dict, action: dict):
        try:
            updated_query_pool = {query: list(results) for query, results in query_pool.items()}

            if action["type"] == "optimization":
                original_query = action["original_query"]
                new_query = action["new_query"]

                updated_query_pool.pop(self.find_most_similar_key(updated_query_pool, original_query))
                retrieval_results = self.retriever.search(new_query)
                updated_query_pool[new_query] = retrieval_results
                return updated_query_pool

            if action["type"] == "expansion":
                new_query = action["new_query"]
                retrieval_results = self.retriever.search(new_query)
                updated_query_pool[new_query] = retrieval_results
                return updated_query_pool

            return None
        except Exception as e:
            raise ValueError("Error in maintaining the query pool") from e

    def should_stop_for_similar_query(self, query_pool: dict, new_query: str) -> bool:
        if self.max_query_similarity is None or self.query_embedder is None or not query_pool:
            return False

        new_query_embedding = np.array(self.query_embedder.embed_query(new_query))
        for existing_query in query_pool.keys():
            existing_query_embedding = np.array(self.query_embedder.embed_query(existing_query))
            similarity = self.cosine_similarity(existing_query_embedding, new_query_embedding)
            if similarity >= self.max_query_similarity:
                return True
        return False

    def should_stop_after_round(self, previous_query_pool: dict, current_query_pool: dict) -> bool:
        return False

    def cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        if x_norm == 0 or y_norm == 0:
            return 0.0
        return float(np.dot(x, y) / (x_norm * y_norm))

    def format_query_pool(self, query_pool: dict):
        query_pool_str = ""
        for i, query in enumerate(query_pool):
            query_pool_str += (
                f"Query {i+1}: {query}\n"
                f"Retrieved Content: {self._format_reference(query_pool[query])}"
            )
        return query_pool_str

    def _format_reference(self, retrieval_result):
        s = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            s += f"Doc {idx+1}(Title: {title}) {text}\n"
        return s

    def find_most_similar_key(self, query_dict: dict, target_query: str):
        min_distance = float("inf")
        most_similar_key = None
        for key in query_dict.keys():
            distance = Levenshtein.distance(key, target_query)
            if distance < min_distance:
                min_distance = distance
                most_similar_key = key
        return most_similar_key
