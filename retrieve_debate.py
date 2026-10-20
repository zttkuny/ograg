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

        self.agents_messages_query_stage = {}
        for i in range(query_proponent_agent):
            self.agents_messages_query_stage[f"Proponent Agent {i}"] = []
        for i in range(query_opponent_agent):
            self.agents_messages_query_stage[f"Opponent Agent {i}"] = []

    def run_query_stage(self, question: str, *, record=None):
        query_pool = {}

        if self.max_query_debate_rounds == 0:
            input_query = question
            retrieval_results = self.retriever.search(input_query)
            query_pool[input_query.strip()] = retrieval_results
            return query_pool

        input_query = question
        retrieval_results = self.retriever.search(input_query)
        query_pool[input_query.strip()] = retrieval_results

        for round_id in range(self.max_query_debate_rounds):
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
                return query_pool

            opponent_output = agents_messages["Opponent Agent 0"][1]
            action = self.parse_opponent_action(opponent_output)
            if action is None:
                return query_pool
            if self.should_stop_for_similar_query(query_pool, action["new_query"]):
                return query_pool

            query_pool_tmp = self.maintain_query_pool(query_pool, action)
            if query_pool_tmp is None:
                return query_pool
            if self.should_stop_after_round(query_pool, query_pool_tmp):
                return query_pool
            query_pool = query_pool_tmp

        return query_pool

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
