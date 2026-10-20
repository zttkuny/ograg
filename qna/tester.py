import json
import pandas as pd
import os
from datasets import Dataset
from ragas.metrics import (
        answer_correctness,
        faithfulness,
        answer_similarity,
        context_precision,
        context_recall,
        answer_relevancy
    )
from ragas import evaluate
from ragas.run_config import RunConfig

from utils.token_usage import EvalTokenUsageCallback

run_config = RunConfig(
    max_workers=1,
    timeout=600,
)

class RAGASTester:
    def __init__(self, llm, embeddings, metrics):
        self.llm = llm
        self.embeddings = embeddings
        self.metrics_names = metrics
        self.metrics = []
        self.last_eval_token_usage = None
        self._load_metrics()

    def _load_metrics(self):
        metric_map = {
            'answer_correctness': answer_correctness,
            'faithfulness': faithfulness,
            'answer_similarity': answer_similarity,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
        }
        for m in self.metrics_names:
            if m in metric_map:
                if metric_map[m] is not None:
                    metric = metric_map[m]
                    if m == 'answer_relevancy' and 'deepseek' in str(getattr(self.llm, 'model_name', '')).lower():
                        metric.strictness = 1
                        print("DeepSeek does not support n>1; set answer_relevancy.strictness=1.")
                    self.metrics.append(metric)
            else:
                print(f"Warning: Metric {m} not found in ragas map.")

    def evaluate_answers_from_json(self, file_path, to_file=True):
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        ragas_data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for item in data:
            ragas_data['question'].append(item.get('question', ''))
            ragas_data['answer'].append(item.get('answer', ''))
            
            contexts = []
            # Priority to retrieved_context (list of dicts/strings)
            if 'retrieved_context' in item and isinstance(item['retrieved_context'], list):
                # Convert dicts to strings if necessary
                ctx_list = []
                for ctx in item['retrieved_context']:
                    if isinstance(ctx, dict):
                        ctx_list.append(json.dumps(ctx))
                    else:
                        ctx_list.append(str(ctx))
                contexts = ctx_list
            elif 'contexts' in item:
                 # Fallback to 'contexts' field logic
                 val = item['contexts']
                 if isinstance(val, list):
                     contexts = [str(v) for v in val]
                 elif isinstance(val, str):
                     try:
                         # Handle stringified lists like "['...']"
                         if val.strip().startswith('[') and val.strip().endswith(']'):
                             parsed = eval(val)
                             if isinstance(parsed, list):
                                 contexts = [str(p) for p in parsed]
                             else:
                                 contexts = [val]
                         else:
                             contexts = [val]
                     except:
                         contexts = [val]
            
            ragas_data['contexts'].append(contexts)
            ragas_data['ground_truth'].append(item.get('ground_truth', ''))

        dataset = Dataset.from_dict(ragas_data)
        usage_cb = EvalTokenUsageCallback()
        
        print("Starting Ragas evaluation...")
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            callbacks=[usage_cb],
            raise_exceptions=False,
            run_config=run_config,
            # Limit concurrency to avoid timeouts with proxy
            # max_workers=4 
        )

        token_usage_summary = usage_cb.snapshot()
        token_usage_summary.update(
            {
                "scope": "eval",
                "answer_file": file_path,
                "metrics": self.metrics_names,
                "questions": len(data),
            }
        )
        self.last_eval_token_usage = token_usage_summary
        print(
            "Eval token usage: "
            f"calls={token_usage_summary['calls']}, "
            f"prompt={token_usage_summary['prompt_tokens']}, "
            f"completion={token_usage_summary['completion_tokens']}, "
            f"total={token_usage_summary['total_tokens']}, "
            f"cache_hit={token_usage_summary['cache_hit_tokens']}, "
            f"cache_miss={token_usage_summary['cache_miss_tokens']}, "
            f"reasoning={token_usage_summary['reasoning_tokens']}"
        )
        
        if to_file:
            output_file = file_path.replace('.json', '_eval_ragas.csv')
            df = results.to_pandas()
            df.to_csv(output_file, index=False)
            print(f"Ragas evaluation saved to {output_file}")

            usage_file = file_path.replace('.json', '_eval_token_usage.json')
            with open(usage_file, 'w') as f:
                json.dump(token_usage_summary, f, indent=2, ensure_ascii=False)
            print(f"Ragas token usage saved to {usage_file}")
            
        return results

class AspectTester:
    def __init__(self, llm, metrics, reference_free, type):
        self.llm = llm
        self.metrics = metrics
        self.reference_free = reference_free
        self.type = type
        self.user_msgs = {} # Empty to skip execution in test_answers.py

    def evaluate_answers_from_json(self, file_path, to_file=True):
        print("AspectTester: Evaluation skipped (placeholder implementation).")

class GuidelineTester:
    def __init__(self, llm):
        self.llm = llm

    def evaluate_answers_from_json(self, file_path, to_file=True):
        print("GuidelineTester: Evaluation skipped (placeholder implementation).")

    def evaluate_answers_from_csv(self, file_path):
        print("GuidelineTester: Evaluation skipped (placeholder implementation).")
