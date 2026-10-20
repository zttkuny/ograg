from utils import create_service_context, get_config, load_llm_and_embeds
from qna import AspectTester, RAGASTester
import os

if __name__ == '__main__':
    config = get_config()
    config.model.enable_thinking=False
    llm, embeddings = load_llm_and_embeds(config.model, config.embedding_model)
    
    answer_dir = os.path.dirname(config.query.answers_file) or "."

    answer_file = os.path.basename(config.query.answers_file)
    if "." in answer_file:
        answer_fname, answer_ftype = answer_file.rsplit(".", 1)
    else:
        answer_fname, answer_ftype = answer_file, "json"

    if "hyperparams" in config.query and config.query.hyperparams:
        hyperparams_suffix = "_".join([f"{k}{v}" for k, v in config.query.hyperparams.items()])
        if hyperparams_suffix:
            config.query.answers_file = f"{answer_dir}/{answer_fname}_{hyperparams_suffix}.{answer_ftype}"

    config.evaluator.eval_file = config.query.answers_file

    # Fallback: if eval_file doesn't exist but answers_file does, use answers_file
    if not os.path.exists(config.evaluator.eval_file) and os.path.exists(config.query.answers_file):
        print(f"Warning: eval_file '{config.evaluator.eval_file}' not found. Switching to '{config.query.answers_file}'")
        config.evaluator.eval_file = config.query.answers_file

    if 'GuidelineFollow' in config.evaluator.metrics:
        from qna.tester import GuidelineTester
        tester = GuidelineTester(llm=llm)
        if '.json' in config.evaluator.eval_file:
            tester.evaluate_answers_from_json(config.evaluator.eval_file, to_file=True)
        elif '.csv' in config.evaluator.eval_file:
            scores = tester.evaluate_answers_from_csv(config.evaluator.eval_file)


    tester = RAGASTester(
        llm=llm,
        embeddings=embeddings,
        metrics=config.evaluator.metrics,
    )
    if tester.metrics != [] and ((not os.path.exists(config.evaluator.eval_file.replace('.json', '_eval_ragas.csv'))) or config.rewrite):
        tester.evaluate_answers_from_json(config.evaluator.eval_file, to_file=True)