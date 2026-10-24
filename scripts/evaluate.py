"""Evaluate retrieval (and optionally answers) on samples/eval_set.json.

Usage:
    uv run python scripts/evaluate.py             # retrieval only, ~30 s
    uv run python scripts/evaluate.py --answers   # + LLM answers & judge, several minutes on CPU

Results are saved to data/eval/ and shown on the app's Evaluation page.
"""

import argparse
import sys

from research_analyst.evaluation import run_evaluation, save_results


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", action="store_true", help="also generate and grade answers (needs the LLM)")
    args = parser.parse_args()

    results = run_evaluation(with_answers=args.answers)

    print(f"Retrieval ({results['config']['cases']} questions, top-{results['config']['top_k']}):\n")
    metrics = [m for m in next(iter(results["retrieval"].values())) if m != "ranks"]
    print(f"  {'setup':<18}" + "".join(f"{m:>12}" for m in metrics))
    for setup, scores in results["retrieval"].items():
        print(f"  {setup:<18}" + "".join(f"{scores[m]:>12.3f}" if isinstance(scores[m], float) else f"{scores[m]:>12}" for m in metrics))

    if answers := results.get("answers"):
        print(f"\nAnswers ({answers['setup']}, judge: {results['config']['judge_model']}):\n")
        for key in ["keyword_recall", "citation_rate", "grounding", "faithfulness", "refusal_accuracy", "latency_s"]:
            print(f"  {key:<18}{answers[key]}")

    print(f"\n[saved to {save_results(results)}]")


if __name__ == "__main__":
    main()
