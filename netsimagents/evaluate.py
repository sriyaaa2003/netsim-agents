"""Run the three systems over the benchmark and summarise with bootstrap intervals and McNemar tests."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from .benchmark import scenarios, score
from .llm import LLM
from .workflow import MultiAgentWorkflow, SingleLLMBaseline

SYSTEMS = ("single_llm_shot", "single_llm_retry", "multi_agent")


def build(system: str, llm: LLM, fmt: str = "json"):
    return {"single_llm_shot": lambda: SingleLLMBaseline(llm, retries=0, fmt=fmt),
            "single_llm_retry": lambda: SingleLLMBaseline(llm, retries=3, fmt=fmt),
            "multi_agent": lambda: MultiAgentWorkflow(llm, fmt=fmt)}[system]()


def run_all(llm: LLM, systems=SYSTEMS, limit: int | None = None, out_dir: str | None = None, fmt: str = "json") -> list[dict]:
    scs = scenarios()[:limit] if limit else scenarios()
    rows, traces = [], []
    for system in systems:
        wf = build(system, llm, fmt)
        for scn in scs:
            outcome = wf.run(scn.text)
            row = score(scn, outcome)
            row["system"] = system
            rows.append(row)
            traces.append({"system": system, "sid": scn.sid, "text": scn.text, **outcome.as_dict()})
            print(f"{system:17s} {scn.sid:14s} ran={row['ran']!s:5s} correct={row['correct']!s:5s} calls={row['llm_calls']}", flush=True)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "rows.json").write_text(json.dumps(rows, indent=1))
        (Path(out_dir) / "traces.json").write_text(json.dumps(traces, default=str))
    return rows


def bootstrap_ci(values: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired booleans."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    by_sys = {s: sorted([r for r in rows if r["system"] == s], key=lambda r: r["sid"]) for s in SYSTEMS if any(r["system"] == s for r in rows)}
    for s, rs in by_sys.items():
        ran, correct = np.array([r["ran"] for r in rs], float), np.array([r["correct"] for r in rs], float)
        recov = [r for r in rs if r["recovered"]]
        out[s] = {
            "n": len(rs), "success_rate": ran.mean(), "correct_rate": correct.mean(),
            "correct_ci95": bootstrap_ci(correct), "recovered_runs": len(recov),
            "asked_human": int(sum(r["asked_human"] for r in rs)),
            "mean_llm_calls": float(np.mean([r["llm_calls"] for r in rs])),
            "mean_seconds": float(np.mean([r["seconds"] for r in rs])),
            "correct_by_category": {c: float(np.mean([r["correct"] for r in rs if r["category"] == c]))
                                    for c in sorted({r["category"] for r in rs})},
        }
    names = list(by_sys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca = np.array([r["correct"] for r in by_sys[a]], bool)
            cb = np.array([r["correct"] for r in by_sys[b]], bool)
            out[f"mcnemar_correct:{a}_vs_{b}"] = mcnemar_p(ca, cb)
    return out


def main(argv: list[str]) -> None:  # pragma: no cover - needs a real model
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model")
    ap.add_argument("--openai-url")
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--format", choices=("json", "xml"), default="json", help="configuration format the LLM writes")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    if a.hf_model:
        from .llm import HFLocalLLM

        llm = HFLocalLLM(a.hf_model)
    else:
        from .llm import OpenAICompatLLM

        llm = OpenAICompatLLM(a.openai_url, a.model)
    rows = run_all(llm, limit=a.limit, out_dir=a.out, fmt=a.format)
    summary = summarise(rows)
    summary["llm"] = llm.name
    Path(a.out, "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
