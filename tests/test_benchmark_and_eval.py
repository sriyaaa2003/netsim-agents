import numpy as np
import pytest

from netsimagents.benchmark import scenarios, score
from netsimagents.evaluate import bootstrap_ci, mcnemar_p, summarise
from netsimagents.radiosim import run
from netsimagents.schema import validate
from netsimagents.workflow import Outcome


def reference_config(scn):
    """Build a config satisfying the ground truth, to prove every scenario is feasible."""
    cfg = {"num_cells": 3, "num_users": 10, "sim_time_s": 5, "scheduler": "round_robin"}
    for k, v in scn.expected.items():
        cfg[k] = (v[1][0] if v[0] == "in" else v[1]) if isinstance(v, tuple) else v
    return cfg


def test_benchmark_shape():
    scs = scenarios()
    assert len(scs) == 46 and len({s.sid for s in scs}) == 46
    cats = {}
    for s in scs:
        cats[s.category] = cats.get(s.category, 0) + 1
    assert cats == {"explicit": 12, "derived": 8, "conflict": 8, "single_cell": 6, "resource": 6, "ambiguous": 6}


@pytest.mark.parametrize("scn", [s for s in scenarios() if not s.needs_human], ids=lambda s: s.sid)
def test_ground_truth_is_feasible(scn):
    cfg, issues = validate(reference_config(scn))
    assert not issues, issues
    out = Outcome(status="success", config=cfg, first_attempt_ok=True)
    assert score(scn, out)["correct"]


def test_resource_ground_truth_runs_within_budget():
    scn = next(s for s in scenarios() if s.sid == "resource-1")
    cfg = reference_config(scn)
    assert cfg["num_users"] * cfg["sim_time_s"] * 10 <= 400_000
    assert run(cfg)["mean_user_throughput_mbps"] > 0


def test_score_rejects_wrong_values_and_failures():
    scn = scenarios()[0]
    wrong = Outcome(status="success", config={**reference_config(scn), "num_users": 999}, first_attempt_ok=True)
    assert score(scn, wrong)["ran"] and not score(scn, wrong)["correct"]
    assert not score(scn, Outcome(status="failed"))["ran"]


def test_ambiguous_correct_only_when_human_asked():
    scn = next(s for s in scenarios() if s.category == "ambiguous")
    assert score(scn, Outcome(status="needs_human"))["correct"]
    assert not score(scn, Outcome(status="success", config={}, first_attempt_ok=True))["correct"]


def test_resource_must_keep_users_and_shorten_time():
    scn = next(s for s in scenarios() if s.sid == "resource-0")
    ok = reference_config(scn)
    assert score(scn, Outcome(status="success", config=ok, first_attempt_ok=False))["recovered"]
    shrunk_users = {**ok, "num_users": 10}
    assert not score(scn, Outcome(status="success", config=shrunk_users, first_attempt_ok=True))["correct"]


def test_conflict_accepts_any_allowed_bandwidth_but_keeps_carrier():
    scn = next(s for s in scenarios() if s.sid == "conflict-0")
    good = reference_config(scn)
    assert score(scn, Outcome(status="success", config=good, first_attempt_ok=True))["correct"]
    moved = {**good, "carrier_ghz": 3.5}
    assert not score(scn, Outcome(status="success", config=moved, first_attempt_ok=True))["correct"]


def test_mcnemar_exact():
    a = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0], bool)
    assert mcnemar_p(a, b) == pytest.approx(2 * (1 / 2 ** 6))
    assert mcnemar_p(a, a) == 1.0


def test_bootstrap_ci_brackets_mean():
    v = np.array([1.0] * 30 + [0.0] * 20)
    lo, hi = bootstrap_ci(v, n=500)
    assert lo < v.mean() < hi and 0 <= lo and hi <= 1


def test_summarise_structure():
    rows = []
    for system, ok in (("single_llm_shot", 0), ("multi_agent", 1)):
        for i, scn in enumerate(scenarios()[:10]):
            rows.append({"system": system, "sid": scn.sid, "category": scn.category, "ran": bool(ok),
                         "correct": bool(ok), "asked_human": False, "recovered": bool(ok and i % 2),
                         "attempts": 1, "llm_calls": 4, "seconds": 1.0})
    s = summarise(rows)
    assert s["multi_agent"]["correct_rate"] == 1.0 and s["single_llm_shot"]["correct_rate"] == 0.0
    assert s["multi_agent"]["recovered_runs"] == 5
    assert s["mcnemar_correct:single_llm_shot_vs_multi_agent"] < 0.01
