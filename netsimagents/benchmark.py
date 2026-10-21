"""Benchmark scenarios with ground-truth checks. 46 scenarios in six categories, two phrasings alternating."""
from __future__ import annotations

from dataclasses import dataclass

from .schema import ALLOWED_BANDWIDTH_MHZ, MAX_USER_STEPS, STEPS_PER_S

SCHED_WORDS = {
    "round_robin": ("round-robin", "round robin"),
    "proportional_fair": ("proportional-fair", "proportional fair"),
    "max_cqi": ("max-CQI", "maximum-CQI"),
}
ENV_SPEED = {"stationary": 0.0, "pedestrian": 3.0, "urban vehicular": 50.0, "highway": 120.0}


@dataclass(frozen=True)
class Scenario:
    sid: str
    category: str  # explicit | derived | conflict | single_cell | resource | ambiguous
    text: str
    expected: dict  # param -> value | ("ge", x) | ("le", x) | ("in", [..]); empty for ambiguous
    needs_human: bool = False


def _explicit() -> list[Scenario]:
    rows = [(7, 50, 20, "round_robin", 3.5), (1, 20, 30, "proportional_fair", 2.6), (19, 100, 15, "max_cqi", 1.8),
            (7, 200, 30, "proportional_fair", 0.7), (3, 40, 60, "round_robin", 3.5), (12, 80, 25, "max_cqi", 2.6),
            (7, 10, 120, "round_robin", 1.8), (19, 60, 40, "proportional_fair", 3.5), (2, 30, 45, "max_cqi", 0.7),
            (5, 150, 20, "round_robin", 2.6), (1, 5, 200, "proportional_fair", 1.8), (9, 120, 30, "max_cqi", 3.5)]
    out = []
    for i, (cells, users, t, sch, ghz) in enumerate(rows):
        s1, s2 = SCHED_WORDS[sch]
        if i % 2 == 0:
            text = f"Simulate a network of {cells} cells serving {users} users for {t} seconds with the {s1} scheduler on a {ghz} GHz carrier."
        else:
            text = f"I want to study {users} users in a {cells}-cell deployment at {ghz} GHz, scheduled {s2}, for a {t} s run."
        out.append(Scenario(f"explicit-{i}", "explicit", text, {
            "num_cells": cells, "num_users": users, "sim_time_s": t, "scheduler": sch, "carrier_ghz": ghz}))
    return out


def _derived() -> list[Scenario]:
    rows = [("pedestrian", 7, 60, 30, "proportional_fair"), ("highway", 7, 40, 30, "round_robin"),
            ("urban vehicular", 19, 100, 20, "proportional_fair"), ("stationary", 3, 30, 20, "max_cqi"),
            ("pedestrian", 1, 25, 40, "round_robin"), ("highway", 12, 70, 25, "max_cqi"),
            ("urban vehicular", 5, 45, 30, "round_robin"), ("stationary", 7, 80, 20, "proportional_fair")]
    out = []
    for i, (env, cells, users, t, sch) in enumerate(rows):
        s1, s2 = SCHED_WORDS[sch]
        if i % 2 == 0:
            text = f"A {env} scenario with {cells} cells and {users} users, {s1} scheduling, simulated for {t} seconds."
        else:
            text = f"Evaluate {users} {env} users over {cells} cells using {s2} scheduling for {t} s."
        out.append(Scenario(f"derived-{i}", "derived", text, {
            "num_cells": cells, "num_users": users, "sim_time_s": t, "scheduler": sch,
            "user_speed_kmh": ENV_SPEED[env]}))
    return out


def _conflict() -> list[Scenario]:
    rows = [(0.7, 20, 7, 40, 30), (1.8, 5, 3, 30, 20), (2.6, 15, 7, 60, 25), (2.6, 40, 5, 50, 30),
            (3.5, 10, 7, 40, 20), (1.8, 100, 1, 20, 30), (0.7, 100, 9, 60, 20), (3.5, 5, 3, 25, 40)]
    out = []
    for i, (ghz, bw, cells, users, t) in enumerate(rows):
        if i % 2 == 0:
            text = f"Run {cells} cells with {users} users for {t} seconds at {ghz} GHz with {bw} MHz of bandwidth, round-robin scheduler."
        else:
            text = f"Use a {bw} MHz channel on the {ghz} GHz carrier, {cells} cells, {users} users, {t} s, round robin."
        out.append(Scenario(f"conflict-{i}", "conflict", text, {
            "carrier_ghz": ghz, "bandwidth_mhz": ("in", list(ALLOWED_BANDWIDTH_MHZ[ghz])), "num_cells": cells,
            "num_users": users, "sim_time_s": t, "scheduler": "round_robin"}))
    return out


def _single_cell() -> list[Scenario]:
    rows = [(4, 20, 30, "round_robin"), (6, 40, 20, "proportional_fair"), (2, 10, 60, "max_cqi"),
            (5, 30, 30, "proportional_fair"), (3, 50, 20, "round_robin"), (8, 25, 40, "max_cqi")]
    out = []
    for i, (hys, users, t, sch) in enumerate(rows):
        s1, s2 = SCHED_WORDS[sch]
        if i % 2 == 0:
            text = f"One single cell with {users} users, {s1} scheduler, handover hysteresis of {hys} dB, {t} seconds."
        else:
            text = f"Simulate a lone cell serving {users} users for {t} s with {s2} scheduling and a {hys} dB handover hysteresis."
        out.append(Scenario(f"single-{i}", "single_cell", text, {
            "num_cells": 1, "num_users": users, "sim_time_s": t, "scheduler": sch, "handover_hysteresis_db": 0.0}))
    return out


def _resource() -> list[Scenario]:
    rows = [(400, 300, 7), (500, 600, 19), (300, 500, 3), (450, 400, 12), (350, 600, 7), (500, 300, 5)]
    out = []
    for i, (users, t, cells) in enumerate(rows):
        limit = MAX_USER_STEPS // (users * STEPS_PER_S)
        if i % 2 == 0:
            text = f"Simulate {users} users in {cells} cells for {t} seconds with the proportional-fair scheduler."
        else:
            text = f"I need a {t} s run of {cells} cells with {users} users, proportional fair scheduling."
        out.append(Scenario(f"resource-{i}", "resource", text, {
            "num_users": users, "num_cells": cells, "scheduler": "proportional_fair", "sim_time_s": ("le", limit)}))
    return out


def _ambiguous() -> list[Scenario]:
    texts = ["Evaluate how well the network performs.", "See what happens when users get busy.",
             "Check the network for me.", "Run a simulation of the system to compare options.",
             "Find out whether the radio network is good enough.", "Test the setup we discussed yesterday."]
    return [Scenario(f"ambiguous-{i}", "ambiguous", t, {}, needs_human=True) for i, t in enumerate(texts)]


def scenarios() -> list[Scenario]:
    return _explicit() + _derived() + _conflict() + _single_cell() + _resource() + _ambiguous()


def _matches(actual, want) -> bool:
    if isinstance(want, tuple):
        kind, arg = want
        if kind == "ge":
            return actual >= arg
        if kind == "le":
            return actual <= arg
        return actual in arg
    if isinstance(want, float) or isinstance(actual, float):
        return abs(float(actual) - float(want)) < 1e-9
    return actual == want


def score(scn: Scenario, outcome) -> dict:
    """success: ran without error; correct: matches the ground truth (for ambiguous: asked a human)."""
    ran = outcome.status == "success"
    if scn.needs_human:
        correct = outcome.status == "needs_human"
    elif ran:
        correct = all(_matches(outcome.config[k], v) for k, v in scn.expected.items())
    else:
        correct = False
    return {"sid": scn.sid, "category": scn.category, "ran": ran, "correct": correct,
            "asked_human": outcome.status == "needs_human",
            "recovered": ran and not outcome.first_attempt_ok, "attempts": outcome.attempts,
            "llm_calls": outcome.llm_calls, "seconds": outcome.seconds}
