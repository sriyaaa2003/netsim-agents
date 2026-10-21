import json

from netsimagents.llm import ScriptedLLM
from netsimagents.workflow import MAX_ATTEMPTS, MultiAgentWorkflow, SingleLLMBaseline

GOOD = {"num_cells": 7, "num_users": 20, "sim_time_s": 5, "scheduler": "round_robin"}
PLAN = {"objective": "throughput", "stated": GOOD, "assumptions": [], "needs_clarification": False, "question": ""}


def role(messages):
    s = messages[0]["content"]
    for r in ("Planner", "Configurator", "Reviewer", "Analyst"):
        if f"You are the {r} agent" in s:
            return r.lower()
    return "single"


def make(configs, plan=None, aligned=None):
    """Scripted LLM: configurator replies come from `configs` in order (the last one repeats)."""
    state = {"i": 0, "seen": []}

    def fn(messages):
        r = role(messages)
        user = json.loads(messages[1]["content"]) if messages[1]["content"].startswith("{") else {}
        if r == "planner":
            return json.dumps(plan if plan is not None else PLAN)
        if r in ("configurator", "single"):
            state["seen"].append(user.get("errors"))
            cfg = configs[min(state["i"], len(configs) - 1)]
            state["i"] += 1
            return cfg if isinstance(cfg, str) else json.dumps({"config": cfg})
        if r == "reviewer":
            v = aligned.pop(0) if aligned else True
            return json.dumps({"aligned": v, "issues": [] if v else ["scheduler differs"]})
        return json.dumps({"summary": "ok"})

    return ScriptedLLM(fn), state


def test_multi_agent_success_first_attempt():
    llm, _ = make([GOOD])
    out = MultiAgentWorkflow(llm).run("text")
    assert out.status == "success" and out.attempts == 1 and out.first_attempt_ok
    assert out.results["mean_user_throughput_mbps"] > 0 and out.summary == "ok"
    assert out.llm_calls == 4  # planner, configurator, reviewer, analyst
    assert [t["agent"] for t in out.trace] == ["planner", "configurator", "reviewer", "analyst"]


def test_recovers_from_validator_error_using_feedback():
    bad = {**GOOD, "carrier_ghz": 0.7, "bandwidth_mhz": 20}
    fixed = {**GOOD, "carrier_ghz": 0.7, "bandwidth_mhz": 10}
    llm, state = make([bad, fixed])
    out = MultiAgentWorkflow(llm).run("text")
    assert out.status == "success" and out.attempts == 2 and not out.first_attempt_ok
    assert state["seen"][0] == [] and state["seen"][1][0]["code"] == "bandwidth_not_allowed_for_carrier"
    assert out.config["bandwidth_mhz"] == 10


def test_recovers_from_execution_error():
    heavy = {**GOOD, "num_users": 500, "sim_time_s": 600}
    ok = {**GOOD, "num_users": 500, "sim_time_s": 60}
    llm, state = make([heavy, ok])
    out = MultiAgentWorkflow(llm).run("text")
    assert out.status == "success" and out.config["sim_time_s"] == 60
    assert state["seen"][1][0]["source"] == "simulator"


def test_unusable_reply_is_fed_back_as_format_error():
    llm, state = make(["I think about 7 cells", GOOD])
    out = MultiAgentWorkflow(llm).run("text")
    assert out.status == "success" and out.attempts == 2
    assert state["seen"][1][0]["code"] == "bad_reply"


def test_needs_human_stops_before_configuring():
    plan = {**PLAN, "needs_clarification": True, "question": "How many users?"}
    llm, state = make([GOOD], plan=plan)
    out = MultiAgentWorkflow(llm).run("vague")
    assert out.status == "needs_human" and out.question == "How many users?" and state["seen"] == []
    assert out.config is None


def test_planner_without_json_fails_cleanly():
    llm = ScriptedLLM(lambda m: "no json here")
    out = MultiAgentWorkflow(llm).run("text")
    assert out.status == "failed" and out.trace[0]["ok"] is False


def test_reviewer_mismatch_triggers_another_attempt():
    other = {**GOOD, "scheduler": "max_cqi"}
    llm, state = make([other, GOOD], aligned=[False, True])
    out = MultiAgentWorkflow(llm).run("text")
    assert out.status == "success" and out.attempts == 2 and out.config["scheduler"] == "round_robin"
    assert state["seen"][1][0]["source"] == "reviewer"


def test_attempt_budget_is_enforced():
    llm, _ = make([{**GOOD, "scheduler": "fifo"}])
    out = MultiAgentWorkflow(llm).run("text")
    assert out.status == "failed" and out.attempts == MAX_ATTEMPTS and out.config is None


def test_reviewer_can_be_switched_off():
    llm, _ = make([GOOD])
    out = MultiAgentWorkflow(llm, use_reviewer=False).run("text")
    assert out.llm_calls == 3 and out.status == "success"


def test_single_shot_does_not_retry():
    bad = {**GOOD, "scheduler": "fifo"}
    llm, _ = make([bad, GOOD])
    out = SingleLLMBaseline(llm, retries=0).run("text")
    assert out.status == "failed" and out.attempts == 1 and out.llm_calls == 1


def test_single_with_retries_recovers():
    bad = {**GOOD, "scheduler": "fifo"}
    llm, _ = make([bad, GOOD])
    out = SingleLLMBaseline(llm, retries=3).run("text")
    assert out.status == "success" and out.attempts == 2 and not out.first_attempt_ok


def test_single_can_ask_for_clarification():
    llm = ScriptedLLM(lambda m: json.dumps({"needs_clarification": True, "question": "which?"}))
    out = SingleLLMBaseline(llm).run("vague")
    assert out.status == "needs_human" and out.question == "which?"


def test_outcome_serialises_to_json():
    llm, _ = make([GOOD])
    json.dumps(MultiAgentWorkflow(llm).run("text").as_dict(), default=str)
