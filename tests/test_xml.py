import json

import pytest

from netsimagents.llm import ScriptedLLM
from netsimagents.radiosim import run
from netsimagents.schema import validate
from netsimagents.workflow import MultiAgentWorkflow, SingleLLMBaseline
from netsimagents.xmlio import extract_xml, parse_xml, to_xml

GOOD = {"num_cells": 7, "num_users": 20, "sim_time_s": 5, "scheduler": "round_robin"}
GOOD_XML = to_xml(GOOD)
PLAN = {"objective": "throughput", "stated": GOOD, "assumptions": [], "needs_clarification": False, "question": ""}


def codes(text):
    return {i.code for i in parse_xml(text)[1]}


def test_roundtrip_keeps_values_and_types():
    cfg = {**GOOD, "carrier_ghz": 2.6, "bandwidth_mhz": 20, "user_speed_kmh": 50, "traffic": "bursty"}
    back, issues = parse_xml(to_xml(cfg))
    assert not issues and back == {**cfg, "user_speed_kmh": 50.0, "carrier_ghz": 2.6}
    assert isinstance(back["num_cells"], int) and isinstance(back["carrier_ghz"], float)


def test_parsed_config_passes_validator_and_runs():
    cfg, issues = parse_xml(GOOD_XML)
    assert not issues and not validate(cfg)[1]
    assert run(cfg)["mean_user_throughput_mbps"] > 0


@pytest.mark.parametrize("text,code", [
    ("<scenario><num_cells>7</scenario>", "xml_not_well_formed"),
    ("<scenario><num_cells>7</num_cells>", "xml_not_well_formed"),
    ("<scenario>7 < 8</scenario>", "xml_not_well_formed"),
    ("not xml at all", "xml_not_well_formed"),
    ("<config><num_cells>7</num_cells></config>", "xml_wrong_root"),
    ("<scenario><num_cells>7</num_cells><num_cells>8</num_cells></scenario>", "xml_duplicate_element"),
    ("<scenario><cells><num_cells>7</num_cells></cells></scenario>", "xml_unexpected_nesting"),
    ("<scenario><num_users>many</num_users></scenario>", "wrong_type"),
    ("<scenario><num_users>7.5</num_users></scenario>", "wrong_type"),
    ("<scenario><user_speed_kmh>fast</user_speed_kmh></scenario>", "wrong_type"),
    ('<!DOCTYPE scenario [<!ENTITY a "x">]><scenario><scheduler>&a;</scheduler></scenario>', "xml_forbidden_construct"),
    ("<!ENTITY a 'b'><scenario/>", "xml_forbidden_construct"),
])
def test_malformed_xml_is_caught_with_a_specific_code(text, code):
    cfg, issues = parse_xml(text)
    assert cfg is None and code in {i.code for i in issues}


def test_entity_expansion_attack_is_rejected_before_parsing():
    bomb = '<!DOCTYPE s [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;">]><scenario><scheduler>&b;</scheduler></scenario>'
    assert codes(bomb) == {"xml_forbidden_construct"}


def test_unknown_element_passes_xml_stage_but_validator_flags_it():
    cfg, issues = parse_xml("<scenario><colour>red</colour></scenario>")
    assert not issues and cfg == {"colour": "red"}
    assert "unknown_parameter" in {i.code for i in validate(cfg)[1]}


def test_whitespace_and_empty_elements():
    cfg, issues = parse_xml("<scenario>\n  <num_cells> 7 </num_cells>\n  <scheduler>\n round_robin\n </scheduler>\n</scenario>")
    assert not issues and cfg == {"num_cells": 7, "scheduler": "round_robin"}
    assert "wrong_type" in codes("<scenario><num_cells/></scenario>")


@pytest.mark.parametrize("reply", [
    "Here is the configuration:\n```xml\n" + GOOD_XML + "\n```\nHope it helps!",
    "Sure. " + GOOD_XML,
    GOOD_XML,
])
def test_extract_xml_handles_fences_and_chatter(reply):
    assert extract_xml(reply).startswith("<scenario") and parse_xml(extract_xml(reply))[0] is not None


def test_extract_xml_fails_without_a_document():
    with pytest.raises(ValueError):
        extract_xml("I cannot do that.")
    with pytest.raises(ValueError):
        extract_xml('{"config": {}}')


def role(messages):
    s = messages[0]["content"]
    for r in ("Planner", "Configurator", "Reviewer", "Analyst"):
        if f"You are the {r} agent" in s:
            return r.lower()
    return "single"


def make(replies, plan=None):
    state = {"i": 0, "seen": []}

    def fn(messages):
        r = role(messages)
        user = json.loads(messages[1]["content"]) if messages[1]["content"].startswith("{") else {}
        if r == "planner":
            return json.dumps(plan or PLAN)
        if r in ("configurator", "single"):
            state["seen"].append(user.get("errors"))
            out = replies[min(state["i"], len(replies) - 1)]
            state["i"] += 1
            return out
        if r == "reviewer":
            return json.dumps({"aligned": True, "issues": []})
        return json.dumps({"summary": "ok"})

    return ScriptedLLM(fn), state


def test_multi_agent_xml_success():
    llm, _ = make([GOOD_XML])
    out = MultiAgentWorkflow(llm, fmt="xml").run("text")
    assert out.status == "success" and out.first_attempt_ok and out.config["num_cells"] == 7


def test_multi_agent_xml_recovers_from_malformed_xml():
    llm, state = make(["<scenario><num_cells>7</scenario>", GOOD_XML])
    out = MultiAgentWorkflow(llm, fmt="xml").run("text")
    assert out.status == "success" and out.attempts == 2 and not out.first_attempt_ok
    assert state["seen"][1][0]["code"] == "xml_not_well_formed" and state["seen"][1][0]["source"] == "xml"
    assert out.trace[1]["ok"] is False


def test_multi_agent_xml_recovers_from_validator_error_after_xml_passes():
    bad = to_xml({**GOOD, "carrier_ghz": 0.7, "bandwidth_mhz": 20})
    fixed = to_xml({**GOOD, "carrier_ghz": 0.7, "bandwidth_mhz": 10})
    llm, state = make([bad, fixed])
    out = MultiAgentWorkflow(llm, fmt="xml").run("text")
    assert out.status == "success" and state["seen"][1][0]["source"] == "validator"
    assert out.config["bandwidth_mhz"] == 10


def test_multi_agent_xml_reply_without_document_is_a_format_error():
    llm, state = make(["I think seven cells", GOOD_XML])
    out = MultiAgentWorkflow(llm, fmt="xml").run("text")
    assert out.status == "success" and state["seen"][1][0]["code"] == "bad_reply"


def configurator_prompts(fmt, reply):
    seen = []

    def fn(messages):
        if "You are the Planner agent" in messages[0]["content"]:
            return json.dumps(PLAN)
        if "You are the Configurator agent" in messages[0]["content"]:
            seen.append(messages[0]["content"])
            return reply
        return json.dumps({"aligned": True, "issues": [], "summary": "ok"})

    MultiAgentWorkflow(ScriptedLLM(fn), fmt=fmt).run("t")
    return seen


def test_xml_prompt_is_used_in_xml_mode_only():
    xml_prompts = configurator_prompts("xml", GOOD_XML)
    json_prompts = configurator_prompts("json", json.dumps({"config": GOOD}))
    assert xml_prompts and "<scenario>" in xml_prompts[0] and "XML" in xml_prompts[0]
    assert json_prompts and "<scenario>" not in json_prompts[0] and "XML" not in json_prompts[0]


def test_single_xml_shot_fails_on_malformed_and_retry_repairs():
    llm, _ = make(["<scenario><num_cells>7</scenario>", GOOD_XML])
    assert SingleLLMBaseline(llm, retries=0, fmt="xml").run("t").status == "failed"
    llm, _ = make(["<scenario><num_cells>7</scenario>", GOOD_XML])
    out = SingleLLMBaseline(llm, retries=3, fmt="xml").run("t")
    assert out.status == "success" and out.attempts == 2


def test_single_xml_can_still_ask_for_clarification_in_json():
    llm = ScriptedLLM(lambda m: json.dumps({"needs_clarification": True, "question": "which?"}))
    out = SingleLLMBaseline(llm, fmt="xml").run("vague")
    assert out.status == "needs_human" and out.question == "which?"


def test_default_format_is_still_json():
    llm = ScriptedLLM(lambda m: json.dumps(PLAN) if "You are the Planner agent" in m[0]["content"] else json.dumps({"config": GOOD}))
    assert MultiAgentWorkflow(llm, use_reviewer=False).run("t").status == "success"
