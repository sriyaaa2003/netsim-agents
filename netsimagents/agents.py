"""The four LLM agents. Validation and execution are deterministic tools, not agents."""
from __future__ import annotations

import json

from .llm import LLM, extract_json
from .schema import parameter_docs

JSON_ONLY = "Reply with one JSON object only, no explanation outside the JSON."

PLANNER_SYSTEM = (
    "You are the Planner agent of a radio-network simulation workflow. Read the user's description of a "
    "scenario and study objective and extract what is stated.\n"
    "Simulator parameters:\n" + parameter_docs() + "\n"
    'Return {"objective": str, "stated": {parameter: value, ...}, "assumptions": [str], '
    '"needs_clarification": bool, "question": str}. Put only values the user stated or clearly implied in '
    '"stated" (for example pedestrian users imply 3 km/h). Set needs_clarification to true only when the '
    "description is too vague to configure any run (for example no network size, no user count and no "
    "scheduler can be inferred). " + JSON_ONLY
)

CONFIGURATOR_SYSTEM = (
    "You are the Configurator agent. Turn the Planner's plan into a simulator configuration.\n"
    "Simulator parameters:\n" + parameter_docs() + "\n"
    "Use only these parameter names. Fill every required parameter; leave optional ones out unless the plan "
    "states them. If you are given validator or simulator errors, fix exactly those problems while keeping "
    'every value the user asked for whenever the simulator allows it. Return {"config": {...}, "notes": str}. '
    + JSON_ONLY
)

REVIEWER_SYSTEM = (
    "You are the Reviewer agent. Compare the user's original description with the proposed configuration "
    "and decide whether the configuration matches the user's intent (numbers, scheduler, mobility, carrier). "
    "Ignore parameters the user did not mention. If the configuration differs from what the user asked only "
    "because of a documented simulator limit, that is acceptable.\n"
    'Return {"aligned": bool, "issues": [str]}. ' + JSON_ONLY
)

ANALYST_SYSTEM = (
    "You are the Analyst agent. Given the user's study objective and the simulator results, write a "
    "three-sentence summary that answers the objective using the numbers. "
    'Return {"summary": str}. ' + JSON_ONLY
)


class Agent:
    def __init__(self, role: str, system: str, llm: LLM) -> None:
        self.role, self.system, self.llm = role, system, llm

    def ask(self, user: str) -> tuple[dict, str]:
        """Return (parsed JSON, raw reply). Raises ValueError when the reply is not JSON."""
        raw = self.llm.generate([{"role": "system", "content": self.system}, {"role": "user", "content": user}])
        return extract_json(raw), raw


def planner(llm: LLM) -> Agent:
    return Agent("planner", PLANNER_SYSTEM, llm)


def configurator(llm: LLM) -> Agent:
    return Agent("configurator", CONFIGURATOR_SYSTEM, llm)


def reviewer(llm: LLM) -> Agent:
    return Agent("reviewer", REVIEWER_SYSTEM, llm)


def analyst(llm: LLM) -> Agent:
    return Agent("analyst", ANALYST_SYSTEM, llm)


def j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
