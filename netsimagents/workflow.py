"""The multi-agent closed-loop workflow and the two single-LLM baselines."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import agents as ag
from .llm import LLM, extract_json
from .radiosim import SimulationError, run as run_sim
from .schema import parameter_docs, validate
from .xmlio import extract_xml, parse_xml

MAX_ATTEMPTS = 4  # configuration attempts per scenario, including the first


@dataclass
class Outcome:
    status: str  # "success" | "needs_human" | "failed"
    config: dict | None = None
    results: dict | None = None
    summary: str = ""
    question: str = ""
    attempts: int = 0
    first_attempt_ok: bool = False
    trace: list = field(default_factory=list)
    llm_calls: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "status", "config", "results", "summary", "question", "attempts",
            "first_attempt_ok", "llm_calls", "seconds", "trace")}


def _tool_check(config: dict) -> tuple[dict | None, list[dict]]:
    """Validator then simulator. Returns (results or None, errors)."""
    cfg, issues = validate(config)
    if issues:
        return None, [{"source": "validator", **i.as_dict()} for i in issues]
    try:
        return run_sim(cfg), []
    except SimulationError as exc:
        return None, [{"source": "simulator", "code": "execution_error", "param": "", "message": str(exc)}]


def _xml_config(raw: str) -> tuple[dict | None, list[dict]]:
    """XML stages in front of the validator: extract, safety and well-formedness, structure and types."""
    try:
        doc = extract_xml(raw)
    except ValueError as exc:
        return None, [{"source": "format", "code": "bad_reply", "param": "", "message": str(exc)}]
    config, issues = parse_xml(doc)
    if issues:
        return None, [{"source": "xml", **i.as_dict()} for i in issues]
    return config, []


class MultiAgentWorkflow:
    """Planner -> Configurator -> (validator, simulator tools) -> Reviewer -> Analyst, with an error loop."""

    name = "multi_agent"

    def __init__(self, llm: LLM, use_reviewer: bool = True, fmt: str = "json") -> None:
        self.llm, self.fmt = llm, fmt
        self.planner, self.configurator = ag.planner(llm), ag.configurator(llm, fmt)
        self.reviewer, self.analyst = ag.reviewer(llm), ag.analyst(llm)
        self.use_reviewer = use_reviewer

    def run(self, text: str) -> Outcome:
        out, start, calls0 = Outcome(status="failed"), time.perf_counter(), self.llm.calls

        def log(step, agent, **kw):
            out.trace.append({"step": step, "agent": agent, **kw})

        try:
            plan, raw = self.planner.ask(text)
        except ValueError as exc:
            log("plan", "planner", ok=False, error=str(exc))
            return self._finish(out, start, calls0)
        log("plan", "planner", ok=True, output=plan)
        if plan.get("needs_clarification") is True:
            out.status, out.question = "needs_human", str(plan.get("question", ""))
            return self._finish(out, start, calls0)

        feedback: list[dict] = []
        previous: dict | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            out.attempts = attempt
            user = ag.j({"description": text, "plan": plan, "previous_config": previous, "errors": feedback})
            if self.fmt == "xml":
                raw = self.configurator.ask_raw(user)
                config, xml_errors = _xml_config(raw)
                if config is None:
                    feedback, previous = xml_errors, raw[:600]
                    log("configure", "configurator", attempt=attempt, ok=False, errors=feedback)
                    continue
            else:
                try:
                    reply, raw = self.configurator.ask(user)
                    config = reply["config"]
                    if not isinstance(config, dict):
                        raise ValueError("'config' is not an object")
                except (ValueError, KeyError) as exc:
                    feedback = [{"source": "format", "code": "bad_reply", "param": "", "message": f"reply was not usable JSON with a 'config' object: {exc}"}]
                    log("configure", "configurator", attempt=attempt, ok=False, errors=feedback)
                    continue
            previous = config
            results, errors = _tool_check(config)
            log("configure", "configurator", attempt=attempt, config=config, ok=not errors, errors=errors)
            if errors:
                feedback = errors
                continue
            if self.use_reviewer:
                try:
                    verdict, _ = self.reviewer.ask(ag.j({"description": text, "config": results["config"]}))
                except ValueError:
                    verdict = {"aligned": True, "issues": []}
                log("review", "reviewer", attempt=attempt, output=verdict)
                if verdict.get("aligned") is False and attempt < MAX_ATTEMPTS:
                    feedback = [{"source": "reviewer", "code": "intent_mismatch", "param": "", "message": str(i)}
                                for i in verdict.get("issues", [])] or [
                        {"source": "reviewer", "code": "intent_mismatch", "param": "", "message": "does not match the description"}]
                    continue
            out.status, out.config, out.results = "success", results["config"], {k: v for k, v in results.items() if k != "config"}
            out.first_attempt_ok = attempt == 1
            try:
                summ, _ = self.analyst.ask(ag.j({"objective": plan.get("objective", text), "results": out.results}))
                out.summary = str(summ.get("summary", ""))
            except ValueError:
                out.summary = ""
            log("analyse", "analyst", ok=bool(out.summary))
            break
        return self._finish(out, start, calls0)

    def _finish(self, out: Outcome, start: float, calls0: int) -> Outcome:
        out.seconds, out.llm_calls = time.perf_counter() - start, self.llm.calls - calls0
        return out


BASELINE_SYSTEM = (
    "You configure a radio-network simulator from a natural-language description.\n"
    "Simulator parameters:\n" + parameter_docs() + "\n"
    "Use only these parameter names. Fill every required parameter. If the description is too vague to "
    "configure any run, return {\"needs_clarification\": true, \"question\": str} instead. Otherwise return "
    '{"config": {...}}. If you are given simulator or validator errors, fix them while keeping every value the '
    "user asked for whenever the simulator allows it. Reply with one JSON object only."
)


BASELINE_XML_SYSTEM = (
    "You configure a radio-network simulator from a natural-language description and write the "
    "configuration as XML.\n"
    "Simulator parameters:\n" + parameter_docs() + "\n"
    "Write one <scenario> element containing one child element per parameter, with the value as plain text, "
    "for example <scenario><num_cells>7</num_cells><scheduler>round_robin</scheduler></scenario>. Use only "
    "these parameter names, each at most once, no attributes, no nested elements and no DOCTYPE. Fill every "
    "required parameter. If the description is too vague to configure any run, reply with the JSON object "
    '{"needs_clarification": true, "question": str} instead. If you are given errors, fix them while keeping '
    "every value the user asked for whenever the simulator allows it. Reply with the XML document only."
)


class SingleLLMBaseline:
    """One LLM, one prompt. With retries=0 it is a single shot; with retries>0 it sees the same error feedback."""

    def __init__(self, llm: LLM, retries: int = 0, fmt: str = "json") -> None:
        self.llm, self.retries, self.fmt = llm, retries, fmt
        self.name = "single_llm_retry" if retries else "single_llm_shot"

    def run(self, text: str) -> Outcome:
        out, start, calls0 = Outcome(status="failed"), time.perf_counter(), self.llm.calls
        feedback: list[dict] = []
        previous = None
        for attempt in range(1, self.retries + 2):
            out.attempts = attempt
            user = ag.j({"description": text, "previous_config": previous, "errors": feedback})
            system = BASELINE_XML_SYSTEM if self.fmt == "xml" else BASELINE_SYSTEM
            raw = self.llm.generate([{"role": "system", "content": system}, {"role": "user", "content": user}])
            if self.fmt == "xml" and "<scenario" in raw:
                config, xml_errors = _xml_config(raw)
                if config is None:
                    feedback, previous = xml_errors, raw[:600]
                    out.trace.append({"step": "configure", "agent": "single", "attempt": attempt, "ok": False, "errors": feedback})
                    continue
            else:
                try:
                    reply = extract_json(raw)
                except ValueError as exc:
                    feedback = [{"source": "format", "code": "bad_reply", "param": "", "message": str(exc)}]
                    out.trace.append({"step": "configure", "agent": "single", "attempt": attempt, "ok": False, "errors": feedback})
                    continue
                if reply.get("needs_clarification") is True:
                    out.status, out.question = "needs_human", str(reply.get("question", ""))
                    break
                config = reply.get("config")
                if not isinstance(config, dict):
                    feedback = [{"source": "format", "code": "bad_reply", "param": "", "message": "reply has no 'config' object"}]
                    out.trace.append({"step": "configure", "agent": "single", "attempt": attempt, "ok": False, "errors": feedback})
                    continue
            previous = config
            results, errors = _tool_check(config)
            out.trace.append({"step": "configure", "agent": "single", "attempt": attempt, "config": config, "ok": not errors, "errors": errors})
            if errors:
                feedback = errors
                continue
            out.status, out.config = "success", results["config"]
            out.results = {k: v for k, v in results.items() if k != "config"}
            out.first_attempt_ok = attempt == 1
            break
        out.seconds, out.llm_calls = time.perf_counter() - start, self.llm.calls - calls0
        return out
