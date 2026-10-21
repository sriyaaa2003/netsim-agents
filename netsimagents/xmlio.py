"""XML scenario format: a first, deterministic checking stage in front of the schema validator.

A scenario is one ``<scenario>`` element with one child element per parameter::

    <scenario>
      <num_cells>7</num_cells>
      <num_users>50</num_users>
      <scheduler>round_robin</scheduler>
    </scenario>

Stage order for XML replies: extract the document, check it is safe and well-formed, check its
structure (root, duplicates, nesting) and convert values to the schema's types. The schema
validator and the simulator then run exactly as for JSON.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .schema import PARAMS, Issue

ROOT = "scenario"
_DOC = re.compile(r"<scenario\b.*?</scenario\s*>", re.S)
_FORBIDDEN = re.compile(r"<!\s*(DOCTYPE|ENTITY)|<\?xml-stylesheet", re.I)


def extract_xml(text: str) -> str:
    """Pull the first <scenario> document out of an LLM reply (code fences, chatter). Raises ValueError."""
    match = _DOC.search(text)
    if not match:
        raise ValueError("no <scenario> document found in the reply")
    return match.group(0)


def _fmt(value) -> str:
    return str(value)


def to_xml(config: dict) -> str:
    root = ET.Element(ROOT)
    for key, value in config.items():
        ET.SubElement(root, str(key)).text = _fmt(value)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def parse_xml(text: str) -> tuple[dict | None, list[Issue]]:
    """Return (config dict with schema types, issues). The dict is None if the document is unusable."""
    if _FORBIDDEN.search(text):
        return None, [Issue("xml_forbidden_construct", "", "DOCTYPE, ENTITY and stylesheet declarations are not allowed")]
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return None, [Issue("xml_not_well_formed", "", f"not well-formed XML: {exc}")]
    if root.tag != ROOT:
        return None, [Issue("xml_wrong_root", "", f"root element must be <{ROOT}>, found <{root.tag}>")]
    issues: list[Issue] = []
    config: dict = {}
    seen: set[str] = set()
    for child in root:
        tag = child.tag
        if tag in seen:
            issues.append(Issue("xml_duplicate_element", tag, f"element <{tag}> appears more than once"))
            continue
        seen.add(tag)
        if len(child):
            issues.append(Issue("xml_unexpected_nesting", tag, f"element <{tag}> must contain text only"))
            continue
        raw = (child.text or "").strip()
        typ = PARAMS[tag][0] if tag in PARAMS else str
        try:
            config[tag] = (int(raw) if typ is int else float(raw) if typ is float else raw)
        except ValueError:
            issues.append(Issue("wrong_type", tag, f"{tag}: expected {typ.__name__}, got {raw!r}"))
    return (config if not issues else None), issues
