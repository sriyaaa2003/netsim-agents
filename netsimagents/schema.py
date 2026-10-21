"""Parameter schema, defaults and the deterministic validator for radiosim configs."""
from __future__ import annotations

from dataclasses import dataclass

CARRIERS_GHZ = (0.7, 1.8, 2.6, 3.5)
ALLOWED_BANDWIDTH_MHZ = {
    0.7: (5, 10),
    1.8: (10, 15, 20),
    2.6: (10, 20),
    3.5: (20, 40, 100),
}
DEFAULT_BANDWIDTH_MHZ = {0.7: 10, 1.8: 20, 2.6: 20, 3.5: 20}
SCHEDULERS = ("round_robin", "proportional_fair", "max_cqi")
TRAFFIC = ("full_buffer", "bursty")

# name -> (type, default or None if required, (low, high) or None, description)
PARAMS = {
    "num_cells": (int, None, (1, 19), "number of base-station cells (hexagonal layout)"),
    "num_users": (int, None, (1, 500), "number of mobile users"),
    "sim_time_s": (int, None, (1, 600), "simulated time in seconds"),
    "scheduler": (str, None, None, "one of " + ", ".join(SCHEDULERS)),
    "carrier_ghz": (float, 3.5, None, "carrier frequency, one of " + ", ".join(map(str, CARRIERS_GHZ))),
    "bandwidth_mhz": (int, 20, None, "channel bandwidth; allowed values depend on the carrier (default 10 at 0.7 GHz, otherwise 20)"),
    "isd_m": (int, 500, (200, 2000), "inter-site distance in metres"),
    "tx_power_dbm": (int, 43, (30, 49), "base-station transmit power in dBm"),
    "user_speed_kmh": (float, 3.0, (0, 120), "user speed in km/h (pedestrian 3, urban vehicle 50, highway 120)"),
    "handover_hysteresis_db": (float, 3.0, (0, 10), "handover hysteresis in dB; 0 when there is only one cell"),
    "traffic": (str, "full_buffer", None, "one of " + ", ".join(TRAFFIC)),
    "seed": (int, 1, (0, 10**6), "random seed"),
}

# Resource budget enforced when the simulation starts (not by the validator):
# num_users * sim_time_s * STEPS_PER_S must not exceed MAX_USER_STEPS.
STEPS_PER_S = 10
MAX_USER_STEPS = 400_000


@dataclass(frozen=True)
class Issue:
    code: str
    param: str
    message: str

    def as_dict(self) -> dict:
        return {"code": self.code, "param": self.param, "message": self.message}


def parameter_docs() -> str:
    lines = []
    for name, (typ, default, rng, desc) in PARAMS.items():
        req = "required" if default is None else f"default {default}"
        rg = f", range {rng[0]}..{rng[1]}" if rng else ""
        lines.append(f"- {name} ({typ.__name__}, {req}{rg}): {desc}")
    bw = "; ".join(f"{c} GHz: {', '.join(map(str, b))} MHz" for c, b in ALLOWED_BANDWIDTH_MHZ.items())
    lines.append(f"Allowed bandwidth per carrier: {bw}.")
    lines.append("handover_hysteresis_db must be 0 when num_cells is 1.")
    return "\n".join(lines)


def with_defaults(config: dict) -> dict:
    out = dict(config)
    for name, (_, default, _, _) in PARAMS.items():
        if default is not None and name not in out:
            out[name] = default
    if "bandwidth_mhz" not in config and out.get("carrier_ghz") in DEFAULT_BANDWIDTH_MHZ:
        out["bandwidth_mhz"] = DEFAULT_BANDWIDTH_MHZ[out["carrier_ghz"]]
    if out.get("num_cells") == 1 and "handover_hysteresis_db" not in config:
        out["handover_hysteresis_db"] = 0.0
    return out


def _coerce(name: str, value):
    typ = PARAMS[name][0]
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid value")
    if typ is int:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, int):
            return value
        raise ValueError(f"expected an integer, got {value!r}")
    if typ is float:
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"expected a number, got {value!r}")
    if isinstance(value, str):
        return value
    raise ValueError(f"expected a string, got {value!r}")


def validate(config: dict) -> tuple[dict, list[Issue]]:
    """Return (normalised config with defaults, issues). Empty issues means valid."""
    issues: list[Issue] = []
    if not isinstance(config, dict):
        return {}, [Issue("not_an_object", "", "configuration must be a JSON object")]
    for key in config:
        if key not in PARAMS:
            issues.append(Issue("unknown_parameter", key, f"unknown parameter '{key}'"))
    cfg = with_defaults({k: v for k, v in config.items() if k in PARAMS})
    for name, (typ, default, rng, _) in PARAMS.items():
        if name not in cfg:
            issues.append(Issue("missing_parameter", name, f"required parameter '{name}' is missing"))
            continue
        try:
            cfg[name] = _coerce(name, cfg[name])
        except ValueError as exc:
            issues.append(Issue("wrong_type", name, f"{name}: {exc}"))
            continue
        if rng and not (rng[0] <= cfg[name] <= rng[1]):
            issues.append(Issue("out_of_range", name, f"{name}={cfg[name]} is outside {rng[0]}..{rng[1]}"))
    if any(i.code in ("wrong_type", "missing_parameter") for i in issues):
        return cfg, issues
    if cfg["scheduler"] not in SCHEDULERS:
        issues.append(Issue("bad_choice", "scheduler", f"scheduler must be one of {', '.join(SCHEDULERS)}"))
    if cfg["traffic"] not in TRAFFIC:
        issues.append(Issue("bad_choice", "traffic", f"traffic must be one of {', '.join(TRAFFIC)}"))
    if cfg["carrier_ghz"] not in CARRIERS_GHZ:
        issues.append(Issue("bad_choice", "carrier_ghz", f"carrier_ghz must be one of {', '.join(map(str, CARRIERS_GHZ))}"))
    else:
        allowed = ALLOWED_BANDWIDTH_MHZ[cfg["carrier_ghz"]]
        if cfg["bandwidth_mhz"] not in allowed:
            issues.append(Issue(
                "bandwidth_not_allowed_for_carrier", "bandwidth_mhz",
                f"{cfg['bandwidth_mhz']} MHz is not allowed at {cfg['carrier_ghz']} GHz; allowed: {', '.join(map(str, allowed))}"))
    if cfg["num_cells"] == 1 and cfg["handover_hysteresis_db"] != 0:
        issues.append(Issue("hysteresis_needs_multiple_cells", "handover_hysteresis_db",
                            "handover_hysteresis_db must be 0 when num_cells is 1"))
    return cfg, issues
