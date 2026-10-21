import pytest

from netsimagents.llm import extract_json
from netsimagents.radiosim import SimulationError, cell_positions, run
from netsimagents.schema import validate

BASE = dict(num_cells=7, num_users=30, sim_time_s=10, scheduler="round_robin")


def codes(cfg):
    return {i.code for i in validate(cfg)[1]}


def test_valid_config_gets_defaults():
    cfg, issues = validate(BASE)
    assert not issues and cfg["carrier_ghz"] == 3.5 and cfg["bandwidth_mhz"] == 20 and cfg["handover_hysteresis_db"] == 3.0


def test_default_bandwidth_follows_the_carrier():
    for ghz, bw in ((0.7, 10), (1.8, 20), (2.6, 20), (3.5, 20)):
        cfg, issues = validate({**BASE, "carrier_ghz": ghz})
        assert not issues and cfg["bandwidth_mhz"] == bw


def test_single_cell_default_hysteresis_is_zero():
    cfg, issues = validate({**BASE, "num_cells": 1})
    assert not issues and cfg["handover_hysteresis_db"] == 0.0


@pytest.mark.parametrize("bad,code", [
    ({"num_users": 0}, "out_of_range"), ({"num_cells": 20}, "out_of_range"),
    ({"scheduler": "fifo"}, "bad_choice"), ({"carrier_ghz": 5.0}, "bad_choice"),
    ({"traffic": "video"}, "bad_choice"), ({"colour": "red"}, "unknown_parameter"),
    ({"num_users": "many"}, "wrong_type"), ({"num_users": True}, "wrong_type"),
    ({"carrier_ghz": 0.7, "bandwidth_mhz": 20}, "bandwidth_not_allowed_for_carrier"),
    ({"num_cells": 1, "handover_hysteresis_db": 4}, "hysteresis_needs_multiple_cells"),
])
def test_validator_catches(bad, code):
    assert code in codes({**BASE, **bad})


def test_missing_required():
    assert "missing_parameter" in codes({"num_cells": 3})
    assert codes([1, 2]) == {"not_an_object"}


def test_cell_layout():
    assert cell_positions(1, 500).shape == (1, 2)
    p = cell_positions(7, 500)
    d = ((p[1:] - p[0]) ** 2).sum(axis=1) ** 0.5
    assert len({round(x) for x in d}) == 1 and round(d[0]) == 500


def test_sim_is_deterministic_and_seed_sensitive():
    a, b = run(BASE), run(BASE)
    assert a == b
    assert run({**BASE, "seed": 2})["mean_user_throughput_mbps"] != a["mean_user_throughput_mbps"]


def test_resource_limit_is_an_execution_error_not_a_validation_error():
    cfg = {**BASE, "num_users": 500, "sim_time_s": 600}
    assert not validate(cfg)[1]
    with pytest.raises(SimulationError, match="resource limit"):
        run(cfg)


def test_invalid_config_refused_by_simulator():
    with pytest.raises(SimulationError, match="invalid configuration"):
        run({**BASE, "scheduler": "fifo"})


def test_scheduler_tradeoff_max_cqi_is_least_fair():
    r = {s: run({**BASE, "scheduler": s, "num_users": 40, "sim_time_s": 20}) for s in ("round_robin", "proportional_fair", "max_cqi")}
    assert r["max_cqi"]["jain_fairness"] < r["round_robin"]["jain_fairness"]
    assert r["max_cqi"]["p5_user_throughput_mbps"] <= r["proportional_fair"]["p5_user_throughput_mbps"]


def test_no_handovers_with_one_cell_or_static_users():
    assert run({**BASE, "num_cells": 1, "handover_hysteresis_db": 0})["handovers"] == 0
    assert run({**BASE, "user_speed_kmh": 0})["handovers"] == 0


def test_bursty_traffic_lowers_load():
    full, burst = run(BASE), run({**BASE, "traffic": "bursty"})
    assert burst["cell_busy_fraction"] <= full["cell_busy_fraction"]


def test_higher_carrier_has_more_path_loss_at_equal_bandwidth():
    one = {**BASE, "num_cells": 1, "handover_hysteresis_db": 0, "bandwidth_mhz": 10}
    low, high = run({**one, "carrier_ghz": 0.7}), run({**one, "carrier_ghz": 1.8})
    assert low["mean_sinr_db"] > high["mean_sinr_db"]
    assert low["mean_user_throughput_mbps"] > high["mean_user_throughput_mbps"]


@pytest.mark.parametrize("text,ok", [
    ('{"a": 1}', True), ('Sure!\n```json\n{"a": {"b": 2}}\n```', True),
    ('Here you go: {"a": 1} hope it helps', True), ('{"a": 1', False), ("no json", False),
    ('{bad} then {"a": 3}', True),
])
def test_extract_json(text, ok):
    if ok:
        assert isinstance(extract_json(text), dict)
    else:
        with pytest.raises(ValueError):
            extract_json(text)
