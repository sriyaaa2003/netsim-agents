"""A small downlink radio-network simulator (hexagonal cells, path loss, SINR, scheduling).

It is deliberately simple: the point of the project is the configuration workflow around it,
not radio-planning accuracy. Everything is seeded and deterministic.
"""
from __future__ import annotations

import math

import numpy as np

from .schema import MAX_USER_STEPS, STEPS_PER_S, validate

DT = 1.0 / STEPS_PER_S
NOISE_FIGURE_DB = 7.0
MAX_SPECTRAL_EFF = 4.8  # bit/s/Hz cap
IMPLEMENTATION_LOSS = 0.6


class SimulationError(Exception):
    """Raised for configurations that validate but cannot be executed."""


def cell_positions(num_cells: int, isd_m: float) -> np.ndarray:
    coords = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            coords.append((q, r))
    coords.sort(key=lambda qr: (max(abs(qr[0]), abs(qr[1]), abs(-qr[0] - qr[1])), math.atan2(qr[1] * math.sqrt(3) / 2, qr[0] + qr[1] / 2)))
    pts = [(isd_m * (q + r / 2), isd_m * r * math.sqrt(3) / 2) for q, r in coords[:num_cells]]
    return np.array(pts, dtype=float)


def run(config: dict) -> dict:
    cfg, issues = validate(config)
    if issues:
        raise SimulationError("invalid configuration: " + "; ".join(i.message for i in issues))
    n_users, n_cells, t_s = cfg["num_users"], cfg["num_cells"], cfg["sim_time_s"]
    if n_users * t_s * STEPS_PER_S > MAX_USER_STEPS:
        raise SimulationError(
            f"resource limit exceeded: num_users * sim_time_s * {STEPS_PER_S} = {n_users * t_s * STEPS_PER_S} "
            f"exceeds the budget of {MAX_USER_STEPS}")
    rng = np.random.default_rng(cfg["seed"])
    cells = cell_positions(n_cells, cfg["isd_m"])
    centre = cells.mean(axis=0)
    radius = float(np.max(np.linalg.norm(cells - centre, axis=1))) + cfg["isd_m"] / 2
    ang = rng.uniform(0, 2 * np.pi, n_users)
    rad = radius * np.sqrt(rng.uniform(0, 1, n_users))
    pos = centre + np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    heading = rng.uniform(0, 2 * np.pi, n_users)
    speed_ms = cfg["user_speed_kmh"] / 3.6
    shadow = rng.normal(0, 6.0, (n_users, n_cells))
    bw_hz = cfg["bandwidth_mhz"] * 1e6
    noise_mw = 10 ** ((-174 + 10 * math.log10(bw_hz) + NOISE_FIGURE_DB) / 10)
    fc_term = 32.4 + 20 * math.log10(cfg["carrier_ghz"])
    steps = t_s * STEPS_PER_S
    hys = cfg["handover_hysteresis_db"]
    bursty = cfg["traffic"] == "bursty"
    sched = cfg["scheduler"]

    def rx_dbm(p: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(p[:, None, :] - cells[None, :, :], axis=2)
        return cfg["tx_power_dbm"] - (fc_term + 35.0 * np.log10(np.maximum(d, 10.0))) - shadow

    serving = np.argmax(rx_dbm(pos), axis=1)
    bits = np.zeros(n_users)
    avg_rate = np.full(n_users, 1.0)
    handovers = 0
    sinr_acc = np.zeros(n_users)
    busy_acc = 0.0
    rr_counter = np.zeros(n_cells, dtype=int)
    for _ in range(steps):
        if speed_ms > 0:
            pos = pos + speed_ms * DT * np.stack([np.cos(heading), np.sin(heading)], axis=1)
            off = np.linalg.norm(pos - centre, axis=1) > radius
            heading = np.where(off, heading + np.pi, heading)
        rx = rx_dbm(pos)
        if n_cells > 1:
            best = np.argmax(rx, axis=1)
            cur = rx[np.arange(n_users), serving]
            move = rx[np.arange(n_users), best] > cur + hys
            handovers += int(move.sum())
            serving = np.where(move, best, serving)
        rx_mw = 10 ** (rx / 10)
        sig = rx_mw[np.arange(n_users), serving]
        interf = rx_mw.sum(axis=1) - sig
        sinr = sig / (interf + noise_mw)
        sinr_acc += 10 * np.log10(sinr)
        se = np.minimum(IMPLEMENTATION_LOSS * np.log2(1 + sinr), MAX_SPECTRAL_EFF)
        rate = se * bw_hz
        active = rng.uniform(0, 1, n_users) < 0.3 if bursty else np.ones(n_users, dtype=bool)
        got = np.zeros(n_users)
        for c in range(n_cells):
            idx = np.flatnonzero((serving == c) & active)
            if idx.size == 0:
                continue
            busy_acc += 1
            if sched == "round_robin":
                got[idx] = rate[idx] / idx.size
            elif sched == "max_cqi":
                got[idx[np.argmax(sinr[idx])]] = rate[idx[np.argmax(sinr[idx])]]
            else:
                w = rate[idx] / avg_rate[idx]
                pick = idx[np.argmax(w)]
                got[pick] = rate[pick]
        bits += got * DT
        avg_rate = 0.99 * avg_rate + 0.01 * np.maximum(got, 1e-3)
    thr = bits / t_s / 1e6
    jain = float(thr.sum() ** 2 / (n_users * np.sum(thr ** 2))) if np.sum(thr ** 2) > 0 else 0.0
    return {
        "mean_user_throughput_mbps": float(thr.mean()),
        "p5_user_throughput_mbps": float(np.percentile(thr, 5)),
        "jain_fairness": jain,
        "mean_sinr_db": float(sinr_acc.sum() / (n_users * steps)),
        "handovers": handovers,
        "cell_busy_fraction": float(busy_acc / (n_cells * steps)),
        "config": cfg,
    }
