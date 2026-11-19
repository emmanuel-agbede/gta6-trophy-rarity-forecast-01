"""GTA VI trophy earn-rate forecaster (make.fun: gta6-trophy-rarity-forecast-01).

Reads one case file per run from the input directory and writes, for every
trophy, five quantiles (5/25/50/75/95%) of its Exophase earn rate at the
case's target, scored by pinball loss on the logit scale.

How it works:

  * Exophase refreshes a game's counts in steps, hours apart, so the recorded
    page is a staircase: the read history is first compressed to the distinct
    page states it shows, each weighted by how long it persisted.
  * For each trophy, a weighted least-squares line of logit(rate) against
    ln(1 + hours since release) is fitted over the last 24 hours of states
    and read off at the target. A short window tracks the current regime:
    earn-rate curves bend (fast early growth, dilution as new players are
    tracked, slower growth later), and the local slope extrapolates better
    than a long average. The move is capped so a wild fit cannot run away.
  * The spread is the house random-walk scale (0.5 logit per root-day)
    combined with the fit's residual scale, times a calibration factor of
    0.85. On simulated launch-week histories (arrival growth plus per-trophy
    time-to-earn curves, refresh steps, recount jitter, weekend play and
    stale-page gaps) this beats the house persistence baseline by a wide
    margin and the house fitted-curve baseline in every scenario tried.
  * Degenerate case: if the whole history is one unchanging page (the
    launch-night practice cases look like this: Exophase has not refreshed
    once, and the tracked pool is still the pre-launch few), no trend can be
    fitted. A refresh before the target is then near-certain and will flood
    the pool with ordinary players: players tracked are projected to
    multiply while earners grow far more slowly, so rates are diluted by
    that ratio, with a wide spread. A trophy every tracked player holds
    (rate 100) is exempt: its earners grow with the pool by construction.

Deterministic, standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

WINDOW_H = 24.0        # states used by the trend fit: the last day
SIGMA_WALK = 0.5       # random-walk logit spread per root-day
RESID_FLOOR = 0.12     # smallest residual scale a fit may report
SPREAD_MULT = 0.85     # calibration factor on the combined spread
SPREAD_MIN = 0.3
SPREAD_MAX = 2.4
MAX_MOVE = 3.0         # cap on the median's logit move over the horizon
STALE_SIGMA = 0.85     # extra spread when no refresh has ever been seen
GROWTH_CAP = 5.0       # launch-pool growth prior for the degenerate case
EARNER_COUPLING = 0.15 # earners grow this fraction of the pool's growth


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, residual sd).
    Falls back to the weighted mean with slope 0 when x has no spread."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    dof = max(1.0, sw - 2.0)
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / dof
    return a, b, math.sqrt(max(var, 0.0))


def distinct_states(history):
    """Compress the read history to the distinct page states it shows, each
    at the first read that shows it; the final read is always appended so the
    current state anchors the fit at its own time."""
    states = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            states.append((parse_time(read["at"]), read["players_tracked"], tuple(read["rates"])))
            prev_key = key
    if history:
        last = history[-1]
        t = parse_time(last["at"])
        if not states or states[-1][0] != t:
            states.append((t, last["players_tracked"], tuple(last["rates"])))
    return states


def degenerate_rows(case, states, walk):
    """One unchanging page in the whole history: project the coming refresh.
    The tracked pool multiplies; earners grow by a coupled fraction; a trophy
    held by every tracked player stays put."""
    trophies = case["trophies"]
    t0, players, rates = states[-1]
    checkpoint = parse_time(case["at"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    age_h = max(0.0, (checkpoint - release).total_seconds() / 3600.0)
    span_h = max(0.0, (target - states[0][0]).total_seconds() / 3600.0)
    growth = 1.0 + GROWTH_CAP * math.exp(-age_h / 36.0) * min(1.0, span_h / 8.0)
    earners_growth = 1.0 + EARNER_COUPLING * (growth - 1.0)
    spread = SPREAD_MULT * math.sqrt(walk ** 2 + RESID_FLOOR ** 2 + STALE_SIGMA ** 2)
    rows = []
    for i, trophy in enumerate(trophies):
        rate = rates[i] if rates[i] is not None else 0.0
        if rate >= 99.9:
            median_logit, s = logit(rate), SPREAD_MIN
        else:
            earners = rate * players / 100.0
            new_earners = earners * earners_growth if earners > 0 else 0.5 * earners_growth
            median_logit = logit(clamp_rate(100.0 * new_earners / (players * growth)))
            s = spread
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(median_logit + z * s)) for z in Z]})
    return rows


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    history = case["history"]
    if not history:
        raise ValueError("no history")
    trophies = case["trophies"]
    checkpoint = parse_time(case["at"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    walk = SIGMA_WALK * math.sqrt(horizon_days)

    def x_of(t):
        hours = (t - release).total_seconds() / 3600.0
        return math.log(max(0.02, 1.0 + hours))

    x_star = x_of(target)
    states = distinct_states(history)
    if len({(p, r) for _, p, r in states}) == 1:
        return {"trophies": degenerate_rows(case, states, walk)}

    # How long each state persisted; the current state runs to the checkpoint.
    durations = []
    for j, s in enumerate(states):
        end = states[j + 1][0] if j + 1 < len(states) else checkpoint
        durations.append(max(0.05, (end - s[0]).total_seconds() / 3600.0))
    keep = [j for j, s in enumerate(states)
            if (checkpoint - s[0]).total_seconds() <= WINDOW_H * 3600.0]
    if len(keep) < 2:
        keep = list(range(len(states)))[-2:]

    last_rates = states[-1][2]
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i]
        last_logit = logit(last_rate if last_rate is not None else 0.0)
        xs, ys, ws = [], [], []
        for j in keep:
            r = states[j][2][i]
            if r is None:
                continue
            xs.append(x_of(states[j][0]))
            ys.append(logit(r))
            ws.append(durations[j])
        if len(xs) >= 2 and max(xs) > min(xs):
            a, b, resid = wls(xs, ys, ws)
            median_logit = a + b * x_star
            move = median_logit - last_logit
            if move > MAX_MOVE:
                median_logit = last_logit + MAX_MOVE
            elif move < -MAX_MOVE:
                median_logit = last_logit - MAX_MOVE
            spread = SPREAD_MULT * math.sqrt(walk ** 2 + max(resid, RESID_FLOOR) ** 2)
        else:
            median_logit = last_logit
            spread = SPREAD_MULT * math.sqrt(walk ** 2 + 0.3 ** 2)
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
        qs = [round4(inv_logit(median_logit + z * spread)) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
