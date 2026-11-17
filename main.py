"""gta6-twitch-ccu-forecast-01 entry: a launch-day shape forecaster.

The idea: every big game launch draws the same daily curve on Twitch once
you scale it by its first hour. Midnight rolls around the world — New
Zealand first, then Europe, the UK, the US east coast, the US west coast —
so viewership dips in the small UTC hours, jumps at each unlock, and peaks
in the evening. The case file ships the release-day curves of dozens of
past big launches (its `comparables`); this program learns the curve from
them instead of guessing it.

For each forecast hour h = 2..24 it fits, over the donor launches, a local
linear model of the log ratio log(v_h / v_1) against the log of the launch's
own hour-1 print log(v_1), weighting each donor by how close its hour-1
print is to this case's (a Gaussian kernel, bandwidth 1.0 in log units):
big launches grow differently from small ones, so a donor the size of this
launch counts more. The fitted value at this case's hour-1 print is the
median ratio r_h; the weighted spread of the residuals around the fit is
the log spread s_h. The five quantiles are then

    q_tau(h) = p1 * r_h * exp(z_tau * s_h), rounded to 2 decimals (half up)

with p1 the case's hour-1 print and z_tau the standard normal quantiles
(-1.645, -0.674, 0, 0.674, 1.645). Slopes are used only when they are
clearly nonzero (|b| >= 2 standard errors, capped at 1.5), and both the
effective log hour-1 print and the predicted log ratio are clamped to the
donors' observed range (plus half a log unit), so an hour-1 print far
outside anything seen before cannot produce an absurd forecast.

Donors are the comparables with a complete 24-hour release day; the Grand
Theft Auto VI category's own pre-release history (no release day yet) is
not a donor. If fewer than five donors survive, the program falls back to
the persistence baseline (every hour at the first hour's count, spread
0.15 * sqrt(h - 1)).

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, standard
library only, no network): it reads every <id>.in in the input directory
($MAKEFUN_INPUT_DIR, else /input) and writes <id>.out to the output
directory ($MAKEFUN_OUTPUT_DIR, else /output). Deterministic: no random
numbers, no clock, no dict-order dependence, so two runs write the same
bytes. Locally:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Persistence fallback spread, in log units per square-root hour.
BASE_SIGMA = 0.15
# Gaussian kernel bandwidth on log hour-1 prints, in log units.
BANDWIDTH = 1.0
# Spread floors and caps, in log units.
SPREAD_MIN = 0.15
SPREAD_MAX = 1.2
# How far outside the donors' observed range a prediction may stray, log units.
EXTRAPOLATION = 0.5
# A slope is kept only when it clears this many standard errors.
SLOPE_SIGMA = 2.0
# The slope itself is capped this far either way.
SLOPE_MAX = 1.5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def release_day(comp: dict) -> dict[int, float] | None:
    """The comparable's release-day hourly counts {hour: viewers}, or None.

    A donor needs all 24 hours of its release day (day_start's date), each
    with a positive count. The pre-release Grand Theft Auto VI category has
    no release day, so it never qualifies.
    """
    day = comp.get("day_start", "")[:10]
    if not day:
        return None
    hours: dict[int, float] = {}
    for point in comp.get("hourly", []):
        start = point.get("from", "")
        if not start.startswith(day):
            continue
        hour = int(start[11:13]) + 1
        viewers = point.get("viewers", 0)
        if viewers is None or viewers <= 0:
            return None
        hours[hour] = float(viewers)
    if len(hours) != 24:
        return None
    return hours


def donors_of(case: dict) -> list[dict[int, float]]:
    """Every usable donor release-day curve in the case, in file order."""
    donors = []
    for comp in case.get("comparables", []):
        if "before release" in comp.get("name", ""):
            continue
        day = release_day(comp)
        if day is not None:
            donors.append(day)
    return donors


def weighted_fit(xs: list[float], ys: list[float], ws: list[float], x0: float) -> tuple[float, float]:
    """Weighted local-linear fit of y on x, evaluated at x0.

    Returns (prediction, residual spread). The slope is kept only when it
    clears SLOPE_SIGMA standard errors, and is capped at SLOPE_MAX.
    """
    total_w = sum(ws)
    if total_w <= 0:
        return sum(ys) / len(ys), 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / total_w
    my = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx if sxx > 0 else 0.0
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    var = sum(w * r * r for r, w in zip(resid, ws)) / total_w
    se = math.sqrt(var / sxx) if sxx > 0 else 0.0
    if abs(slope) < SLOPE_SIGMA * se:
        slope = 0.0
    slope = max(-SLOPE_MAX, min(SLOPE_MAX, slope))
    return my + slope * (x0 - mx), math.sqrt(var)


def shape_forecast(p1: float, donors: list[dict[int, float]]) -> dict[int, list[float | int]]:
    """Median ratios and spreads per hour from the donor curves."""
    log_p1s = [math.log(day[1]) for day in donors]
    lo = min(log_p1s) - EXTRAPOLATION
    hi = max(log_p1s) + EXTRAPOLATION
    target = max(lo, min(hi, math.log(max(1.0, p1))))
    out: dict[int, list[float | int]] = {}
    for h in range(2, 25):
        xs, ys, ws = [], [], []
        for day, lp in zip(donors, log_p1s):
            xs.append(lp)
            ys.append(math.log(day[h] / day[1]))
            ws.append(math.exp(-0.5 * ((lp - target) / BANDWIDTH) ** 2))
        pred, spread = weighted_fit(xs, ys, ws, target)
        # Never predict a ratio outside what the donors have shown, give or
        # take half a log unit.
        floor = min(ys) - EXTRAPOLATION
        ceil = max(ys) + EXTRAPOLATION
        pred = max(floor, min(ceil, pred))
        ratio = math.exp(pred)
        spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))
        out[h] = [round2(p1 * ratio * math.exp(z * spread)) for z in Z]
    return out


def persistence_forecast(p1: float) -> dict[int, list[float | int]]:
    """Fallback: every hour at the first hour's count, widening spread."""
    p1 = max(1.0, p1)
    return {
        h: [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))) for z in Z]
        for h in range(2, 25)
    }


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    donors = donors_of(case)
    hours = shape_forecast(p1, donors) if len(donors) >= 5 else persistence_forecast(p1)
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = hours[h]
        # Quantiles must be non-decreasing and above zero.
        fixed = []
        prev = 0.0
        for q in qs:
            q = max(q, prev, 0.01)
            fixed.append(q)
            prev = q
        rows.append({"hour": h, "q": fixed})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
