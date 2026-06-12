"""Per-confluence edge attribution (ICT deep-dive).

Runs the SAME engine/harness but with the confluence threshold dropped to 0, so
nearly every directional setup with a valid liquidity target becomes a trade —
a large sample to measure which confluences actually carry edge. Each trade is
tagged with the confluences present; we then report, per confluence, the realized
edge (win rate, avg R) when it is present vs. absent, on BOTH the in-sample and
out-of-sample windows.

Read it as: a confluence with a real edge should show a positive avg-R *lift*
(present minus absent) that survives out-of-sample. One that doesn't is noise.

Caveat: the single-position constraint means setups overlapping an open trade are
not sampled, so this is directional evidence, not a clean factor model.

Run: ``uv run python scripts/attribute_confluences.py``
"""

from __future__ import annotations

import copy
from dataclasses import fields

import pandas as pd

from ict_bot.backtest.runner import run_backtest
from ict_bot.backtest.split import train_oos_split
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore
from ict_bot.strategy.confluence import Confluences

CONFLUENCE_FIELDS = [f.name for f in fields(Confluences)]


def _trades_frame(stats: pd.Series) -> pd.DataFrame:
    tr = stats._trades.copy()
    if len(tr) == 0:
        return tr
    tr["R"] = tr["PnL"] / ((tr["EntryPrice"] - tr["SL"]).abs() * tr["Size"].abs())
    for name in CONFLUENCE_FIELDS:
        tr[name] = [bool(getattr(tag, name)) for tag in tr["Tag"]]
    return tr


def _attribution(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in CONFLUENCE_FIELDS:
        present, absent = tr[tr[name]], tr[~tr[name]]
        rows.append(
            {
                "confluence": name,
                "n_present": len(present),
                "win%_present": round(100 * (present["PnL"] > 0).mean(), 1) if len(present) else float("nan"),
                "avgR_present": round(present["R"].mean(), 3) if len(present) else float("nan"),
                "avgR_absent": round(absent["R"].mean(), 3) if len(absent) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out["lift"] = (out["avgR_present"] - out["avgR_absent"]).round(3)
    return out.sort_values("lift", ascending=False)


def main() -> None:
    settings = load_settings()
    probe = copy.deepcopy(settings)
    probe["signal"]["min_confluence_score"] = 0  # take (almost) every directional setup

    store = BarStore(REPO_ROOT / "data" / "parquet")
    tf = settings["entry_timeframe"]
    entry = store.read_bars(settings["symbol"], tf)
    daily = store.read_bars(settings["symbol"], "1d")
    h1 = store.read_bars(settings["symbol"], "1h")
    refs = {r: store.read_bars(r, tf) for r in settings["smt_reference_symbols"]}
    train, oos = train_oos_split(entry, oos_months=settings["backtest"]["oos_months"])

    for label, seg in (("IN-SAMPLE", train), ("OUT-OF-SAMPLE", oos)):
        stats = run_backtest(seg, daily, h1, refs, probe, cash=100_000.0)
        tr = _trades_frame(stats)
        print(f"\n===== {label}  ({len(tr)} trades, overall avgR={tr['R'].mean():.3f}) =====")
        if len(tr):
            print(_attribution(tr).to_string(index=False))


if __name__ == "__main__":
    main()
