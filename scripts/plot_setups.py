"""Visualize ICT detections for sanity-checking (Phase 2 acceptance).

Renders a SPY 15m candlestick chart with FVGs, order blocks, swing points, and
structure breaks drawn, to eyeball against TradingView (exact match not expected
— Gotcha #2). Saves an interactive Bokeh HTML file.

    uv run python scripts/plot_setups.py
    uv run python scripts/plot_setups.py --symbol SPY --timeframe 15m --bars 300
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, save

from ict_bot.config import REPO_ROOT
from ict_bot.data.store import BarStore
from ict_bot.ict._smc import smc
from ict_bot.ict.fvg import detect_fvgs
from ict_bot.ict.order_blocks import detect_order_blocks

FVG_COLOR = {"bullish": "#2e7d32", "bearish": "#c62828"}
OB_COLOR = {"bullish": "#1565c0", "bearish": "#ef6c00"}


def render(df: pd.DataFrame, symbol: str, timeframe: str, out_path: Path) -> Path:
    w = df.reset_index(drop=True)
    pos_of = {ts: i for i, ts in enumerate(df.index)}
    x = list(range(len(w)))
    last = len(w) - 1

    fig = figure(
        width=1500,
        height=750,
        title=f"{symbol} {timeframe} — ICT setups (FVG boxes, OB boxes, swings, structure)",
        x_axis_label="bar",
        y_axis_label="price",
    )

    # Candles.
    inc = w["close"] >= w["open"]
    dec = ~inc
    fig.segment(x, w["high"], x, w["low"], color="#666666")
    for mask, color in ((inc, "#26a69a"), (dec, "#ef5350")):
        src = ColumnDataSource(
            dict(x=[i for i in x if mask[i]],
                 top=w["open"][mask], bottom=w["close"][mask])
        )
        fig.vbar(x="x", width=0.7, top="top", bottom="bottom", source=src,
                 fill_color=color, line_color=color)

    # FVG boxes: span from formation to mitigation (or chart end if active).
    for f in detect_fvgs(df):
        left = pos_of[f.formed_at]
        right = pos_of[f.mitigated_at] if f.mitigated_at is not None else last
        fig.quad(left=left - 0.5, right=right + 0.5, top=f.top, bottom=f.bottom,
                 fill_color=FVG_COLOR[f.type], fill_alpha=0.12, line_alpha=0)

    # Order block boxes (highlight active/breaker more strongly).
    for ob in detect_order_blocks(df, swing_length=20):
        left = pos_of[ob.formed_at]
        right = pos_of[ob.broken_at] if ob.broken_at is not None else last
        alpha = 0.28 if ob.state != "mitigated" else 0.16
        fig.quad(left=left - 0.5, right=right + 0.5, top=ob.top, bottom=ob.bottom,
                 fill_color=OB_COLOR[ob.type], fill_alpha=alpha, line_alpha=0)

    # Swings.
    swings = smc.swing_highs_lows(w, swing_length=20)
    sh = swings[swings["HighLow"] == 1]
    sl = swings[swings["HighLow"] == -1]
    fig.scatter([i for i in sh.index], sh["Level"], marker="inverted_triangle",
                size=9, color="#b71c1c", legend_label="swing high")
    fig.scatter([i for i in sl.index], sl["Level"], marker="triangle",
                size=9, color="#1b5e20", legend_label="swing low")

    # Structure breaks (BOS/CHOCH): line from level to where it broke.
    bc = smc.bos_choch(w, swings)
    broken = bc[bc["BrokenIndex"].notna()]
    for i, row in broken.iterrows():
        fig.segment(i, row["Level"], row["BrokenIndex"], row["Level"],
                    color="#5e35b1", line_dash="dashed", line_width=2)

    fig.legend.click_policy = "hide"
    output_file(out_path, title=f"{symbol} {timeframe} ICT setups")
    save(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ICT detections for sanity check.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--bars", type=int, default=300)
    parser.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    df = BarStore(REPO_ROOT / "data" / "parquet").read_bars(args.symbol, args.timeframe)
    if df.empty:
        raise SystemExit("No data — run scripts/download_history.py first.")
    df = df.tail(args.bars)

    out_dir = REPO_ROOT / "data" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.symbol}_{args.timeframe}_setups.html"
    render(df, args.symbol, args.timeframe, out_path)

    print(f"Wrote {out_path}")
    if not args.no_open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
