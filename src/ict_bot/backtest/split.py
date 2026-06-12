"""Train / out-of-sample split (Phase 5, Gotcha #6).

Tuning eight confluence weights on one in-sample window is itself overfitting;
the guard is holding out the most recent months untouched and judging there. The
OOS set is the verdict — never tuned against.
"""

from __future__ import annotations

import pandas as pd


def train_oos_split(
    df: pd.DataFrame, oos_months: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (in-sample train, out-of-sample) by calendar months.

    The most recent ``oos_months`` of bars become the untouched OOS set; the
    earlier bars are the in-sample tuning set. The two are disjoint and cover the
    whole frame, with train ending strictly before OOS begins.
    """
    cutoff = df.index.max() - pd.DateOffset(months=oos_months)
    train = df[df.index <= cutoff]
    oos = df[df.index > cutoff]
    return train, oos
