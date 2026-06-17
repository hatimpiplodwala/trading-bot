"""OHLC storage (Phase 1).

DuckDB-backed reads over Parquet partitioned by ``symbol/timeframe/YYYY-MM``.
Bars are indexed by OPEN (start) time in UTC. ``append_bars`` upserts by
timestamp (re-downloading overlapping ranges is safe); ``read_bars`` returns a
half-open ``[start, end)`` range, sorted ascending.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]
_INDEX = "timestamp"


class BarStore:
    """Parquet bar store with DuckDB-powered range reads."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # --- paths -----------------------------------------------------------
    def _series_dir(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol / timeframe

    def _partition_path(self, symbol: str, timeframe: str, year_month: str) -> Path:
        return self._series_dir(symbol, timeframe) / f"{year_month}.parquet"

    # --- write -----------------------------------------------------------
    def append_bars(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Upsert OHLCV bars, partitioning per year-month. Newer rows win."""
        if df.empty:
            return
        if df.index.tz is None:
            raise ValueError("append_bars requires a tz-aware (UTC) index")

        df = df[COLUMNS].sort_index()
        series_dir = self._series_dir(symbol, timeframe)
        series_dir.mkdir(parents=True, exist_ok=True)

        for year_month, chunk in df.groupby(df.index.strftime("%Y-%m")):
            path = self._partition_path(symbol, timeframe, year_month)
            if path.exists():
                existing = pd.read_parquet(path)
                chunk = pd.concat([existing, chunk])
                chunk = chunk[~chunk.index.duplicated(keep="last")].sort_index()
            chunk.index.name = _INDEX
            chunk.to_parquet(path, engine="pyarrow")

    # --- read ------------------------------------------------------------
    def read_bars(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Read bars in ``[start, end)``; both bounds optional."""
        empty = pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name=_INDEX))
        files = sorted(self._series_dir(symbol, timeframe).glob("*.parquet"))
        if not files:
            return empty

        glob = str(self._series_dir(symbol, timeframe) / "*.parquet")
        clauses = []
        params: list[object] = []
        if start is not None:
            clauses.append(f"{_INDEX} >= ?")
            params.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            clauses.append(f"{_INDEX} < ?")
            params.append(pd.Timestamp(end).to_pydatetime())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        con = duckdb.connect()
        try:
            con.execute("SET TimeZone='UTC'")
            query = (
                f"SELECT {_INDEX}, {', '.join(COLUMNS)} "
                f"FROM read_parquet(?) {where} ORDER BY {_INDEX}"
            )
            out = con.execute(query, [glob, *params]).df()
        finally:
            con.close()

        out[_INDEX] = pd.to_datetime(out[_INDEX], utc=True)
        return out.set_index(_INDEX)[COLUMNS]
