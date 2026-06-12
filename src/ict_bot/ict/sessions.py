"""NY kill zones (Phase 3).

ET session windows with DST awareness, computed from UTC + current ET offset
(never hardcoded GST — Gotcha #4). Functions: ``is_silver_bullet(ts)``,
``is_ny_am(ts)``, ``is_ny_pm(ts)``, ``is_regular_hours(ts)``,
``current_kill_zone(ts)``. Pre-market is observe-only in v1 (Gotcha #9).
"""

from __future__ import annotations
