"""Fair Value Gaps (Phase 2).

Wraps ``smc.fvg(join_consecutive=True)`` and tracks mitigation across bars.
Returns ``FVG`` dataclasses: top, bottom, type, formed_at, mitigated,
mitigated_at.
"""

from __future__ import annotations
