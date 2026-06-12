"""Order blocks (Phase 2).

Wraps ``smc.ob()`` with mitigation tracking. State machine:
active -> mitigated -> breaker (when price closes through an OB). OBs are the
most discretionary ICT concept — pin the smc definition and don't second-guess
against guru charts (Gotcha #2).
"""

from __future__ import annotations
