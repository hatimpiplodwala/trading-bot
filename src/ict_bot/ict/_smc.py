"""Internal shim for importing :mod:`smartmoneyconcepts`.

The library prints a banner containing a non-cp1252 emoji at import time, which
raises ``UnicodeEncodeError`` on Windows consoles using the cp1252 codec. We
import it with stdout redirected to an in-memory buffer, which both suppresses
the banner and sidesteps the console's encoding entirely.

All detectors import smc from here: ``from ict_bot.ict._smc import smc``.
"""

from __future__ import annotations

import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    from smartmoneyconcepts import smc

__all__ = ["smc"]
