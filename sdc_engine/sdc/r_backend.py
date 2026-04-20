"""
R/sdcMicro availability check with TTL cache.
==============================================

Single implementation used by LOCSUPR.py and NOISE.py.  Caches the
result for _R_CHECK_TTL_SECONDS (default 300s / 5 min) so the R
subprocess is not hammered, but a mid-session install of sdcMicro is
picked up automatically after the TTL expires.

Call ``reset_r_check()`` to force an immediate re-probe (e.g. from a
Streamlit sidebar button).
"""

import os
import time
import logging
import warnings

log = logging.getLogger(__name__)

# Suppress rpy2 thread warning (harmless in Streamlit)
warnings.filterwarnings('ignore', message='R is not initialized by the main thread')

_R_CHECK_CACHE = {"result": None, "checked_at": 0.0}
_R_CHECK_TTL_SECONDS = 300  # Re-check every 5 minutes


def _check_r_available(force: bool = False) -> bool:
    """Lazily check if R/sdcMicro is available.

    Result is cached for ``_R_CHECK_TTL_SECONDS`` to avoid hammering the
    R subprocess.  Pass ``force=True`` to bypass the cache (e.g. after
    the user installs sdcMicro in a live session).  Call
    ``reset_r_check()`` to clear the cache entirely.
    """
    now = time.monotonic()
    cached = _R_CHECK_CACHE["result"]
    cached_at = _R_CHECK_CACHE["checked_at"]

    if not force and cached is not None and (now - cached_at) < _R_CHECK_TTL_SECONDS:
        return cached

    # Skip R on Streamlit Cloud
    if os.environ.get('STREAMLIT_SHARING_MODE') or os.path.exists('/mount/src'):
        _R_CHECK_CACHE["result"] = False
        _R_CHECK_CACHE["checked_at"] = now
        return False

    try:
        import rpy2.robjects as ro
        try:
            ro.r('library(sdcMicro)')
            result = True
        except (RuntimeError, ValueError) as exc:
            log.warning("[r_backend] sdcMicro library not available: %s", exc)
            result = False
    except ImportError:
        result = False

    _R_CHECK_CACHE["result"] = result
    _R_CHECK_CACHE["checked_at"] = now
    log.debug("R/sdcMicro availability check: %s", result)
    return result


def reset_r_check():
    """Clear the R availability cache.

    Call after installing/uninstalling sdcMicro so the next
    ``_check_r_available()`` call re-probes.
    """
    _R_CHECK_CACHE["result"] = None
    _R_CHECK_CACHE["checked_at"] = 0.0
