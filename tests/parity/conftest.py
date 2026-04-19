"""Parity test configuration. Skips entire suite if rpy2 or sdcMicro unavailable."""
import pytest

try:
    from sdc_engine.sdc.r_backend import _check_r_available
    R_AVAILABLE = _check_r_available()
except Exception:
    R_AVAILABLE = False

skip_if_no_r = pytest.mark.skipif(
    not R_AVAILABLE,
    reason="R + sdcMicro not available — parity tests require both",
)
