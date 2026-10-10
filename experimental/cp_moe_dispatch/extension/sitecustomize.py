"""Opt-in experiment loader. Keep unsupported/non-CP paths unchanged."""
import os
if os.environ.get("CP_MOE_VARIANT", "baseline") != "baseline":
    from sglang_extension import install
    install()
