"""Context Parallel for Gated Delta Net.

Splits the temporal sequence across GPUs and communicates only the
D x D recurrent state between ranks. At single-GPU inference time the
runtime is a no-op (cp_world_size=1).
"""

from .config import (
    CpRuntimeConfig,
    cp_enabled,
    get_cp_group,
    set_cp_group,
)
from .halo_exchange import cp_halo_exchange

__all__ = [
    "cp_enabled",
    "cp_halo_exchange",
    "CpRuntimeConfig",
    "get_cp_group",
    "set_cp_group",
]
