"""Keep the random number generators where they were while a model is built.

Building a module runs its random initialisation, which advances the generator of the device the
parameters are created on. A pipeline that seeds once and then draws noise stage by stage would
get different noise if a model were built in between, and so a different result. Loading models
lazily must not do that.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def preserved_rng() -> Iterator[None]:
    """Restore the CPU, MPS and CUDA generator states after the block."""
    cpu = torch.get_rng_state()
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if mps is not None:
            torch.mps.set_rng_state(mps)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
