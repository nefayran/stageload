"""Keep the random number generators where they were while a model is built.

Building a module runs its random initialisation, which advances the generator of the device the
parameters are created on. A pipeline that seeds once and then draws noise stage by stage would
get different noise if a model were built in between, and so a different result. Loading models
lazily must not do that.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def preserved_rng() -> Iterator[None]:
    """Restore the torch CPU, MPS and CUDA generators, Python's ``random`` and NumPy's global
    generator (when NumPy is already imported) after the block.

    Generators a library creates for itself, such as a ``torch.Generator`` or a
    ``numpy.random.Generator``, are not touched: nothing outside the library can see them.
    """
    cpu = torch.get_rng_state()
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    # this initialises CUDA when nothing has yet; skipping it then would lose a seed queued by
    # torch.manual_seed, which CUDA applies on initialisation, the first time a load draws from it
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if mps is not None:
            torch.mps.set_rng_state(mps)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(python)
        if numpy is not None:
            numpy.random.set_state(numpy_state)
