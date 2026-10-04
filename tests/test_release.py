import gc
import weakref

import pytest
import torch
from torch import nn

from stageload import ReleasedModuleError, release
from stageload.release import is_released, module_nbytes


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 8)
        self.register_buffer("scale", torch.ones(8))

    def forward(self, x):
        return self.lin(x) * self.scale


def tensors(module):
    return [*module.parameters(), *module.buffers()]


def test_release_moves_every_tensor_to_meta_and_reports_bytes():
    module = Tiny()
    expected = module_nbytes(module)
    stats = release(module, name="tiny", empty_cache=False)
    assert expected == (16 * 8 + 8 + 8) * 4
    assert stats.nbytes == expected
    assert stats.kept == 0
    assert {t.device.type for t in tensors(module)} == {"meta"}
    assert is_released(module)


def test_release_drops_the_last_reference_to_old_parameters():
    module = Tiny()
    ref = weakref.ref(module.lin.weight)
    release(module, empty_cache=False)
    gc.collect()
    assert ref() is None


def test_a_released_module_raises_a_named_error_when_called():
    module = Tiny()
    release(module, name="tiny", stage="decode", empty_cache=False)
    message = "'tiny' was released when entering stage 'decode'"
    with pytest.raises(ReleasedModuleError, match=message):
        module(torch.zeros(1, 16))


def test_storage_shared_with_a_kept_module_is_left_alone():
    a, b = Tiny(), Tiny()
    b.lin.weight = a.lin.weight
    stats = release(b, keep=[a], empty_cache=False)
    assert stats.kept == 1
    assert b.lin.weight is a.lin.weight
    assert a.lin.weight.device.type == "cpu"
    assert b.lin.bias.device.type == "meta"


def test_tied_storage_is_counted_once():
    module = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    module[1].weight = module[0].weight
    stats = release(module, empty_cache=False)
    assert stats.nbytes == 4 * 4 * 4


def test_releasing_twice_frees_nothing_the_second_time():
    module = Tiny()
    release(module, empty_cache=False)
    assert release(module, empty_cache=False).nbytes == 0
