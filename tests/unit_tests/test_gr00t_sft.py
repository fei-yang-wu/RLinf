# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract tests for the GR00T supervised adapter."""

import pytest
import torch
from torch.utils.data import IterableDataset
from transformers import BatchFeature

pytest.importorskip("gr00t")
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import Gr00tN17ForSFT, Gr00tSftLoader


class Samples(IterableDataset):
    def __init__(self, count):
        self.count = count

    def __iter__(self):
        yield from range(self.count)


def test_loader_has_declared_epoch_length_and_complete_batches():
    loader = Gr00tSftLoader(Samples(5), list, batch_size=2, batches_per_epoch=4)
    assert len(loader) == 4
    assert list(loader) == [[0, 1], [2, 3], [0, 1], [2, 3]]


def test_loader_rejects_empty_rank_and_invalid_epoch():
    with pytest.raises(ValueError):
        Gr00tSftLoader(Samples(1), list, 1, 0)
    with pytest.raises(RuntimeError, match="no complete batch"):
        list(Gr00tSftLoader(Samples(1), list, 2, 1))


def test_supervised_route_preserves_native_loss_and_gradient(monkeypatch):
    # Treat the vendor model as the boundary; no checkpoint download in unit tests.
    monkeypatch.setattr(
        Gr00tN1d7, "__init__", lambda self: torch.nn.Module.__init__(self)
    )
    monkeypatch.setattr(
        Gr00tN1d7, "forward", lambda self, inputs: {"loss": inputs["x"].square().mean()}
    )
    model = Gr00tN17ForSFT()
    x = torch.tensor([2.0], requires_grad=True)
    output = model(
        forward_type=ForwardType.SFT, data=BatchFeature(data={"inputs": {"x": x}})
    )
    output["loss"].backward()
    assert x.grad.item() == 4.0
    with pytest.raises(ValueError, match="supervised"):
        model(
            forward_type=ForwardType.DEFAULT,
            data=BatchFeature(data={"inputs": {"x": x}}),
        )
    with pytest.raises(ValueError, match="collator"):
        model(forward_type=ForwardType.SFT, data={"x": x})
