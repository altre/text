# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Optional
import pytest
import torch
from torch import Tensor

from torchtext.models.BERT.bert_text_encoder import bert_text_encoder


@pytest.fixture(autouse=True)
def rnd():
    set_rng_seed(4)


class TestBERTTextEncoder:
    @pytest.fixture
    def encoder(self):
        return bert_text_encoder(hidden_size=3, num_attention_heads=1, dropout=0.0)

    def test_forward(self, encoder):
        input_ids = torch.randint(10, (2, 2))
        text_atts = Tensor([[1, 1], [1, 0]])
        output = encoder(input_ids, text_atts)
        expected = Tensor(
            [
                [[-0.658658, -0.754473, 1.413131], [-0.501156, -0.894687, 1.395843]],
                [[-0.148285, -1.143851, 1.292136], [0.424911, -1.380611, 0.955700]],
            ]
        )
        assert_expected(output.last_hidden_state, expected, rtol=0, atol=1e-4)


def set_rng_seed(seed):
    """Sets the seed for pytorch and numpy random number generators"""
    torch.manual_seed(seed)
    random.seed(seed)


def assert_expected(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    check_device=True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )
