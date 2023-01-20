# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Optional
import pytest
import torch

from torchtext.models.BERT.bert_text_transform import BertTextTransform



class TestBertTextTransform:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self, set_seed):
        tokenizer = BertTextTransform()
        return tokenizer

    def test_single_transform(self, utils):
        tokenizer = utils
        text = "raw text sample for testing tokenizer"
        out = tokenizer(text)
        assert_expected(
            actual=out,
            expected=torch.as_tensor(
                [101, 6315, 3793, 7099, 2005, 5604, 19204, 17629, 102]
            ),
        )

    def test_multi_transform(self, utils):
        tokenizer = utils
        text = ["raw text sample for testing tokenizer", "second shorter text"]
        out = tokenizer(text)
        assert_expected(
            actual=out,
            expected=torch.as_tensor(
                [
                    [101, 6315, 3793, 7099, 2005, 5604, 19204, 17629, 102],
                    [101, 2117, 7820, 3793, 102, 0, 0, 0, 0],
                ]
            ),
        )

# FIXME duplications
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
