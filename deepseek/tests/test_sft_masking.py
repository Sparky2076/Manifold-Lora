"""Unit tests for Alpaca/chat SFT prompt masking."""

from __future__ import annotations

import unittest

from deepseek.utils_sft import (
    MaskedSFTCollator,
    _resolve_preset,
    alpaca_legacy_prompt_and_full,
)


class TestSFTPresetsAndAlpaca(unittest.TestCase):
    def test_resolve_preset_alpaca_full(self):
        self.assertEqual(_resolve_preset("alpaca_train_full"), ("tatsu-lab/alpaca", "train"))

    def test_alpaca_legacy_prompt_suffix(self):
        p, f = alpaca_legacy_prompt_and_full("Say hi", "", "Hello.")
        self.assertTrue(p.endswith("### Response:\n"))
        self.assertEqual(f, p + "Hello.")


class _FakeTokAlpaca:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"

    def __call__(self, text, add_special_tokens=True, truncation=False, **_kw):
        del add_special_tokens, truncation
        return {"input_ids": [1] + [ord(c) % 40 + 2 for c in text]}


class _FakeTokChat:
    chat_template = "{{messages}}"
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_tensors=None):
        del tokenize, return_tensors
        u = messages[0]["content"]
        base = [10, 20] + [min(127, ord(c)) for c in u[:16]]
        if len(messages) == 1:
            return base + ([999] if add_generation_prompt else [])
        a = messages[1]["content"]
        return base + [999] + [888] + [min(127, ord(c)) for c in a[:16]]


class TestMaskedCollator(unittest.TestCase):
    def test_masked_collator_alpaca_masks_prompt(self):
        col = MaskedSFTCollator(_FakeTokAlpaca(), max_length=512, sft_format="alpaca")
        batch = col(
            [{"kind": "structured", "text": "", "instruction": "Do X", "input": "", "output": "Y"}]
        )
        labels = batch["labels"][0]
        self.assertTrue((labels == -100).any())
        self.assertTrue((labels != -100).any())

    def test_masked_collator_chat_masks_prompt_prefix(self):
        col = MaskedSFTCollator(_FakeTokChat(), max_length=512, sft_format="chat")
        batch = col(
            [{"kind": "structured", "text": "", "instruction": "Hello", "input": "", "output": "World"}]
        )
        labs = batch["labels"][0]
        first_resp = int((labs != -100).nonzero(as_tuple=False)[0])
        self.assertTrue((labs[:first_resp] == -100).all())
        self.assertNotEqual(int(labs[first_resp].item()), -100)


if __name__ == "__main__":
    unittest.main()
