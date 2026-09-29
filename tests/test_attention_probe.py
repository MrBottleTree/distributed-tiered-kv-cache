"""One CPU-only correctness check for paging, GQA, masking, and windows."""

import unittest

import torch

from attention_probe.scorer import compute_window_mass, gather_visible_keys


class AttentionProbeTest(unittest.TestCase):
    def test_window_mass(self) -> None:
        torch.manual_seed(0)
        # Logical pages are deliberately not in physical page order.
        cache = torch.zeros(2, 3, 128, 2, 4)
        logical_keys = torch.randn(300, 2, 4)
        cache[0, 2] = logical_keys[:128]
        cache[0, 0] = logical_keys[128:256]
        cache[0, 1, :44] = logical_keys[256:]
        keys = gather_visible_keys(cache, torch.tensor([2, 0, 1]), 300)
        torch.testing.assert_close(keys, logical_keys)

        # Four query heads share two KV heads. Zero logits give uniform mass.
        query = torch.zeros(1, 4, 4)
        mass = compute_window_mass(query, keys, window_size=256, scale=0.5)
        torch.testing.assert_close(mass[0, :, 0], torch.full((4,), 256 / 300))
        torch.testing.assert_close(mass[0, :, 1], torch.full((4,), 44 / 300))
        torch.testing.assert_close(mass.sum(-1), torch.ones(1, 4))

        varied_query = torch.randn(1, 4, 4)
        logits = torch.einsum(
            "hd,shd->hs", varied_query[0], keys.repeat_interleave(2, dim=1)
        ) * 0.5
        probabilities = logits.softmax(dim=-1)
        expected = torch.stack(
            (probabilities[:, :256].sum(-1), probabilities[:, 256:].sum(-1)), dim=-1
        )
        torch.testing.assert_close(
            compute_window_mass(varied_query, keys, window_size=256, scale=0.5)[0],
            expected,
        )

        # A two-row prefill must mask the future token from its first row.
        prefill = compute_window_mass(
            torch.zeros(2, 4, 4), keys, window_size=256, scale=0.5
        )
        torch.testing.assert_close(prefill[0, :, 1], torch.full((4,), 43 / 299))
        torch.testing.assert_close(prefill.sum(-1), torch.ones(2, 4))


if __name__ == "__main__":
    unittest.main()
