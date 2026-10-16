"""Executable predictions for Day 01. No GPU or model download required."""

import unittest

import torch

from kernels.tensor_layout import build_examples, same_storage


class TensorLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tensors = build_examples()
        self.base = self.tensors["base"]

    def test_transpose_is_a_view_with_reversed_strides(self):
        value = self.tensors["transpose"]
        self.assertEqual(value.stride(), (1, 4))
        self.assertTrue(same_storage(self.base, value))
        self.assertFalse(value.is_contiguous())
        value[2, 1] = -7
        self.assertEqual(self.base[1, 2].item(), -7)

    def test_slice_first_element_pointer_is_not_storage_pointer(self):
        value = self.tensors["offset_slice"]
        self.assertTrue(same_storage(self.base, value))
        self.assertNotEqual(value.data_ptr(), self.base.data_ptr())
        self.assertEqual(value.storage_offset(), 5)
        self.assertEqual(value[0, 0].item(), 5)

    def test_contiguous_reshape_can_share_storage(self):
        value = self.tensors["reshape_contiguous"]
        self.assertTrue(same_storage(self.base, value))
        value[0] = -1
        self.assertEqual(self.base[0, 0].item(), -1)

    def test_transpose_flatten_reshape_copies_for_this_layout(self):
        value = self.tensors["reshape_transpose"]
        torch.testing.assert_close(
            value, torch.tensor([0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.])
        )
        self.assertFalse(same_storage(self.base, value))
        value[0] = -1
        self.assertEqual(self.base[0, 0].item(), 0)

    def test_incompatible_view_raises_instead_of_copying(self):
        with self.assertRaises(RuntimeError):
            self.tensors["transpose"].view(-1)

    def test_noncontiguous_does_not_always_require_a_reshape_copy(self):
        source = self.tensors["step_slice"]
        value = self.tensors["reshape_step_slice"]
        self.assertFalse(source.is_contiguous())
        self.assertTrue(same_storage(source, value))
        self.assertEqual(value.stride(), (2,))
        torch.testing.assert_close(value, torch.arange(0, 12, 2, dtype=torch.float32))

    def test_advanced_indexing_allocates_a_copy(self):
        value = self.tensors["advanced_index"]
        self.assertFalse(same_storage(self.base, value))
        value[0, 0] = -1
        self.assertEqual(self.base[0, 0].item(), 0)

    def test_detach_is_not_a_frozen_weight_snapshot(self):
        parameter = torch.tensor([1., 2.], requires_grad=True)
        detached = parameter.detach()
        snapshot = parameter.detach().clone()
        self.assertFalse(detached.requires_grad)
        self.assertFalse(snapshot.requires_grad)
        with torch.no_grad():
            parameter.add_(10)
        torch.testing.assert_close(detached, torch.tensor([11., 12.]))
        torch.testing.assert_close(snapshot, torch.tensor([1., 2.]))
        self.assertTrue(same_storage(parameter, detached))
        self.assertFalse(same_storage(parameter, snapshot))


if __name__ == "__main__":
    unittest.main()
