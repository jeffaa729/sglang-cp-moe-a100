"""Regression tests for GLM MoE gate weights used by FP32 routing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _model_load_weights_direct,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models import glm4_moe
from sglang.srt.models.glm4_moe import Glm4MoeGate, Glm4MoeSparseMoeBlock
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteGate
from sglang.test.test_utils import CustomTestCase

_CONFIG = SimpleNamespace(n_routed_experts=3, hidden_size=4)


class TestGlmMoeGateFp32Weight(CustomTestCase):
    def test_packed_shared_experts_do_not_require_weight_attribute(self):
        config = SimpleNamespace(
            hidden_size=4,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            routed_scaling_factor=1.0,
            hidden_act="silu",
            moe_intermediate_size=16,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
        )
        projection = SimpleNamespace(
            weight_packed=torch.zeros((4, 2), dtype=torch.uint8),
            quant_method=SimpleNamespace(
                quant_config=SimpleNamespace(get_name=lambda: "compressed-tensors")
            ),
        )
        shared = SimpleNamespace(gate_up_proj=projection, down_proj=projection)
        with (
            patch.object(
                glm4_moe, "get_parallel", return_value=SimpleNamespace(tp_size=8)
            ),
            patch.object(
                glm4_moe, "is_shared_experts_fusion_disabled", return_value=True
            ),
            patch.object(
                glm4_moe,
                "get_moe_impl_class",
                return_value=lambda **kwargs: SimpleNamespace(
                    should_fuse_routed_scaling_factor_in_topk=False
                ),
            ),
            patch.object(glm4_moe, "TopK", return_value=MagicMock()),
            patch.object(glm4_moe, "Glm4MoeMLP", return_value=shared),
            patch.object(
                glm4_moe, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE
            ),
            patch.object(
                glm4_moe,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            ),
            patch.object(
                glm4_moe.SboFlags, "fuse_shared_experts_inside_sbo", return_value=False
            ),
        ):
            block = Glm4MoeSparseMoeBlock(config, layer_id=0)
        self.assertFalse(block.shared_experts_is_int8)
        self.assertFalse(block.shared_experts_is_fp8)

    def test_bf16_load_updates_fp32_weight_in_place(self):
        """BF16 runtime updates must overwrite the canonical FP32 gate weight."""
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        initial = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        updated = initial + 1

        for gate_cls in (Glm4MoeGate, Glm4MoeLiteGate):
            with self.subTest(gate=gate_cls.__name__):
                with set_default_torch_dtype(torch.bfloat16):
                    gate = gate_cls(_CONFIG)

                self.assertEqual(gate.weight.dtype, torch.float32)
                self.assertFalse(hasattr(gate, "_weight_fp32"))
                weight_ptr = gate.weight.data_ptr()

                default_weight_loader(gate.weight, initial)
                torch.testing.assert_close(gate.weight, initial.float())

                _model_load_weights_direct(gate, [("weight", updated)])
                self.assertEqual(gate.weight.data_ptr(), weight_ptr)
                torch.testing.assert_close(gate.weight, updated.float())
                torch.testing.assert_close(
                    gate(hidden_states),
                    F.linear(hidden_states.float(), updated.float()),
                )


if __name__ == "__main__":
    unittest.main()
