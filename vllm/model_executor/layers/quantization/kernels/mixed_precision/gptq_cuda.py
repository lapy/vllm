# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq import ExllamaState
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


class GPTQCUDALinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 70


    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "GPTQ CUDA kernel only supported on CUDA"
        
        # Only support 4-bit symmetric for now (matching previous logic)
        if c.weight_type != scalar_types.uint4b8:
            return False, "GPTQ CUDA kernel only supports 4-bit weights"
        
        if c.zero_points:
            return False, "GPTQ CUDA kernel only supports symmetric quantization"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_packed = layer.weight_packed.data
        weight_scale = layer.weight_scale.data
        
        # Original logic used pack_factor=8 for 4-bit. 
        # Here we can derive it from weight_type or config.
        # Since we checked uint4b8 in can_implement, pack_factor is 8.
        pack_factor = 8 
        
        N = weight_packed.shape[0]
        K_pack = weight_packed.shape[1]
        K = K_pack * pack_factor
        
        group_size = self.config.group_size
        if group_size == -1:
            group_size = K # Channelwise
            
        # Repack qweight: transpose from [N, K/pack] to [K/pack, N]
        gptq_qweight = weight_packed.T.contiguous()  # [K/pack, N]

        # Handle g_idx - need to convert group indices to permutation array
        if self.config.has_g_idx and hasattr(layer, "weight_g_idx"):
            # Convert group indices to permutation array (same as Exllama does)
            gptq_g_idx = torch.argsort(layer.weight_g_idx.data).to(torch.int32)
        else:
            # Create empty g_idx (no reordering)
            gptq_g_idx = torch.empty(
                (0,), dtype=torch.int32, device=weight_packed.device
            )

        # CRITICAL: Shuffle the weights to match GPTQ kernel's expected bit layout
        ops.gptq_shuffle(gptq_qweight, gptq_g_idx, 4)  # 4-bit

        # Repack scales: transpose from [N, K/group] to [K/group, N]
        gptq_scales = weight_scale.T.contiguous().to(torch.float16)  # [K/group, N]

        # Create qzeros for symmetric quantization
        gptq_qzeros = torch.full(
            (K // group_size, N // pack_factor),
            0x77777777,  # Each 4-bit nibble = 7 (bias-1 for uint4b8)
            dtype=torch.int32,
            device=weight_packed.device,
        )

        # Register GPTQ-format parameters
        layer.register_parameter(
            "gptq_qweight", Parameter(gptq_qweight, requires_grad=False)
        )
        layer.register_parameter(
            "gptq_scales", Parameter(gptq_scales, requires_grad=False)
        )
        layer.register_parameter(
            "gptq_qzeros", Parameter(gptq_qzeros, requires_grad=False)
        )
        layer.register_parameter(
            "gptq_g_idx", Parameter(gptq_g_idx, requires_grad=False)
        )

        # Set exllama_state
        layer.exllama_state = ExllamaState.READY
    
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.gptq_qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1])

        output = ops.gptq_gemm(
            reshaped_x,
            layer.gptq_qweight,
            layer.gptq_qzeros,
            layer.gptq_scales,
            layer.gptq_g_idx,
            layer.exllama_state == ExllamaState.READY,
            False,  # use_v2_format
            4,  # bit (4-bit)
        )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
