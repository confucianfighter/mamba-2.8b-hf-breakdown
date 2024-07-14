# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model, # for 2.8b it's 2560
        d_state=16,
        d_conv=4, # doesn't seem to change across variants
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model) # 5120
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # dims 2560 by 5120 * 2 = 2560 by 10240

        self.conv1d = nn.Conv1d( # dims 5120 by 5120 by 4
            in_channels=self.d_inner, # 5120
            out_channels=self.d_inner, # 5120
            bias=conv_bias,
            kernel_size=d_conv, # 4
            groups=self.d_inner, # 5120 -- the higher the number of groups the less columns in each kernel
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear( # Dims: 5120 by 160 + 16 * 2 = 5120 by 192
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs) # Dims: 160 by 5120

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs) # Dims: 5120 by 2560

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D) # D is 2560
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange( # xz shape: (batch, d_inner, seqlen) = (B, 10240, L)
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1) # x and z shapes: (B, d_inner, L) = (B, 5120, L)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen]) # x shape after conv1d and activation: (batch, d_inner, seqlen) = (B, 5120, L)
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)... x_dbl shape: (batch * seqlen, dt_rank + d_state * 2) = (B * L, 160 + 16 * 2)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1) # dt shape: (batch * seqlen, dt_rank) = (B * L, 160)
            # B and C shapes: (batch * seqlen, d_state) = (B * L, 16)
            dt = self.dt_proj.weight @ dt.t() # dt shape after projection: (d_inner, batch, * seqlength) = (5120, B, L)
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen) # dt shape after rearrange: (B, d_inner, L) = (B, 5120, L)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # B shape: (batch, d_state, seqlen) = (B, 16, L)
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # C shape: (batch, d_state, seqlen) = (B, 16, L)
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn( # y shape after this: (batch, d_inner, seqlen) = (B, 5120, L)
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d") # y shape after rearrange: (batch, seqlen, d_inner) = (B, L, 5120)
            out = self.out_proj(y) # out shape: (batch, seqlen, d_model) = (B, L, 2560)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # Ensure that hidden_states only contain one token (for autoregressive decoding)
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        
        # Project input hidden states to twice the inner dimension (B, 2D)
        xz = self.in_proj(hidden_states.squeeze(1))  # Shape: (batch, 2 * d_inner) = (B, 10240)
        
        # Split the projected hidden states into x and z components (each of shape (B, D))
        x, z = xz.chunk(2, dim=-1)  # Shape of x and z: (batch, d_inner) = (B, 5120)

        # Convolution step
        if causal_conv1d_update is None:
            # Roll the conv_state to the left by 1 (shifting for causal convolution)
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Shape: (B, d_inner, d_conv)
            
            # Set the last element of the conv_state to x
            conv_state[:, :, -1] = x
            
            # Perform convolution by summing over the product of conv_state and conv1d weights
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # Shape: (B, d_inner)
            
            # Add bias if applicable
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            
            # Apply activation function
            x = self.act(x).to(dtype=dtype)
        else:
            # Use optimized causal convolution update
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        # Project x to obtain dt, B, and C components (B, dt_rank + 2 * d_state)
        x_db = self.x_proj(x)  # Shape: (batch, dt_rank + 2 * d_state) = (B, 192)
        
        # Split the projection into dt, B, and C
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt shape: (batch, dt_rank) = (B, 160)
        # B and C shapes: (batch, d_state) = (B, 16)

        # Linear projection of dt
        dt = F.linear(dt, self.dt_proj.weight)  # Shape: (batch, d_inner) = (B, 5120)
        
        # Compute A matrix (exponentially decayed)
        # the result of this function is always the same after training
        # This is so the weights can be learned on a logarithmic scale. It adds stability to training process.
        A = -torch.exp(self.A_log.float())  # Shape: (d_inner, d_state) = (5120, 16)

        # Selective State-Space Model (SSM) step
        if selective_state_update is None:
            # Discretize A and B using softplus activation
            # softplus is log(1 + exp(x)). It's being used so that the value is always positive. 
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            
            # Compute dA and dB using matrix multiplication
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))  # Shape: (batch, d_inner, d_state) = (B, 5120, 16)
            dB = torch.einsum("bd,bn->bdn", dt, B)  # Shape: (batch, d_inner, d_state) = (B, 5120, 16)
            
            # Update the ssm_state
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)  # Shape: (B, d_inner, d_state) = (B, 5120, 16)
            
            # Compute the output y using einsum and adding the scaled input
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)  # Shape: (batch, d_inner) = (B, 5120)
            y = y + self.D.to(dtype) * x  # Shape: (B, 5120)
            y = y * self.act(z)  # Shape: (B, 5120)
        else:
            # Use optimized selective state update
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        # Project the output y back to the model's hidden size
        out = self.out_proj(y)  # Shape: (batch, hidden_size) = (B, 2560)
        
        # Return the output, conv_state, and ssm_state
        return out.unsqueeze(1), conv_state, ssm_state  # Output shape: (batch, 1, hidden_size) = (B, 1, 2560)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
