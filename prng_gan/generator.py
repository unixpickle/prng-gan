from typing import List, Union

import torch
import torch.nn as nn


class Generator(nn.Module):
    device: torch.device
    dtype: torch.dtype
    n_outputs: int


class MLPGenerator(Generator):
    def __init__(
        self,
        *,
        input_bits: int,
        n_outputs: int,
        d_hidden: int,
        depth: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        self.input_bits = input_bits
        self.n_outputs = n_outputs
        self.d_hidden = d_hidden
        self.depth = depth
        self.device = device
        self.dtype = dtype

        self.in_proj = nn.Linear(input_bits, d_hidden, device=device, dtype=dtype)

        layers = []
        for _ in range(depth):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(d_hidden, d_hidden, device=device, dtype=dtype))
        self.layers = nn.Sequential(*layers)

        self.out_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(d_hidden, n_outputs, device=device, dtype=dtype)
        )

    def forward(self, inputs: Union[List[int], torch.Tensor]) -> torch.Tensor:
        h = maybe_tensorize_bits(
            self.input_bits, inputs, device=self.device, dtype=self.dtype
        )
        h = self.in_proj(h)
        h = self.layers(h)
        h = self.out_proj(h)
        return torch.sigmoid(h)


def maybe_tensorize_bits(
    n_bits: int,
    inputs: Union[List[int], torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        return inputs
    bits = []
    for x in inputs:
        vec = []
        for i in range(n_bits):
            if x & (1 << i):
                vec.append(1.0)
            else:
                vec.append(0.0)
        bits.append(vec)
    return torch.tensor(bits, device=device, dtype=dtype)
