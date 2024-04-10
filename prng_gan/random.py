import numpy as np
import torch

PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85
PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57


class StepRNG:
    def __init__(self, seed: int, device: torch.device, dtype: torch.dtype):
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.offset = 0

    def rand(self, *shape: int) -> torch.Tensor:
        n = int(np.prod(shape))
        res = random_floats(
            seed=self.seed,
            start=self.offset,
            length=n,
            device=self.device,
            dtype=self.dtype,
        )
        self.offset += n
        return res


def random_floats(
    seed: int,
    start: int,
    length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    rounds: int = 20,
) -> torch.Tensor:
    """
    Generate a chunk of random floats with the Philox RNG.

    Defaults to an aggressively high number of rounds to ensure that the
    discriminator cannot infer RNG behavior.
    """
    # Based on https://github.com/openai/triton/blob/bf5346b9dbb458f103f8a5009906597f0454761c/python/triton/language/random.py#L13
    rounded_start = (start // 4) * 4
    num_items = start + length - rounded_start
    num_items = num_items // 4 + int(num_items % 4 != 0)

    c0 = torch.arange(
        rounded_start // 4,
        rounded_start // 4 + num_items,
        1,
        device=device,
        dtype=torch.int32,
    )
    c1, c2, c3 = [c0 * 0] * 3
    k0 = seed & ((1 << 32) - 1)
    k1 = seed >> 32

    for _ in range(rounds):
        prev_c0, prev_c2 = c0, c2
        c0 = ((PHILOX_ROUND_B * c2.long()) >> 32).int() ^ c1 ^ k0
        c2 = ((PHILOX_ROUND_A * prev_c0.long()) >> 32).int() ^ c3 ^ k1
        c1 = PHILOX_ROUND_B * prev_c2
        c3 = PHILOX_ROUND_A * prev_c0
        k0 = (k0 + PHILOX_KEY_A) & ((1 << 32) - 1)
        k1 = (k1 + PHILOX_KEY_B) & ((1 << 32) - 1)

    return (
        torch.stack([c0, c1, c2, c3], dim=1)
        .flatten()[start - rounded_start :][:length]
        .to(dtype)
        / (2**32)
        + 0.5
    )
