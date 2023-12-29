import json
import os
from typing import Optional

import fire
import torch

from prng_gan.discriminator import TransformerDiscriminator
from prng_gan.eval import GzipEval
from prng_gan.generator import MLPGenerator
from prng_gan.input_sampler import (
    MixtureSampler,
    RandomInputSampler,
    SequentialInputSampler,
    StridedInputSampler,
)
from prng_gan.train_loop import TrainLoop


def train(
    *,
    generator_lr: float,
    discriminator_lr: float,
    batch_size: int,
    save_dir: str,
    gen_input_bits: int = 64,
    gen_n_outputs: int = 1,
    gen_d_hidden: int = 128,
    gen_depth: int = 8,
    disc_width: int = 128,
    disc_depth: int = 4,
    seq_len: int = 64,
    save_interval: int = 1000,
    microbatch: Optional[int] = None,
    min_disc_loss: Optional[float] = None,
    stats_loss_coeff: float = 0.0,
):
    args = locals()
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(args, f)
    print(f"Args: {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    loop = TrainLoop(
        generator=MLPGenerator(
            input_bits=gen_input_bits,
            n_outputs=gen_n_outputs,
            d_hidden=gen_d_hidden,
            depth=gen_depth,
            device=device,
            dtype=dtype,
        ),
        discriminator=TransformerDiscriminator(
            width=disc_width,
            depth=disc_depth,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        ),
        sampler=MixtureSampler(
            [
                (
                    1.0,
                    SequentialInputSampler(
                        num_bits=gen_input_bits, seq_len=seq_len // gen_n_outputs
                    ),
                ),
                (
                    1.0,
                    StridedInputSampler(
                        num_bits=gen_input_bits, seq_len=seq_len // gen_n_outputs
                    ),
                ),
                (
                    1.0,
                    RandomInputSampler(
                        num_bits=gen_input_bits, seq_len=seq_len // gen_n_outputs
                    ),
                ),
            ]
        ),
        evals=dict(
            gzip_u8_64=GzipEval(seq_len=64),
            gzip_u8_128=GzipEval(seq_len=128),
            gzip_u8_512=GzipEval(seq_len=512),
            gzip_u8_1024=GzipEval(seq_len=1024),
            gzip_u8_4096=GzipEval(seq_len=4096),
            gzip_binary_4096=GzipEval(seq_len=4096, num_bins=2),
        ),
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        batch_size=batch_size,
        microbatch=microbatch,
        save_interval=save_interval,
        save_dir=save_dir,
        min_disc_loss=min_disc_loss,
        stats_loss_coeff=stats_loss_coeff,
    )
    loop.load_latest()
    loop.run()


if __name__ == "__main__":
    fire.Fire(train)
