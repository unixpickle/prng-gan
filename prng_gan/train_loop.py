import glob
import os
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .discriminator import Discriminator
from .generator import Generator
from .input_sampler import InputSampler


class TrainLoop:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        sampler: InputSampler,
        *,
        generator_lr: float,
        discriminator_lr: float,
        batch_size: int,
        save_interval: int,
        save_dir: str,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.sampler = sampler
        self.opt = AdamW(
            [
                dict(params=generator.parameters(), lr=generator_lr),
                dict(params=discriminator.parameters(), lr=discriminator_lr),
            ],
            betas=(0.0, 0.999),  # match BigGAN
        )
        self.batch_size = batch_size
        self.rng = random.Random(0)
        self.steps_completed = 0
        self.seqs_completed = 0
        self.save_interval = save_interval
        self.save_dir = save_dir

    def run(self):
        while True:
            self.step()

    def step(self):
        gen_input = [
            self.sampler.sample_inputs(self.rng) for _ in range(self.batch_size)
        ]
        disc_input = torch.rand(
            len(gen_input),
            len(gen_input[0]) * self.generator.n_outputs,
            device=self.generator.device,
            dtype=self.generator.dtype,
        )
        losses = self.compute_losses(gen_input, disc_input)

        disc_grads = torch.autograd.grad(
            losses["disc_loss"],
            list(self.discriminator.parameters()),
            retain_graph=True,
        )
        for p, g in zip(self.discriminator.parameters(), disc_grads):
            p.grad = g
        gen_grads = torch.autograd.grad(
            losses["gen_loss"], list(self.generator.parameters())
        )
        for p, g in zip(self.generator.parameters(), gen_grads):
            p.grad = g
        self.opt.step()

        outputs = [f"{k}={v:.05f}" for k, v in losses.items()]
        outputs.append(f"seqs={self.seqs_completed}")
        print(f"step {self.steps_completed}: {' '.join(outputs)}")

        self.steps_completed += 1
        self.seqs_completed += len(gen_input)
        if self.steps_completed % self.save_interval == 0:
            out_path = os.path.join(self.save_dir, f"{self.steps_completed:012}.pt")
            print(f"saving to {out_path} ...")
            self.save(out_path)

    def compute_losses(
        self, gen_input: List[List[int]], disc_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        gen_out = self.generator([x for y in gen_input for x in y]).reshape(
            [len(gen_input), -1]
        )
        disc_out = self.discriminator(torch.cat([gen_out, disc_input], dim=0))
        disc_targets = torch.tensor(
            [False] * len(gen_input) + [True] * len(disc_input),
            device=disc_out.device,
            dtype=disc_out.dtype,
        )
        disc_loss = F.binary_cross_entropy_with_logits(
            input=disc_out,
            target=disc_targets,
        )
        gen_loss = F.binary_cross_entropy_with_logits(
            input=-disc_out[: len(gen_out)],
            target=disc_targets[: len(gen_out)],
        )
        gen_mean = gen_out.mean()
        gen_std = gen_out.std(-1).mean()
        return dict(
            gen_mean=gen_mean,
            gen_std=gen_std,
            disc_loss=disc_loss,
            gen_loss=gen_loss,
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            torch.save(
                dict(
                    generator=self.generator.state_dict(),
                    discriminator=self.discriminator.state_dict(),
                    opt=self.opt.state_dict(),
                    steps_completed=self.steps_completed,
                    seqs_completed=self.seqs_completed,
                    rng=self.rng,
                ),
                f,
            )

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                state = torch.load(f, map_location="cpu")
            self.generator.load_state_dict(state["generator"])
            self.discriminator.load_state_dict(state["discriminator"])
            self.opt.load_state_dict(state["opt"])
            self.rng = state["rng"]
            self.steps_completed = state["steps_completed"]
            self.seqs_completed = state["seqs_completed"]

    def load_latest(self):
        paths = sorted(glob.glob(os.path.join(self.save_dir, "*.pt")))
        if len(paths):
            print(f"loading state from {paths[-1]} ...")
            self.load(paths[-1])
