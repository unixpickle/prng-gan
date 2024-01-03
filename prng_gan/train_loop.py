import glob
import os
import pickle
import random
import traceback
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue
from torch.optim import AdamW

from .discriminator import Discriminator
from .eval import Eval
from .generator import Generator
from .input_sampler import InputSampler


class TrainLoop:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        sampler: InputSampler,
        *,
        evals: Optional[Dict[str, Eval]] = None,
        generator_lr: float,
        discriminator_lr: float,
        batch_size: int,
        save_interval: int,
        save_dir: str,
        microbatch: Optional[int] = None,
        min_disc_loss: Optional[float] = None,
        stats_loss_coeff: float = 0.0,
        disc_steps: Optional[int] = None,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.sampler = sampler
        self.evals = evals or {}
        self.opt = AdamW(
            [
                dict(params=generator.parameters(), lr=generator_lr),
                dict(params=discriminator.parameters(), lr=discriminator_lr),
            ],
            betas=(0.0, 0.999),  # match BigGAN
        )
        self.batch_size = batch_size
        self.microbatch = microbatch
        self.rng = random.Random(0)
        self.steps_completed = 0
        self.seqs_completed = 0
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.min_disc_loss = min_disc_loss
        self.stats_loss_coeff = stats_loss_coeff
        self.disc_steps = disc_steps

    def run(self):
        input_it = self.iterate_input_batches()
        while True:
            self.step(input_it)

    def step(self, input_it: Iterator[List[List[int]]]):
        if self.disc_steps is None:
            all_losses = self.step_models(input_it, gen=True, disc=True)
        else:
            for _ in range(self.disc_steps):
                self.step_models(input_it, gen=False, disc=True)
            all_losses = self.step_models(input_it, gen=True, disc=False)

        outputs = [f"{k}={v:.05f}" for k, v in all_losses.items()]
        outputs.append(f"seqs={self.seqs_completed}")
        print(f"step {self.steps_completed}: {' '.join(outputs)}")

        self.steps_completed += 1
        self.seqs_completed += self.batch_size
        if self.steps_completed % self.save_interval == 0:
            out_path = os.path.join(self.save_dir, f"{self.steps_completed:012}.pt")
            print(f"saving to {out_path} ...")
            self.save(out_path)

    def step_models(
        self, input_it: Iterator[List[List[int]]], gen: bool, disc: bool
    ) -> Dict[str, float]:
        for p in self.discriminator.parameters():
            if disc:
                p.grad = torch.zeros_like(p)
            else:
                p.grad = None
        for p in self.generator.parameters():
            if gen:
                p.grad = torch.zeros_like(p)
            else:
                p.grad = None

        all_gen_input = next(input_it)
        microbatch = self.microbatch or len(all_gen_input)
        all_losses = defaultdict(float)
        for i in range(0, self.batch_size, microbatch):
            gen_input = all_gen_input[i : i + microbatch]
            mb_weight = len(gen_input) / self.batch_size
            disc_input = torch.rand(
                len(gen_input),
                len(gen_input[0]) * self.generator.n_outputs,
                device=self.generator.device,
                dtype=self.generator.dtype,
            )
            losses = self.compute_losses(gen_input, disc_input)
            for k, v in losses.items():
                all_losses[k] += v.item() * mb_weight

            if disc:
                disc_grads = torch.autograd.grad(
                    losses["disc_loss"],
                    list(self.discriminator.parameters()),
                    retain_graph=True,
                )
                for p, g in zip(self.discriminator.parameters(), disc_grads):
                    p.grad.add_(g * mb_weight)

            if gen:
                gen_grads = torch.autograd.grad(
                    losses["gen_loss"], list(self.generator.parameters())
                )
                for p, g in zip(self.generator.parameters(), gen_grads):
                    p.grad.add_(g * mb_weight)

        if self.min_disc_loss and all_losses["disc_loss"] < self.min_disc_loss:
            for p in self.discriminator.parameters():
                p.grad = None

        if gen:
            for k, eval in self.evals.items():
                all_losses[k] = eval.eval(self.generator)

        self.opt.step()

        return all_losses

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
        gen_mean = gen_out.mean(-1)
        gen_var = gen_out.var(-1)
        stats_loss = (
            (gen_mean - disc_input.mean()) ** 2 + (gen_var - disc_input.var()) ** 2
        ).mean()
        return dict(
            gen_mean=gen_mean.mean(),
            gen_var=gen_var.mean(),
            disc_loss=disc_loss,
            gen_loss=gen_loss + stats_loss * self.stats_loss_coeff,
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

    def iterate_input_batches(self) -> Iterator[List[List[int]]]:
        queue = Queue(maxsize=32)
        proc = Process(
            target=dataset_worker,
            name="dataset-worker",
            args=(queue, self.sampler, self.batch_size, self.rng),
        )
        proc.start()
        try:
            while True:
                batch = queue.get()
                if "error" in batch:
                    raise RuntimeError(f"error from input sampler: {batch['error']}")
                self.rng = pickle.loads(batch["rng"])
                yield batch["batch"]
        finally:
            proc.kill()
            proc.join()
            del queue


def dataset_worker(
    queue: Queue, sampler: InputSampler, batch_size: int, rng: random.Random
):
    while True:
        try:
            batch = [sampler.sample_inputs(rng) for _ in range(batch_size)]
            queue.put(
                dict(
                    batch=batch,
                    rng=pickle.dumps(rng),
                )
            )
        except Exception as exc:
            traceback.print_exc()
            queue.put(dict(error=str(exc)))
