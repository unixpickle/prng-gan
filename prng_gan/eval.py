import gzip
import io
import math
import random
from abc import ABC, abstractmethod

import torch

from .generator import Generator


class Eval(ABC):
    @abstractmethod
    def eval(self, gen: Generator) -> float:
        """
        Evaluate the generator and return the result.
        """


class GzipEval(Eval):
    def __init__(self, *, seq_len: int, num_bins: int = 256, num_seqs: int = 16):
        assert num_bins <= 256
        self.seq_len = seq_len
        self.num_bins = num_bins
        self.num_seqs = num_seqs

    def eval(self, gen: Generator) -> float:
        inputs = []
        for _ in range(self.num_seqs):
            maximum = 2**gen.input_bits
            start = random.randrange(maximum)
            inputs.extend((x + start) % maximum for x in range(self.seq_len))
        with torch.no_grad():
            outs = (
                (gen(inputs).view(self.num_seqs, self.seq_len) * self.num_bins)
                .floor()
                .clamp(0, self.num_bins - 1)
                .to(torch.uint8)
                .tolist()
            )
        in_bits = round(math.log2(self.num_bins) * self.seq_len * self.num_seqs)
        out_bits = 0
        for out in outs:
            data = bytes(out)
            out_bits += measure_gzip_bits(data)
        return out_bits / in_bits


def measure_gzip_bits(inputs: bytes) -> int:
    out = io.BytesIO()
    with gzip.GzipFile(mode="wb", fileobj=out, compresslevel=9) as gf:
        gf.write(inputs)
    return len(out.getvalue()) * 8
