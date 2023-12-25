from abc import ABC, abstractmethod
from random import Random
from typing import List, Tuple


class InputSampler(ABC):
    def __init__(self, num_bits: int, seq_len: int):
        self.num_bits = num_bits
        self.seq_len = seq_len

    @abstractmethod
    def sample_inputs(self, rng: Random) -> List[int]:
        """
        Sample a sequence of input indices of the given length.
        """


class SequentialInputSampler(InputSampler):
    def sample_inputs(self, rng: Random) -> List[int]:
        maximum = 2**self.num_bits
        start = rng.randrange(0, maximum)
        return [(start + i) % maximum for i in range(self.seq_len)]


class StridedInputSampler(InputSampler):
    def sample_inputs(self, rng: Random) -> List[int]:
        maximum = 2**self.num_bits
        start = rng.randrange(0, maximum)
        stride = round(2 ** (rng.random() * self.num_bits))
        results = []
        while len(results) < self.seq_len:
            n = (start + len(results) * stride) % maximum
            if n in results:
                # Avoid looping around when stride is very high.
                start = rng.randrange(0, maximum)
                continue
            results.append(n)


class RandomInputSampler(InputSampler):
    def sample_inputs(self, rng: Random) -> List[int]:
        maximum = 2**self.num_bits
        return [rng.randrange(maximum) for _ in range(self.seq_len)]


class MixtureSampler(InputSampler):
    def __init__(self, inner: List[Tuple[float, InputSampler]]):
        weights = [x for x, _ in inner]
        samplers = [y for _, y in inner]
        super().__init__(num_bits=samplers[0].num_bits, seq_len=samplers[0].seq_len)
        self.samplers = samplers
        self.probs = [x / sum(weights) for x in weights]

    def sample_inputs(self, rng: Random) -> List[int]:
        return rng.choices(self.samplers, weights=self.probs)[0].sample_inputs()
