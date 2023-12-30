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
        start = rng.randrange(maximum)
        return [(start + i) % maximum for i in range(self.seq_len)]


class BlockSequentialInputSampler(InputSampler):
    def __init__(self, num_blocks: int, max_skip: int, num_bits: int, seq_len: int):
        super().__init__(num_bits=num_bits, seq_len=seq_len)
        self.num_blocks = num_blocks
        self.max_skip = max_skip
        assert seq_len % num_blocks == 0

    def sample_inputs(self, rng: Random) -> List[int]:
        maximum = 2**self.num_bits
        start = rng.randrange(maximum)
        block_size = self.seq_len // self.num_blocks
        results = []
        for _ in range(self.num_blocks):
            results.extend((start + i) % maximum for i in range(block_size))
            start += block_size + rng.randrange(self.max_skip)
        return results


class FlipRandomWalkInputSampler(InputSampler):
    def sample_inputs(self, rng: Random) -> List[int]:
        maximum = 2**self.num_bits
        x = rng.randrange(maximum)
        result = [x]
        seen = {x}
        while len(result) < self.seq_len:
            new_x = x
            while new_x not in seen:
                flip = 1 << rng.randrange(self.num_bits)
                new_x = new_x ^ flip
            seen.add(new_x)
            result.append(new_x)
        return result


class StridedInputSampler(InputSampler):
    def sample_inputs(self, rng: Random) -> List[int]:
        maximum = 2**self.num_bits
        start = rng.randrange(maximum)
        stride = round(2 ** (rng.random() * self.num_bits))
        results = []
        seen = set()
        while len(results) < self.seq_len:
            n = (start + len(results) * stride) % maximum
            if n in seen:
                # Avoid looping around when stride is very high.
                start = rng.randrange(maximum)
                continue
            seen.add(n)
            results.append(n)
        return results


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
        return rng.choices(self.samplers, weights=self.probs)[0].sample_inputs(rng)
