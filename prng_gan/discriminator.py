import torch
import torch.nn as nn

from .transformer import Transformer


class Discriminator(nn.Module):
    device: torch.device
    dtype: torch.dtype


class TransformerDiscriminator(Discriminator):
    """
    https://github.com/openai/shap-e/blob/main/shap_e/models/generation/transformer.py
    """

    def __init__(
        self,
        *,
        width: int,
        depth: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        init_scale: float = 0.25
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.pos_emb = nn.Parameter(torch.randn((seq_len, width), device=device, dtype=dtype))
        self.input_emb = SirenEmbedding(width=width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=seq_len,
            width=width,
            layers=depth,
            heads=width // 64,
            init_scale=init_scale,
        )
        self.norm = nn.LayerNorm((width,), device=device, dtype=dtype)
        self.out_proj = nn.Linear(width, 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        """
        :param x: a sequence of shape [N x seq_len].
        :return: discriminator logits of shape [N].
        """
        h = self.input_emb(x)
        h = h + self.pos_emb
        h = self.backbone(h)
        h = self.norm(h)
        h = self.out_proj(h.mean(1))
        return h.squeeze(-1)


class SirenEmbedding(nn.Module):
    def __init__(self, *, width: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.proj = nn.Linear(1, width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        return torch.sin(self.proj(x.unsqueeze(-1)))
