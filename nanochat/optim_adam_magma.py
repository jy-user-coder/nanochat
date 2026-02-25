"""
Adam + Magma optimizer implementations.

Paper:
On Surprising Effectiveness of Masking Updates in Adaptive Optimizers
https://arxiv.org/pdf/2602.15322v1

Algorithm sketch used here (block-wise, with each tensor treated as one block):
1. Run dense Adam/AdamW moment updates.
2. Compute momentum-gradient cosine similarity.
3. Convert alignment to a damping factor via sigmoid(cos / temperature), EMA-smoothed.
4. Sample a Bernoulli block mask and scale the Adam update by (damping * mask).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


def _validate_group(group: dict) -> None:
    lr = group["lr"]
    beta1, beta2 = group["betas"]
    eps = group["eps"]
    survival_prob = group["survival_prob"]
    temperature = group["temperature"]
    ema_decay = group["ema_decay"]

    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"Invalid beta1: {beta1}")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"Invalid beta2: {beta2}")
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon: {eps}")
    if not 0.0 < survival_prob <= 1.0:
        raise ValueError(f"survival_prob must be in (0, 1], got: {survival_prob}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got: {temperature}")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got: {ema_decay}")
    if group["ddp_shard_min_size"] <= 0:
        raise ValueError(f"ddp_shard_min_size must be > 0, got: {group['ddp_shard_min_size']}")


def _safe_cosine_similarity(x: Tensor, y: Tensor, eps: float = 1e-12) -> Tensor:
    x_f = x.float()
    y_f = y.float()
    dot = (x_f * y_f).sum()
    denom = x_f.norm() * y_f.norm()
    cos = dot / denom.clamp_min(eps)
    return cos.clamp_(-1.0, 1.0)


def _adam_magma_update_(
    p: Tensor,
    grad: Tensor,
    state: dict,
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    decoupled_weight_decay: bool,
    magma: bool,
    survival_prob: float,
    temperature: float,
    ema_decay: float,
    unbiased: bool,
) -> None:
    if grad.is_sparse:
        raise RuntimeError("AdamMagma does not support sparse gradients")

    if not state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        state["magma_scale"] = torch.tensor(1.0, dtype=torch.float32, device=p.device)

    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    state["step"] += 1
    step = state["step"]

    beta1, beta2 = betas
    adam_grad = grad
    if (not decoupled_weight_decay) and weight_decay != 0.0:
        adam_grad = adam_grad.add(p, alpha=weight_decay)

    exp_avg.lerp_(adam_grad, 1.0 - beta1)
    exp_avg_sq.lerp_(adam_grad.square(), 1.0 - beta2)

    if decoupled_weight_decay and weight_decay != 0.0:
        p.mul_(1.0 - lr * weight_decay)

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
    update = exp_avg / denom

    if magma:
        cosine = _safe_cosine_similarity(exp_avg, adam_grad)
        align_score = torch.sigmoid(cosine / temperature)
        magma_scale = state["magma_scale"]
        magma_scale.mul_(ema_decay).add_(align_score.to(torch.float32), alpha=1.0 - ema_decay)

        mask = (torch.rand((), device=p.device) < survival_prob).to(torch.float32)
        block_scale = magma_scale * mask
        if unbiased:
            block_scale = block_scale / survival_prob
        update = update * block_scale.to(update.dtype)

    step_size = lr / bias_correction1
    p.add_(update, alpha=-step_size)


class AdamMagma(torch.optim.Optimizer):
    """
    Adam/AdamW + Magma wrapper.

    Defaults are taken from the paper's main setting for Magma:
    - survival_prob = 0.5
    - temperature = 2.0
    - ema_decay = 0.9

    Each parameter tensor is treated as one masking block.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        decoupled_weight_decay: bool = True,
        magma: bool = True,
        survival_prob: float = 0.5,
        temperature: float = 2.0,
        ema_decay: float = 0.9,
        unbiased: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            magma=magma,
            survival_prob=survival_prob,
            temperature=temperature,
            ema_decay=ema_decay,
            unbiased=unbiased,
            ddp_shard_min_size=1024,  # unused here, kept for group compatibility with DistAdamMagma
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            _validate_group(group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            _validate_group(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                _adam_magma_update_(
                    p,
                    p.grad,
                    self.state[p],
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    decoupled_weight_decay=group["decoupled_weight_decay"],
                    magma=group["magma"],
                    survival_prob=group["survival_prob"],
                    temperature=group["temperature"],
                    ema_decay=group["ema_decay"],
                    unbiased=group["unbiased"],
                )
        return loss


class DistAdamMagma(torch.optim.Optimizer):
    """
    Distributed AdamMagma with ZeRO-2 style sharded Adam states for large tensors.

    Communication pattern:
    - Small tensors: all_reduce grads, update full tensor on each rank.
    - Large tensors: reduce_scatter grads, update local shard, all_gather updated shards.

    Note: for sharded tensors, Magma alignment/damping is computed per shard (local block).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        decoupled_weight_decay: bool = True,
        magma: bool = True,
        survival_prob: float = 0.5,
        temperature: float = 2.0,
        ema_decay: float = 0.9,
        unbiased: bool = False,
        ddp_shard_min_size: int = 1024,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            magma=magma,
            survival_prob=survival_prob,
            temperature=temperature,
            ema_decay=ema_decay,
            unbiased=unbiased,
            ddp_shard_min_size=ddp_shard_min_size,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            _validate_group(group)

    def _reduce_group(self, group: dict, world_size: int) -> dict:
        param_infos = {}
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("DistAdamMagma does not support sparse gradients")

            can_shard = (
                p.numel() >= group["ddp_shard_min_size"]
                and grad.ndim > 0
                and grad.shape[0] % world_size == 0
            )
            if not can_shard:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)

        return dict(param_infos=param_infos)

    def _compute_group(self, group: dict, info: dict, gather_futures: list, rank: int, world_size: int) -> None:
        for p in group["params"]:
            pinfo = info["param_infos"].get(p)
            if pinfo is None:
                continue

            pinfo["future"].wait()
            grad_slice = pinfo["grad_slice"]
            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]

            _adam_magma_update_(
                p_slice,
                grad_slice,
                self.state[p],
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                magma=group["magma"],
                survival_prob=group["survival_prob"],
                temperature=group["temperature"],
                ema_decay=group["ema_decay"],
                unbiased=group["unbiased"],
            )

            if not pinfo["is_small"]:
                gather_future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_futures.append(gather_future)

    @torch.no_grad()
    def step(self, closure=None):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("DistAdamMagma requires torch.distributed to be initialized")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_infos: list[dict] = []
        for group in self.param_groups:
            _validate_group(group)
            reduce_infos.append(self._reduce_group(group, world_size))

        gather_futures = []
        for group, info in zip(self.param_groups, reduce_infos):
            self._compute_group(group, info, gather_futures, rank, world_size)

        for future in gather_futures:
            future.wait()

        return loss
