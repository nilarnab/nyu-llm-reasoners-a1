import math
from typing import Any, Optional, Callable, Iterable
import torch
from numpy.random import normal
from torch import Tensor
from einops import rearrange
from jaxtyping import Bool, Float, Int
from student.sec_3.linear_class import run_log_softmax_util


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "lmbda": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["b1"]
            b2 = group["b2"]
            lmbda = group["lmbda"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.

                grad = p.grad.data

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["m"] = b1 * state["m"] + (1 - b1) * grad
                state["v"] = b2 * state["v"] + (1 - b2) * (grad ** 2)

                lr_new = lr * ((1 - b2 ** t) ** 0.5) / (1 - b1 ** t)

                p.data -= lr_new * (state["m"]/ (torch.sqrt(state["v"]) + group["eps"]))
                p.data -= lr * lmbda * p.data

                state["lr"] = lr_new
                state["t"] += 1

                self.state[p] = state

def get_lr_cosine_schedule(
        it,
        max_learning_rate,
        min_learning_rate,
        warmup_iters,
        cosine_cycle_iters
):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters

    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)

    else:
        return min_learning_rate

    # return None

def run_gradient_clipping_util(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):

    total_sqr = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        # print("parameter grad", parameter.grad, "norm", parameter.grad.norm(2))
        total_sqr += parameter.grad.norm(2).item() ** 2

    l2_norm = total_sqr ** 0.5

    if l2_norm < max_l2_norm:
        return l2_norm

    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter.grad.mul_(max_l2_norm / (l2_norm + eps))

    return None


def run_cross_entropy_util(
inputs: Float[Tensor, " batch_size vocab_size"],
        targets: Int[Tensor, " batch_size"], device="cpu"
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
        loss across examples.

        Args:
            inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
                unnormalized logit of jth class for the ith example.
            targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
                Each value must be between 0 and `num_classes - 1`.

        Returns:
                    Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # TODO: Understand the math again
    probs = run_log_softmax_util(inputs, dim=-1)

    targets_expanded = rearrange(targets, '... -> ... 1')
    # TODO: learn a bit on torch.gather
    # TODO: WE CAN REMOVE THE CPU HERE FOR CUDA
    p_correct = torch.gather(probs.to("cpu"), dim=-1, index=targets_expanded.to("cpu")).to(device)
    p_correct = rearrange(p_correct, '... 1 -> ...')
    loss_per_example = -p_correct

    return loss_per_example.mean()

