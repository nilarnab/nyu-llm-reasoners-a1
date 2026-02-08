import math
import os
from typing import Any, Optional, Callable, Iterable, BinaryIO, IO
import torch
from numpy.random import normal
from torch import Tensor
import numpy as np
import numpy.typing as npt
import random
from einops import rearrange
from jaxtyping import Bool, Float, Int



def nucleus_sampling():
    return None

def decode(model_output, temperature=1):
    return None


def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    original_tensor = torch.Tensor(dataset)
    length = original_tensor.shape[0]
    start_indices = np.random.randint(0, length - context_length, batch_size)

    inputs = []
    outputs = []

    for start_index in start_indices:
        inputs.append(original_tensor[start_index: start_index + context_length])
        outputs.append(original_tensor[start_index + 1: start_index + context_length + 1])

    input_tensor = torch.stack(inputs).long().to(device)
    output_tensor = torch.stack(outputs).long().to(device)

    return input_tensor, output_tensor


def save_checkpoint(
        model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]):

    state_dict = model.state_dict()

    opt_state = optimizer.state_dict()


    state_system = {
        "model_state": state_dict,
        "opt_state": opt_state,
        "it": iteration
    }

    torch.save(state_system, out)

    return None


def load_checkpoint(
        model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    src: str | os.PathLike | BinaryIO | IO[bytes]) -> int:

    system_state = torch.load(src)

    state_dict = system_state['model_state']
    opt_state = system_state['opt_state']
    iteration = system_state['it']

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(opt_state)

    return iteration


