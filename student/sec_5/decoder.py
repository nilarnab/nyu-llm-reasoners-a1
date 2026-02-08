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

from student.sec_3.linear_class import run_softmax_util, TransformerLm
from student.sec_5.main_training.main_loop import tokenizer_training

# ===========
FIXTURES_PATH = "../../../tests/fixtures"
BPE_TRAIN_INPUT_PATH = f"{FIXTURES_PATH}/tinystories_sample.txt"
INPUT_PATH = BPE_TRAIN_INPUT_PATH
ENCODED_TOKEN_PATH = "encoded_tokens.npy"
CHECKPOINT_FOLDER = "checkpoints"

# ===
BATCH_SIZE = 24
CONTEXT_LENGTH = 256
ITERATIONS = 100

# ====
SAVE_CHECK_POINT_ITERATION = 25

DEVICE = "cpu"

# OPTIMIZER
lr = 1e-3
betas=(0.9, 0.999)
weight_decay = 0.01

CHUNKING_NUM_PROCESSES = 20
VOCAB_LENGTH = 100000
SPECIAL_TOKENS = ["<|endoftext|>"]

# TRAINING MODE
ENCODE_CORPUS = False

# MODEL ARCCHITECTURE CONTROL
D_MODEL = 256
NUM_BLOCKS = 4
NUM_HEADS = 2
D_FF = D_MODEL * (3/8)
ROPE_THETA = 10000.0
# =========

def nucleus_sampling():
    return None

def decode(
        model,
        prompt = "",
        tokenizer=None,
        max_new_tokens=256,
        device="cpu",
        temperature=1):

    model.eval()
    input_ids = tokenizer.encode(prompt)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for new_token_id in range(max_new_tokens):
            outp = model(input_tensor)
            next_tok = outp[0, -1, :]

            if temperature != 1.0:
                next_tok = next_tok / temperature

            eos_token = tokenizer.encode("<|endoftext|>")[0]

            softmax_val = run_softmax_util(next_tok, dim=-1)
            next_token_id = torch.argmax(softmax_val, dim=-1)
            next_token_id = rearrange(next_token_id, '-> 1 1')

            input_tensor = torch.cat([input_tensor, next_token_id], dim=1)

            if next_token_id.item() == eos_token:
                break

    generated_ids = input_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


if __name__ == "__main__":
    tokenizer = tokenizer_training()
    model = TransformerLm(
        vocab_size=len(tokenizer.vocab),
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers= NUM_BLOCKS,
        num_heads=NUM_HEADS,
        d_ff=int(D_FF), # TODO: NEED to allow float as well
        rope_theta=ROPE_THETA,
        weights=None,
    )

    print(len(tokenizer.vocab))

    generated_text = decode(model,
                            prompt="Hello how are you?",
                            tokenizer=tokenizer,
                            )

    print("generated text", generated_text)