import math
import os
import pathlib
import threading
from typing import Any, Optional, Callable, Iterable, BinaryIO, IO
import torch
from numpy.random import normal
from torch import Tensor
import numpy as np
import numpy.typing as npt
import random
from einops import rearrange
from jaxtyping import Bool, Float, Int
from tqdm import tqdm

import student.bpe_trainer_sec_one as bpe_trainer_sec_one
from student.ablation_studies.pre_norm_ablation import burn_gpu, stop_gpu_burn
from student.pretokenization_example import find_chunk_boundaries
from student.sec_3.linear_class import TransformerLm
from student.sec_4.training_utils import run_cross_entropy_util, AdamW
from student.sec_5.training_loop import data_loader, save_checkpoint
from tests.adapters import get_adamw_cls

# file paths
FIXTURES_PATH = "../../../tests/fixtures"
BPE_TRAIN_INPUT_PATH = f"{FIXTURES_PATH}/tinystories_sample.txt"
INPUT_PATH = BPE_TRAIN_INPUT_PATH
ENCODED_TOKEN_PATH = "encoded_tokens.npy"
ENCODED_VAL_TOKEN_PATH = "encoded_val_tokens.npy"
CHECKPOINT_FOLDER = "checkpoints"

# ===
BATCH_SIZE = 24
CONTEXT_LENGTH = 256
ITERATIONS = 100

# ====
SAVE_CHECK_POINT_ITERATION = 25

DEVICE = "mps"

# OPTIMIZER
lr = 1e-3
betas=(0.9, 0.999)
weight_decay = 0.01

CHUNKING_NUM_PROCESSES = 20
VOCAB_LENGTH = 100000
SPECIAL_TOKENS = ["<|endoftext|>"]

# TRAINING MODE
ENCODE_CORPUS = True

# MODEL ARCCHITECTURE CONTROL
D_MODEL = 256
NUM_BLOCKS = 4
NUM_HEADS = 2
D_FF = D_MODEL * (3/8)
ROPE_THETA = 10000.0

def tokenizer_training(input_path=INPUT_PATH):
    gpu_thread = threading.Thread(target=burn_gpu)
    gpu_thread.start()
    vocab, merges = bpe_trainer_sec_one.run_train_bpe_util(
        input_path,
        VOCAB_LENGTH,
        SPECIAL_TOKENS
    )


    # TODO: Probably can save it
    tokenizer = bpe_trainer_sec_one.get_tokenizer_util(vocab, merges, SPECIAL_TOKENS)

    return tokenizer

def encode_and_save_data(tokenizer: bpe_trainer_sec_one.Tokenizer, train_test_split=0.8):
    file = open(INPUT_PATH, "rb")
    boundaries = find_chunk_boundaries(file, CHUNKING_NUM_PROCESSES, b"<|endoftext|>")

    all_tokens = []

    # taken from profs code:
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:]))):
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        # print("chunk", chunk)
        encoding = tokenizer.encode(chunk)
        # print("encoding", encoding)
        all_tokens.extend(encoding)

    token_array = np.array(all_tokens, dtype=np.int32)

    if train_test_split < 1.0:
        split_idx = int(len(token_array) * train_test_split)
        train_tokens = token_array[:split_idx]
        val_tokens = token_array[split_idx:]
        np.save(ENCODED_TOKEN_PATH, train_tokens)
        np.save(ENCODED_VAL_TOKEN_PATH, val_tokens)
    else:
        np.save(ENCODED_TOKEN_PATH, token_array)

def get_validation_loss(model, val_data, num_batches=4):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            input_tensor, target_tensor = data_loader(
                val_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE
            )
            logits = model(input_tensor)
            loss = run_cross_entropy_util(logits, target_tensor, DEVICE)
            total_loss += loss.item()

    model.train()
    avg_val_loss = total_loss / num_batches
    return avg_val_loss

def main_training_loop():
    train_data = np.load(ENCODED_TOKEN_PATH, mmap_mode='r')
    model = TransformerLm(
        vocab_size=VOCAB_LENGTH,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers= NUM_BLOCKS,
        num_heads=NUM_HEADS,
        d_ff=int(D_FF), # TODO: NEED to allow float as well
        rope_theta=ROPE_THETA,
        weights=None,
    )
    print("model initialized")
    optimizer = get_adamw_cls()(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=1e-8,
    )

    for it_id in tqdm(range(ITERATIONS)):
        input_tensor, target_tensor = data_loader(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        # print("input shape", input_tensor.shape)
        logits = model(input_tensor)
        loss = run_cross_entropy_util(logits, target_tensor, DEVICE)

        print(">> LOSS", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # checkpoint saving
        if it_id % SAVE_CHECK_POINT_ITERATION == 0:
            save_checkpoint(model, optimizer, it_id + 1, f"{CHECKPOINT_FOLDER}/checkpoint.pt")

if __name__ == '__main__':
    if ENCODE_CORPUS:
        print("training tokenizer")
        tokenizer = tokenizer_training()
        print("encoding corpus")
        encode_and_save_data(tokenizer, train_test_split=0.8)
        print("encoding complete")

    # main_training_loop()





