import json
import math
import os
import pathlib
import threading
import time
from csv import writer
from datetime import datetime
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
from multiprocessing import Pool

import student.bpe_trainer_sec_one as bpe_trainer_sec_one
from student.pretokenization_example import find_chunk_boundaries
from student.sec_3.linear_class import TransformerLm
from student.sec_4.training_utils import run_cross_entropy_util, get_lr_cosine_schedule, run_gradient_clipping_util
from student.sec_5.main_training.main_loop import BATCH_SIZE
from student.sec_5.training_loop import data_loader, save_checkpoint
from tests.adapters import get_adamw_cls


# file paths
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# ABSTRACTED
# all files are supposed to be in fixtures
INPUT_TRAIN_FILE_NAME = "TinyStoriesV2-GPT4-train.txt"
INPUT_VAL_FILE_NAME = "TinyStoriesV2-GPT4-valid.txt"
BPE_TRAIN_FILE_NAME = "TinyStoriesV2-GPT4-train.txt"


# file paths
FIXTURES_PATH = PROJECT_ROOT / "tests" / "fixtures"
INPUT_TRAIN_FILE_PATH_ABS = f"{FIXTURES_PATH}/{INPUT_TRAIN_FILE_NAME}"
INPUT_VAL_FILE_PATH_ABS = f"{FIXTURES_PATH}/{INPUT_VAL_FILE_NAME}"
BPE_TRAIN_FILE_PATH_ABS = f"{FIXTURES_PATH}/{BPE_TRAIN_FILE_NAME}"

ENCODED_TOKEN_PATH = str(SCRIPT_DIR / "encoded_tokens.npy")
ENCODED_VAL_TOKEN_PATH = str(SCRIPT_DIR / "encoded_val_tokens.npy")
CHECKPOINT_FOLDER = str(SCRIPT_DIR / "checkpoints")
LOGGER_FOLDER = str(SCRIPT_DIR / "loss_logs")
VOCAB_SAVE_FILE = str(SCRIPT_DIR / "vocab_save.json")
MERGES_SAVE_FILE = str(SCRIPT_DIR / "merges_save.txt")

# ===
# BATCH_SIZE = 128
BATCH_SIZE = 512
CONTEXT_LENGTH = 256
ITERATIONS = 2500

# ====
SAVE_CHECK_POINT_ITERATION = 50
FIND_VAL_LOSS_ITERATION=50

if torch.cuda.is_available():
    print("device set to CUDA")
    DEVICE = "cuda"
elif torch.mps.is_available():
    print("device set to MPS")
    DEVICE = "mps"
else:
    print("no gpu available, defaulting to CPU")
    DEVICE = "cpu"


# getting chekcpoint directory ready
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(LOGGER_FOLDER, exist_ok=True)

# getting session id
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_ID = str(timestamp)
print("Running SESSION:", SESSION_ID)

# OPTIMIZER
# lr = 1e-3
betas=(0.9, 0.999)
weight_decay = 0.01

CHUNKING_PROCESS_COUNT = 8
CHUNKING_NUM_PROCESSES = CHUNKING_PROCESS_COUNT * 4
VOCAB_LENGTH = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]

# TRAINING MODE
ENCODE_CORPUS = False

# MODEL ARCCHITECTURE CONTROL
D_MODEL = 512
NUM_BLOCKS = 4
NUM_HEADS = 16
# D_FF = D_MODEL * (3/8)
D_FF = 1344
ROPE_THETA = 10000.0

STOP_GPU_BURN = True
def burn_gpu():
    device = DEVICE
    print(f"Heavy dummy GPU running on {device}")

    while not STOP_GPU_BURN:
        for _ in range(10):
            x = torch.randn(2048, 2048, device=device)
            y = x @ x
            z = y * 0.5
        if device == "cuda":
            torch.cuda.synchronize()
        time.sleep(0.01)

    print("GPU burn thread stopped")


def save_vocab_json(vocab, path):
    vocab_json = {}

    for idx, token in vocab.items():
        if isinstance(token, bytes):
            token = token.decode("utf-8", errors="replace")
        vocab_json[token] = int(idx)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)


def save_merges_txt(merges, path):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            a = a.decode("utf-8", errors="replace") if isinstance(a, bytes) else a
            b = b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
            f.write(f"{a} {b}\n")

def tokenizer_training(vocab_path=VOCAB_SAVE_FILE, merge_path=MERGES_SAVE_FILE):
    print("Training BPE on", BPE_TRAIN_FILE_PATH_ABS)
    vocab, merges = bpe_trainer_sec_one.run_train_bpe_util(
        BPE_TRAIN_FILE_PATH_ABS,
        VOCAB_LENGTH,
        SPECIAL_TOKENS
    )

    # TODO: Probably can save it
    tokenizer = bpe_trainer_sec_one.get_tokenizer_util(vocab, merges, SPECIAL_TOKENS)

    save_vocab_json(vocab, vocab_path)
    save_merges_txt(merges, merge_path)


    return tokenizer


def encode_chunk(args):
    start, end, input_path, vocab, merges, special_tokens = args

    tokenizer = bpe_trainer_sec_one.get_tokenizer_util(vocab, merges, special_tokens)

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        encoding = tokenizer.encode(chunk)

    return encoding

def encode_and_save_data(
        tokenizer: bpe_trainer_sec_one.Tokenizer,
        input_path=INPUT_TRAIN_FILE_PATH_ABS,
        output_path=ENCODED_TOKEN_PATH,
        num_chunks=CHUNKING_NUM_PROCESSES,
        num_processes=CHUNKING_PROCESS_COUNT
):
    file = open(input_path, "rb")
    boundaries = find_chunk_boundaries(file, num_chunks, b"<|endoftext|>")
    file.close()

    vocab = tokenizer.vocab
    merges = tokenizer.merges
    special_tokens = SPECIAL_TOKENS

    args_list = [
        (start, end, input_path, vocab, merges, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    all_tokens = []
    with Pool(processes=num_processes) as pool:
        for encoding in tqdm(pool.imap(encode_chunk, args_list), total=len(args_list)):
            all_tokens.extend(encoding)

    token_array = np.array(all_tokens, dtype=np.int32)
    np.save(output_path, token_array)
    print("Saved at location", output_path)

def get_validation_loss(model, val_data, num_batches=1):
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

def main_training_loop(learning_rate,
                       train_encoded_token_path=ENCODED_TOKEN_PATH,
                       val_encoded_token_path=None
                       ):
    # LOGGER
    file_path = f"{LOGGER_FOLDER}/tuning_learning_rate{str(learning_rate).replace(".", "_")}_{SESSION_ID}.csv"
    if not os.path.isfile(file_path):
        # Create an empty file
        logger_file = open(file_path, 'w')
        print("file created at", file_path)
    else:
        res = input("File already exists, delete it [y,n]?")
        if res == 'y':
            os.remove(file_path)
        logger_file = open(file_path, 'w')

    logger_csv_writer = writer(logger_file)



    train_data = np.load(train_encoded_token_path, mmap_mode='r')
    val_data = np.load(val_encoded_token_path, mmap_mode='r') if val_encoded_token_path else None

    model = TransformerLm(
        vocab_size=VOCAB_LENGTH,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers= NUM_BLOCKS,
        num_heads=NUM_HEADS,
        d_ff=int(D_FF), # TODO: NEED to allow float as well
        rope_theta=ROPE_THETA,
        weights=None,
        device=DEVICE
    )
    model.to(DEVICE)
    print("model initialized")
    optimizer = get_adamw_cls()(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=1e-8,
    )

    use_cuda_amp = (DEVICE == "cuda")

    if use_cuda_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for it_id in tqdm(range(ITERATIONS)):
        input_tensor, target_tensor = data_loader(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        # print("input shape", input_tensor.shape)
        optimizer.zero_grad()

        if use_cuda_amp:
            with torch.cuda.amp.autocast():
                logits = model(input_tensor)
                loss = run_cross_entropy_util(logits, target_tensor, DEVICE)

            train_loss_val = loss.item()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            run_gradient_clipping_util(
                model.parameters(), max_l2_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(input_tensor)
            loss = run_cross_entropy_util(logits, target_tensor, DEVICE)

            train_loss_val = loss.item()

            loss.backward()
            run_gradient_clipping_util(
                model.parameters(), max_l2_norm=1.0
            )
            optimizer.step()

        # loss = run_cross_entropy_util(logits, target_tensor, DEVICE)
        #
        # train_loss_val = loss.item()
        #
        #
        # loss.backward()
        # run_gradient_clipping_util(model.parameters(), max_l2_norm=1.0)
        # optimizer.step()

        # validation loss calculation
        if val_data is not None and it_id % FIND_VAL_LOSS_ITERATION == 0:
            validation_loss = get_validation_loss(model, val_data, 1)
            print(">> LOSS", train_loss_val, validation_loss)
            logger_csv_writer.writerow(
                [it_id, train_loss_val, validation_loss]
            )
            logger_file.flush()
        else:
            logger_csv_writer.writerow(
                [it_id, train_loss_val, "_"]
            )
        # if val_data is not None:
        #     if it_id % FIND_VAL_LOSS_ITERATION == 0:
        #         print(">> LOSS", loss.item())
        #         validation_loss = get_validation_loss(model, val_data, 1)
        #         logger_csv_writer.writerow([it_id, train_loss_val, validation_loss])
        #     else:
        #         logger_csv_writer.writerow([it_id, train_loss_val, '_'])
        # else:
        #     logger_csv_writer.writerow([it_id, train_loss_val, '_'])



        # checkpoint saving
        if it_id % SAVE_CHECK_POINT_ITERATION == 0:
            save_checkpoint(model, optimizer, it_id + 1, f"{CHECKPOINT_FOLDER}/checkpoint_tuning_learning_rate_SESSION{SESSION_ID}_IT{it_id}.pt")

if __name__ == '__main__':
    if ENCODE_CORPUS:
        gpu_thread = threading.Thread(target=burn_gpu)
        gpu_thread.start()
        print("training tokenizer")
        tokenizer = tokenizer_training()
        print("encoding corpus")

        print("encoding training data", INPUT_TRAIN_FILE_PATH_ABS)
        encode_and_save_data(tokenizer, input_path=INPUT_TRAIN_FILE_PATH_ABS)

        print("encoding training data", INPUT_VAL_FILE_PATH_ABS)
        if INPUT_VAL_FILE_PATH_ABS is not None:
            encode_and_save_data(tokenizer,
                                 input_path=INPUT_VAL_FILE_PATH_ABS,
                                 output_path=ENCODED_VAL_TOKEN_PATH)
        print("encoding complete")
        stop_gpu_burn = True
        gpu_thread.join()



    learning_rate = 10**-3
    learning_rate_max = 1
    # while learning_rate <= learning_rate_max:
    #     print("LEARNING RATE", learning_rate)
    #     main_training_loop(learning_rate, val_encoded_token_path=ENCODED_VAL_TOKEN_PATH)
    #     learning_rate = learning_rate * 10


    main_training_loop(learning_rate, val_encoded_token_path=ENCODED_VAL_TOKEN_PATH)
