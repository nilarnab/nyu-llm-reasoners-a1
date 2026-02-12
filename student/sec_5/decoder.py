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

from student.bpe_trainer_sec_one import load_bpe_model, Tokenizer
from student.sec_3.linear_class import run_softmax_util, TransformerLm
from student.sec_5.training_loop import load_checkpoint


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


BPE_CORPUS_PATH = "/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/tests/fixtures/tinystories_sample.txt"
CHECK_POINT_PATH = "/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/student/ablation_studies/checkpoints/checkpoint_tuning_learning_rate_SESSION20260211_185330_IT41.pt"

VOCAB_PATH = "/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/student/ablation_studies/vocab_save.json"
MERGES_PATH = "/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/student/ablation_studies/merges_save.txt"

def nucleus_sampling(input, top_p=0):
    sorted_probs, sorted_indices = torch.sort(input, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_index = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
    if len(cutoff_index) > 0:
        cutoff_index = cutoff_index[0].item()
    else:
        cutoff_index = len(sorted_probs)

    sorted_probs = sorted_probs[:cutoff_index + 1]
    sorted_indexes = sorted_indices[:cutoff_index + 1]

    sorted_probs = sorted_probs / sorted_probs.sum()

    sampled_index = torch.multinomial(sorted_probs, num_samples=1)
    token_id = sorted_indexes[sampled_index]

    return token_id


def decode(
        model,
        prompt = "",
        tokenizer=None,
        max_new_tokens=256,
        device="cpu",
        temperature=1,
        top_p=None):

    print("generating for: ", prompt)
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

            if top_p is not None and top_p < 1.0:
                next_token_id = nucleus_sampling(next_tok, top_p=top_p)
            else:
                softmax_val = run_softmax_util(next_tok, dim=-1)
                next_token_id = torch.argmax(softmax_val, dim=-1)

            if next_token_id.dim() == 0:
                next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)
            else:
                next_token_id = rearrange(next_token_id, 'n -> 1 1')

            input_tensor = torch.cat([input_tensor, next_token_id], dim=1)

            if next_token_id.item() == eos_token:
                break

    generated_ids = input_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text



def get_prompt_output(
        tokenizer,
        prompt,
):
    model = TransformerLm(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,  # TODO: NEED to allow float as well
        rope_theta=100000,
        weights=None,
    )

    model = torch.compile(model)


    load_checkpoint(model, None, src=CHECK_POINT_PATH)

    # print(len(tokenizer.vocab))

    generated_text = decode(model,
                            prompt=prompt,
                            tokenizer=tokenizer,
                            )

    print(generated_text)



if __name__ == "__main__":
    vocab, merges = load_bpe_model(VOCAB_PATH, MERGES_PATH)
    tokenizer = Tokenizer(vocab, merges, ['<|endoftext|>'])
    print("training complete")

    get_prompt_output(tokenizer, "Hello, how are you doing?")

    get_prompt_output(tokenizer, "When I die doing grad work, I want")

    get_prompt_output(tokenizer, "I hate there is so much ice everywhere that")

    get_prompt_output(tokenizer, "death is a")
