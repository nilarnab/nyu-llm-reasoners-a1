import json
import os
import time
from multiprocessing import Pool
from typing import Iterable, Iterator

import regex as re
from collections import defaultdict

from student.pretokenization_example import find_chunk_boundaries
from functools import lru_cache

import cProfile
from memory_profiler import memory_usage
import pstats

# from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Answer to problem (tokenizer)
# ===========
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        if special_tokens is None:
            special_tokens = []
        self.vocab = vocab
        self.reverse_vocab = {}
        self.merges = merges
        self.special_tokens = special_tokens
        self._make_reverse_vocab()

        self.merge_ranks = {
            merge: i for i, merge in enumerate(self.merges)
        }

        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_regex = re.compile(
                '(' + '|'.join(re.escape(st) for st in self.special_tokens) + ')'
            )
        else:
            self.special_regex = None

    def _bpe_merge(self, token):
        token = list(token)

        while True:
            best_rank = None
            best_pair = None

            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_token = []
            i = 0
            while i < len(token):
                if i + 1 < len(token) and (token[i], token[i + 1]) == best_pair:
                    new_token.append(token[i] + token[i + 1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1

            token = new_token

        return tuple(token)

    def _make_reverse_vocab(self):
        for key in self.vocab:
            self.reverse_vocab[self.vocab[key]] = key

    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text_raw):
        # for special_token in self.special_tokens:
        #     text = text.replace(special_token, "")
        print("got", text_raw)
        pattern_special = None
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=lambda s: len(s), reverse=True)

            pattern_special = '(' + '|'.join(re.escape(st) for st in self.special_tokens) + ')'

        if pattern_special is not None:
            text_chunks = re.split(pattern_special, text_raw)
        else:
            text_chunks = [text_raw]

        print("text_chunks", text_chunks)

        res = []
        for text in text_chunks:
            if text in self.special_tokens:
                print("textrying to get", text.encode("utf-8"))
                print("got", self._token_to_id(text.encode("utf-8")))
                res.append(self._token_to_id(text.encode("utf-8")))
                pass
            elif text:
                pre_tokens = []
                for token in re.finditer(PAT, text):
                    print("while enoding, getting token", token.group().encode("utf-8"))

                    pre_tokens.append(tuple(bytes([b]) for b in token.group().encode("utf-8")))

                print("pre tokens", pre_tokens)

                new_pre_tokens = []
                for token_raw in pre_tokens:
                    merged = self._bpe_cache(token_raw)
                    new_pre_tokens.append(merged)

                print("new pre tokens:", new_pre_tokens)

                for token in new_pre_tokens:
                    # print("token in new pre token", token)
                    for sub_token in token:
                        # print("sub token", sub_token)
                        token_id = self._token_to_id(sub_token)
                        res.append(token_id)

        return res

    @lru_cache(maxsize=100_000)
    def _bpe_cache(self, token):
        return self._bpe_merge(token)

    def decode(self, ids: list[int]):
        res = ""
        buffer = b''
        for id_val in ids:
            if id_val in self.vocab:
                token_byte = self.vocab[id_val]
                buffer += token_byte
                try:
                    res += buffer.decode("utf-8")
                    buffer = b''
                except UnicodeDecodeError:
                    pass

            else:
                print("unknown id", id_val)
                res += "<?>"

        return res

    def encode_iterable(self, iterable: Iterable[int]) -> Iterator[int]:
        # print("in encode iterable funtion")
        res = []
        for text in iterable:
            encoding = self.encode(text)
            res.extend(encoding)

        # print(res)
        for token_id in res:
            yield token_id


    def _token_to_id(self, token):
        if token in self.reverse_vocab:
            return self.reverse_vocab[token]
        # for token_id in self.vocab:
        #     if token == self.vocab[token_id]:
        #         return token_id
        # print("problem with token", token)
        print(f" MISSING TOKEN: {token}")
        print(f"   Token length: {len(token)} bytes")
        print(f"   First 10 vocab tokens: {list(self.reverse_vocab.keys())[:10]}")
        raise Exception(f"No token id found for given token: {token} in the vocab")

def get_tokenizer_util(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
):
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    return tokenizer

# ===========



# Answer to Problem (train_bpe)
# =================

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {}
    counter = 0
    for i in range(len(special_tokens)):
        vocab[counter] = special_tokens[i].encode("utf-8")
        counter += 1

    for i in range(256):
        vocab[counter] = bytes([i])
        counter += 1

    return vocab


# pre tokens
def pre_token_chunk(args):
    start, end, input_path, special_tokens = args
    frequency_table = defaultdict(int)

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = '|'.join(escaped_tokens)
        text_segments = re.split(split_pattern, chunk)
    else:
        text_segments = [chunk]

    for segment in text_segments:
        if not segment:
            continue
        for token in re.finditer(PAT, segment):
            token_bytes_tuple = tuple(bytes([b]) for b in token.group().encode("utf-8"))
            frequency_table[token_bytes_tuple] += 1

    return frequency_table


def merge_frequency_tables(tables):
    result = defaultdict(int)
    for table in tables:
        for key, count in table.items():
            result[key] += count
    return result


def get_pre_tokens(input_path, special_tokens, num_processes=8):
    frequency_table = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes*4, b"<|endoftext|>")

    args_list = [(start, end, input_path, special_tokens)
                 for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(processes=num_processes) as pool:
        results = pool.map(pre_token_chunk, args_list)

    # merge frequency tables
    frequency_table = merge_frequency_tables(results)
    return frequency_table

# def run_train_bpe_one_iteration(
#         input_path: str | os.PathLike,
#             vocab_size: int,
#             ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

def get_most_frequent_pair(pair_frequency_table):
    return max([[pair_frequency_table[key], key] for key in pair_frequency_table])[1]


def merge_pairs(mergable_pair, frequency_table, special_tokens_set):
    new_val = mergable_pair[0] + mergable_pair[1]
    merges = []
    new_keys = []
    new_frequency_table = defaultdict(lambda: 0)
    for key in frequency_table:
        i = 0
        new_key = []
        while i < len(key):
            if i + 1 < len(key):
                pair = (key[i], key[i + 1])
                if (pair == mergable_pair and
                        key[i] not in special_tokens_set and
                        key[i + 1] not in special_tokens_set):
                    new_key.append(new_val)
                    i += 2
                    continue
            new_key.append(key[i])
            i += 1

        new_frequency_table[tuple(new_key)] = frequency_table[key]

        # frequency_table[tuple(new_key)] = frequency_table[key]
        # del frequency_table[key]


    return new_frequency_table


def run_train_bpe_util_profiled(
            input_path: str | os.PathLike,
            vocab_size: int,
            special_tokens: list[str],
            **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    print("BPE TRAINING STARTED")
    start_time = time.time()

    vocabulary = init_vocab(special_tokens)

    # print("Base vocabulary:", vocabulary)
    # print("base vocabulary size", len(vocabulary))
    pre_tokenization_start_time = time.time()
    frequency_table = get_pre_tokens(input_path, special_tokens)
    time_now = time.time()
    print("tokenization time", time_now - pre_tokenization_start_time, time_now - start_time)
    # print("Frequency table:", frequency_table)

    special_tokens_bytes = {token.encode('utf-8') for token in special_tokens}

    pair_frequency_table = defaultdict(lambda: 0)
    pair_index = defaultdict(set)
    print("initial processing complete", time.time() - start_time)

    for key in frequency_table:
        for i in range(len(key) - 1):
            a, b = key[i], key[i + 1]
            if a in special_tokens_bytes or b in special_tokens_bytes:
                continue
            pair = (a, b)
            pair_frequency_table[pair] += frequency_table[key]
            pair_index[pair].add(key)

    merges = []

    if pair_frequency_table:
        max_pair = max(pair_frequency_table.items(), key=lambda x: (x[1], x[0]))[0]
    else:
        max_pair = None

    merging_start_time = time.time()
    # one iteration
    while len(vocabulary) < vocab_size and max_pair is not None:

        # decide which pair to merge
        mergable_pair = max_pair
        merges.append(mergable_pair)
        new_token = mergable_pair[0] + mergable_pair[1]
        # print("Mergable pair:", mergable_pair)

        affected_keys = list(pair_index[mergable_pair])
        for key in affected_keys:
            count = frequency_table[key]
            new_key = []
            i = 0
            while i < len(key):
                if i + 1 < len(key) and (key[i], key[i + 1]) == mergable_pair:
                    new_key.append(new_token)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_key_tuple = tuple(new_key)

            frequency_table[new_key_tuple] = frequency_table.pop(key)

            for i in range(len(key) - 1):
                old_pair = (key[i], key[i + 1])
                if old_pair in pair_frequency_table:
                    pair_frequency_table[old_pair] -= count
                    pair_index[old_pair].discard(key)
                    if pair_frequency_table[old_pair] <= 0:
                        del pair_frequency_table[old_pair]
                        del pair_index[old_pair]

            for i in range(len(new_key_tuple) - 1):
                a, b = new_key_tuple[i], new_key_tuple[i + 1]
                if a in special_tokens_bytes or b in special_tokens_bytes:
                    continue
                pair = (a, b)
                pair_frequency_table[pair] += count
                pair_index[pair].add(new_key_tuple)

        new_id = len(vocabulary)
        vocabulary[new_id] = new_token
        if pair_frequency_table:
            max_pair = max(pair_frequency_table.items(), key=lambda x: (x[1], x[0]))
            max_pair = max_pair[0]
        else:
            max_pair = None

        # print("Vocabulary size", len(vocabulary))
    time_now = time.time()
    print("merging time", time_now - merging_start_time, time_now - start_time)

    return vocabulary, merges

def get_longest_token(vocab: dict[int, bytes]):
    longest_token = max(vocab.values(), key=lambda x: len(x))
    print("the longest token", longest_token)
    return longest_token.decode("utf-8"), len(longest_token)

def run_train_bpe_util(
            input_path: str | os.PathLike,
            vocab_size: int,
            special_tokens: list[str],
            **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocabulary = init_vocab(special_tokens)


    # print("Base vocabulary:", vocabulary)
    # print("base vocabulary size", len(vocabulary))
    frequency_table = get_pre_tokens(input_path, special_tokens)
    # print("Frequency table:", frequency_table)

    special_tokens_bytes = {token.encode('utf-8') for token in special_tokens}

    pair_frequency_table = defaultdict(lambda: 0)
    pair_index = defaultdict(set)

    for key in frequency_table:
        for i in range(len(key) - 1):
            a, b = key[i], key[i + 1]
            if a in special_tokens_bytes or b in special_tokens_bytes:
                continue
            pair = (a, b)
            pair_frequency_table[pair] += frequency_table[key]
            pair_index[pair].add(key)

    merges = []

    if pair_frequency_table:
        max_pair = max(pair_frequency_table.items(), key=lambda x: (x[1], x[0]))[0]
    else:
        max_pair = None

    # one iteration
    while len(vocabulary) < vocab_size and max_pair is not None:

        # decide which pair to merge
        mergable_pair = max_pair
        merges.append(mergable_pair)
        new_token = mergable_pair[0] + mergable_pair[1]
        # print("Mergable pair:", mergable_pair)

        affected_keys = list(pair_index[mergable_pair])
        for key in affected_keys:
            count = frequency_table[key]
            new_key = []
            i = 0
            while i < len(key):
                if i + 1 < len(key) and (key[i], key[i + 1]) == mergable_pair:
                    new_key.append(new_token)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_key_tuple = tuple(new_key)

            frequency_table[new_key_tuple] = frequency_table.pop(key)

            for i in range(len(key) - 1):
                old_pair = (key[i], key[i + 1])
                if old_pair in pair_frequency_table:
                    pair_frequency_table[old_pair] -= count
                    pair_index[old_pair].discard(key)
                    if pair_frequency_table[old_pair] <= 0:
                        del pair_frequency_table[old_pair]
                        del pair_index[old_pair]

            for i in range(len(new_key_tuple) - 1):
                a, b = new_key_tuple[i], new_key_tuple[i + 1]
                if a in special_tokens_bytes or b in special_tokens_bytes:
                    continue
                pair = (a, b)
                pair_frequency_table[pair] += count
                pair_index[pair].add(new_key_tuple)


        new_id = len(vocabulary)
        vocabulary[new_id] = new_token
        if pair_frequency_table:
            max_pair = max(pair_frequency_table.items(), key=lambda x: (x[1], x[0]))
            max_pair = max_pair[0]
        else:
            max_pair = None

        # print("Vocabulary size", len(vocabulary))

    return vocabulary, merges


def save_bpe_model(
        vocabulary: dict[int, bytes],
        merges,
        vocab_filepath,
        merges_filepath,
) -> None:
    vocab_dict = {}
    for token_id, token_bytes in vocabulary.items():
        try:
            token_str = token_bytes.decode('utf-8')
        except Exception:
            token_str = token_bytes.decode('utf-8', errors='replace')
        vocab_dict[token_str] = token_id

    with open(vocab_filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    with open(merges_filepath, 'w', encoding='utf-8') as f:
        for left, right in merges:
            try:
                left_str = left.decode('utf-8')
                right_str = right.decode('utf-8')
            except Exception:
                left_str = left.decode('utf-8', errors='replace')
                right_str = right.decode('utf-8', errors='replace')
            f.write(f"{left_str} {right_str}\n")



def load_bpe_model(
        vocab_filepath,
        merges_filepath
) :
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)

    vocabulary = {}
    for token_str, token_id in vocab_dict.items():
        vocabulary[token_id] = token_str.encode('utf-8')

    merges = []
    with open(merges_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    left_str, right_str = parts
                    left = left_str.encode('utf-8') if left_str else b' '
                    right = right_str.encode('utf-8')
                    merges.append((left, right))

    return vocabulary, merges



# starting 1:35
# =====

# i am getting (b'i', b'n') -> 585
# reference first merge (b' ', b't'),


def main():
    vocab, merges = run_train_bpe_util_profiled(
        input_path="/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/tests/fixtures/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"])

    save_bpe_model(vocab, merges,
                   '/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/vocab.json',
                   '/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/mreges.txt')

    vocab, merges = load_bpe_model('/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/vocab.json',
                           '/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/mreges.txt')

    print(vocab)
    print(merges)
    tokenizer = Tokenizer(vocab, merges, [])
    print("tokenizer ready")
    # run_train_bpe_util_profiled(
    #     input_path="/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/tests/fixtures/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"])
    # peak_mem, result = memory_usage(
    #     (run_train_bpe_util_profiled, (), {  # function, args tuple, kwargs dict
    #         "input_path": '/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/tests/fixtures/TinyStoriesV2-GPT4-train.txt',
    #         "vocab_size": 10000,
    #         "special_tokens": ["<|endoftext|>"]
    #     }),
    #     retval=True,
    #     max_usage=True,
    # )
    #
    # vocab, merges = result
    # longest_token, length = get_longest_token(vocab)
    #
    # print("longest token", length)
    #
    # print(f"Peak memory usage: {peak_mem:.2f} MB")
    # return result


if __name__ == "__main__":
    # run_train_bpe_util_profiled(
    #     input_path="/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/tests/fixtures/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"])
    main()

    # cProfile.run("main()", "bpe_profile.prof")
