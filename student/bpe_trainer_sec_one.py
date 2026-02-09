import os
from typing import Iterable, Iterator

import regex as re
from collections import defaultdict
from student.pretokenization_example import find_chunk_boundaries
# from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Answer to problem (tokenizer)
# ===========
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        if special_tokens is None:
            special_tokens = []
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text_raw):
        # for special_token in self.special_tokens:
        #     text = text.replace(special_token, "")
        pattern_special = None
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=lambda s: len(s), reverse=True)

            pattern_special = '(' + '|'.join(re.escape(st) for st in self.special_tokens) + ')'

        if pattern_special is not None:
            text_chunks = re.split(pattern_special, text_raw)
        else:
            text_chunks = [text_raw]

        # print("text_chunks", text_chunks)

        res = []
        for text in text_chunks:
            if text in self.special_tokens:
                # print("textrying to get", text.encode("utf-8"))
                # print("got", self._token_to_id(text.encode("utf-8")))
                res.append(self._token_to_id(text.encode("utf-8")))
                pass
            elif text:
                pre_tokens = []
                for token in re.finditer(PAT, text):
                    # print("while enoding, getting token", token.group().encode("utf-8"))

                    pre_tokens.append(tuple(bytes([b]) for b in token.group().encode("utf-8")))

                # print("pre tokens", pre_tokens)

                new_pre_tokens = []
                for token_raw in pre_tokens:
                    token = [el for el in token_raw]
                    for merge in self.merges:
                    # for merge in []:
                        i = 0
                        new_token = []
                        while i < len(token):
                            if i + 1 < len(token):
                                pair = (token[i], token[i + 1])
                                if pair == merge:
                                    # print("merging", pair)
                                    new_token_sub = merge[0] + merge[1]
                                    new_token.append(new_token_sub)
                                    i += 2
                                    continue

                            new_token.append(token[i])
                            i += 1
                        token = [el for el in new_token]
                    new_pre_tokens.append(tuple(token))

                # print("new pre tokens:", new_pre_tokens)

                for token in new_pre_tokens:
                    # print("token in new pre token", token)
                    for sub_token in token:
                        # print("sub token", sub_token)
                        token_id = self._token_to_id(sub_token)
                        res.append(token_id)

        return res



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
        for token_id in self.vocab:
            if token == self.vocab[token_id]:
                return token_id
        # print("problem with token", token)
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


def get_pre_tokens(input_path: str | os.PathLike, special_tokens: list[str]):
    # TODO: not sure, whether pretokens are the frequency table or the list of bytes-tuples ?
    file = open(input_path, "rb")
    frequency_table = defaultdict(lambda: 0)

    num_processes = 4
    boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

    # taken from profs code:
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

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

    merges = []

    pair_frequency_table = defaultdict(lambda: 0)
    for key in frequency_table:
        for i in range(0, len(key) - 1):
            if key[i] not in special_tokens_bytes and key[i + 1] not in special_tokens_bytes:
                pair_frequency_table[(key[i], key[i + 1])] += frequency_table[key]

    # one iteration
    while len(vocabulary) < vocab_size:
        if len(pair_frequency_table) == 0:
            break

        # decide which pair to merge
        mergable_pair = get_most_frequent_pair(pair_frequency_table)
        merges.append(mergable_pair)
        # print("Mergable pair:", mergable_pair)

        new_val = mergable_pair[0] + mergable_pair[1]
        new_frequency_table = defaultdict(lambda: 0)

        for key in frequency_table:
            i = 0
            new_key = []

            while i < len(key):
                if i + 1 < len(key):
                    pair = (key[i], key[i + 1])
                    if (pair == mergable_pair and
                            key[i] not in special_tokens_bytes and
                            key[i + 1] not in special_tokens_bytes):
                        new_key.append(new_val)
                        i += 2
                        continue
                new_key.append(key[i])
                i += 1

            new_key_tuple = tuple(new_key)
            new_frequency_table[new_key_tuple] += frequency_table[key]

            if key != new_key_tuple:
                for i in range(len(key) - 1):
                    old_pair = (key[i], key[i + 1])
                    if (key[i] not in special_tokens_bytes and
                            key[i + 1] not in special_tokens_bytes):
                        pair_frequency_table[old_pair] -= frequency_table[key]
                        if pair_frequency_table[old_pair] <= 0:
                            del pair_frequency_table[old_pair]

                for i in range(len(new_key_tuple) - 1):
                    new_pair = (new_key_tuple[i], new_key_tuple[i + 1])
                    if (new_key_tuple[i] not in special_tokens_bytes and
                            new_key_tuple[i + 1] not in special_tokens_bytes):
                        pair_frequency_table[new_pair] += frequency_table[key]

        frequency_table = new_frequency_table

        new_id = len(vocabulary)
        vocabulary[new_id] = new_val

        # print("Vocabulary size", len(vocabulary))

    return vocabulary, merges

# =====

# i am getting (b'i', b'n') -> 585
# reference first merge (b' ', b't'),

# if __name__ == "__main__":
    # run_train_bpe_util(input_path="/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/student/assets/test_corpus", vocab_size=258, special_tokens=["<|endoftext|>"])
    # merges = [('a', 'b'), ('ab', 'c')]
    # vocab = {1: 'a'.encode("utf-8"),
    #          2: 'b'.encode("utf-8"),
    #          3: 'c'.encode("utf-8"),
    #          4: 'ab'.encode("utf-8"),
    #          5: ' '.encode("utf-8"),
    #          6: 'abc'.encode("utf-8"),}
    # merges = [(el[0].encode("utf-8"), el[1].encode("utf-8")) for el in merges]
    # tokenizer = Tokenizer(vocab, merges)
    # file = open("/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/student/assets/test_corpus", "r")
    # encoding = tokenizer.encode_iterable(file)
    # print(encoding)

    # decoded_text = tokenizer.decode(encoding)
    # print(decoded_text)
