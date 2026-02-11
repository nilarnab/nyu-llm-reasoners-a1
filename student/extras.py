import pstats
import random
import time

from student.bpe_trainer_sec_one import Tokenizer
from tests.adapters import run_train_bpe


TINY_STORIES = "/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/tests/fixtures/TinyStoriesV2-GPT4-train.txt"


def sample_documents(file_path: str, n: int = 10, block_size: int = 100000):
    blocks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_block = []
        for line in f:
            line = line.strip()
            if line:
                current_block.append(line)
            if len(current_block) == block_size:
                blocks.append(" ".join(current_block))
                current_block = []

        if current_block:
            blocks.append(" ".join(current_block))

    sampled_blocks = random.sample(blocks, min(n, len(blocks)))

    return sampled_blocks


def get_comopression(tokenizer, docs):
    encoded_docs = [tokenizer.encode(doc) for doc in docs]
    compression_ratios = []

    for i in range(len(docs)):
        byte_count = len(docs[i].encode('utf-8'))
        token_count = len(encoded_docs[i])

        # print("byte count", byte_count, "token_count", token_count)

        ratio = byte_count / token_count if token_count > 0 else 0
        compression_ratios.append(ratio)

    return encoded_docs, compression_ratios


if __name__ == "__main__":

    # print("training")
    # vocab, merges = run_train_bpe(TINY_STORIES, vocab_size=10000, special_tokens=["<|endoftext|>"])
    # tokenizer = Tokenizer(vocab, merges)
    # print("sampling doc")
    # sampled_docs = sample_documents(TINY_STORIES, n=10)
    #
    # print("encoding docs")
    # start_time = time.time()
    # encoded_docs, compression_ratios = get_comopression(tokenizer, sampled_docs)
    # end_time = time.time()
    #
    # total_bytes = sum(len(doc.encode("utf-8")) for doc in sampled_docs)
    # encoding_time = end_time - start_time
    # throughput = total_bytes / encoding_time if encoding_time > 0 else 0
    #
    # print(f"\nTokenizer encoding throughput: {throughput} bytes/sec\n")
    #
    # res = []
    # for i, (doc, encoded, ratio) in enumerate(zip(sampled_docs, encoded_docs, compression_ratios), 1):
    #     print(f"Document {i}:")
    #     print(f"Original length (bytes): {len(doc.encode('utf-8'))}")
    #     print(f"Number of tokens: {len(encoded)}")
    #     print(f"Compression ratio (bytes/token): {ratio:.2f}")
    #     res.append(ratio)
    #     print(f"First 10 token IDs: {encoded[:10]}")
    #     print("-" * 50)
    #
    # print("mean", sum(res)/len(res))

    profile_path = "/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/bpe_profile.prof"
    p = pstats.Stats(profile_path)
    p.sort_stats("cumtime").print_stats(8)


    """
    compression ratio: 4.53 byte / token
    4.15
    
    encoding through put: 1183809.60 bytes / sec
                          2902018
                          5500000
                          5730801
    
    """