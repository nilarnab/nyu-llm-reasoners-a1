import os


def split_text_file(input_path, train_path, test_path, train_ratio=0.8):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    split_idx = int(len(content) * train_ratio)

    train_data = content[:split_idx]
    test_data = content[split_idx:]

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_data)

    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_data)


if __name__ == '__main__':
    split_text_file(
        input_path='/Users/nilarnabdebnath/Documents/course_work/sem2/llm_reasoners/assignment1/nyu-llm-reasoners-a1/tests/fixtures/tinystories_sample_5M.txt',
        train_path='tinystories_sample_5M_train.txt',
        test_path='tinystories_sample_5M_val.txt',
        train_ratio=0.8
    )