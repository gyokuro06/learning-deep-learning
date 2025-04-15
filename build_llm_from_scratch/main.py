from importlib.metadata import version
import os
import re
import urllib.request
from typing import List

import tiktoken
import torch

from dataset import create_v1_dataloader


def vocabularies(text: str) -> dict[str, int]:
    regex = r'([,.:;?_!"()\']|--|\s)'
    items = re.split(regex, text)
    tokens = [item.strip() for item in items if item.strip()]
    all_words = sorted(set(tokens))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocabularies = {token: i for i, token in enumerate(all_words)}
    print("Total number of vocabularies:", len(vocabularies))
    return vocabularies


def the_verdict() -> List[str]:
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-verdict.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))

    return raw_text


if __name__ == "__main__":
    print("tiktoken version:", version("tiktoken"))
    print("torch version", version("torch"))
    tokenizer = tiktoken.get_encoding("gpt2")

    enc_text = tokenizer.encode(the_verdict())
    print("Total number of encoded tokens:", len(enc_text))
    enc_sample = enc_text[50:]
    context_size = 4
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

    text = "Akwirw ier"
    print("\nInput text:", text)

    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("\nEncoded:", ids)
    print("Decoded:", tokenizer.decode(ids))

    dataloader = create_v1_dataloader(
        txt=the_verdict(), batch_size=8, max_length=4, stride=4, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("\nInputs:\n", inputs)
    print("Targets:\n", targets)

    test_input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
