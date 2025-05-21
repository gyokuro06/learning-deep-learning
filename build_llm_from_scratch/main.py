import os
import re
import urllib.request
from importlib.metadata import version
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

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_v1_dataloader(
        txt=the_verdict(),
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("\nToken IDs:\n", inputs)
    print("Input shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
