import re
import urllib.request
from typing import List

from tokenizer import SimpleTokenizerV2


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
    urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))

    return raw_text


if __name__ == "__main__":
    tokenizer = SimpleTokenizerV2(vocabularies(the_verdict()))

    # text = "I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication."
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print("\nInput text:", text)

    ids = tokenizer.encode(text)
    print("\nEncoded:", ids)
    print("Decoded:", tokenizer.decode(ids))
