import urllib.request
import re
from typing import List


class SimpleTokenizerV1:
    def __init__(self, vocabularies: dict[str, int]):
        self.str_to_int = vocabularies
        self.int_to_str = {i: s for s, i in vocabularies.items()}

    def __preprocess(self, text: str) -> List[str]:
        regex = r'([,.:;?_!"()\']|--|\s)'
        items = re.split(regex, text)
        result = [item.strip() for item in items if item.strip()]
        return result

    def encode(self, text: str) -> List[int]:
        preprocessed = self.__preprocess(text)
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        # all_words = sorted(set(tokens))
        # vocabularies = {token: integer for integer, token in enumerate(all_words)}
        # result = []
        # for i, item in enumerate(vocabularies.items()):
        #     print(item)
        #     result.append(i)
        #     if i >= 50:
        #         break
        # return result

    def decode(self, ids: List[int]) -> str:
        text = " ".join([self.int_to_str[id]] for id in ids)
        text = re.sub(r'\s+([,.?!"()\')', r"\1", text)
        return text


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
    print(raw_text[:99])

    return raw_text


if __name__ == "__main__":
    # text = "Hello, world. This, is a test."
    text = the_verdict()
    preprocess(text)
