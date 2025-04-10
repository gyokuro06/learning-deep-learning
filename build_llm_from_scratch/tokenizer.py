from abc import abstractmethod
import re
from typing import List


class Tokenizer:
    __token_separator: str = r'([,.:;?_!"()\']|--|\s)'
    __capture_symbol_before_space: str = r'\s+([,.?!"()\'])'

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass


class SimpleTokenizerV1(Tokenizer):
    def __init__(self, vocabularies: dict[str, int]):
        self.str_to_int = vocabularies
        self.int_to_str = {i: s for s, i in vocabularies.items()}

    def __preprocess(self, text: str) -> List[str]:
        items = re.split(self._Tokenizer__token_separator, text)
        result = [item.strip() for item in items if item.strip()]
        return result

    def encode(self, text: str) -> List[int]:
        preprocessed = self.__preprocess(text)
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        text = " ".join([self.int_to_str[id] for id in ids])
        text = re.sub(self._Tokenizer__capture_symbol_before_space, r"\1", text)
        return text


class SimpleTokenizerV2(Tokenizer):
    def __init__(self, vocabularies: dict[str, int]):
        self.str_to_int = vocabularies
        self.int_to_str = {i: s for s, i in vocabularies.items()}

    def __preprocessed(self, text: str) -> List[str]:
        splitted = re.split(self._Tokenizer__token_separator, text)
        trimmed = [item.strip() for item in splitted if item.strip()]
        unk_replaced = [
            item if item in self.str_to_int else "<|unk|>" for item in trimmed
        ]
        return unk_replaced

    def encode(self, text: str) -> List[int]:
        preprocessed = self.__preprocessed(text)
        ids = [self.str_to_int[item] for item in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        text = " ".join([self.int_to_str[id] for id in ids])
        text = re.sub(self._Tokenizer__capture_symbol_before_space, r"\1", text)
        return text
