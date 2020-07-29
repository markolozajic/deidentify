from abc import ABC, abstractmethod
from typing import Iterable

import spacy
from loguru import logger


class Tokenizer(ABC):

    def __init__(self, disable: Iterable[str] = ()):
        """Tokenizer base class.

        Parameters
        ----------
        disable : Iterable[str]
            Steps of the spacy pipeline to disable.
            See: https://spacy.io/usage/processing-pipelines/#disabling

        """
        self.disable = disable

    @abstractmethod
    def parse_text(self, text: str) -> spacy.tokens.doc.Doc:
        pass


class TokenizerFactory():
    """Construct tokenizer instance per corpus. Use default German spaCy tokenizer."""

    @staticmethod
    def tokenizer(corpus: str, disable: Iterable[str] = (), model: str = 'de_core_news_sm'):
        logger.info('Tokenizer for corpus: {}'.format(corpus))

        from deidentify.tokenizer.tokenizer_de import TokenizerDE
        return TokenizerDE(disable=disable)
