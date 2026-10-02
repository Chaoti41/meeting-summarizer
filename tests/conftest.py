import pytest

from meetsum.config import Config
from meetsum.embeddings import HashingEmbedder
from meetsum.preprocess import load_nlp
from meetsum.summarize import MeetingSummarizer


@pytest.fixture(scope="session")
def nlp():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_nlp("en_core_web_sm")


@pytest.fixture(scope="session")
def summarizer(nlp):
    return MeetingSummarizer(Config(), embedder=HashingEmbedder(), nlp=nlp)
