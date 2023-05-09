__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import re

from gensim import corpora
from gensim.models import TfidfModel
from nltk import word_tokenize
from tqdm import tqdm

from snowball.commons import blocks


class VectorSpaceModel:
    # pylint: disable=too-few-public-methods
    """
    Vector Space Model class
    # remove stop words and tokenize
    """

    def __init__(self, sentences_file: str, stopwords: set) -> None:
        self.dictionary = None
        self.corpus = None

        with open(sentences_file, "r", encoding="utf8") as f_in:
            total = sum(bl.count("\n") for bl in blocks(f_in))

        with open(sentences_file, "rt", encoding="utf8") as f_sentences:
            documents = []
            for sentence in tqdm(f_sentences, total=total):
                sentence_clean = re.sub("<[A-Z]+>[^<]+</[A-Z]+>", "", sentence)
                document = [word for word in word_tokenize(sentence_clean.lower()) if word not in stopwords]
                documents.append(document)

        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(text) for text in documents]
        self.tf_idf_model = TfidfModel(self.corpus)
        print(f"{len(self.dictionary)} unique tokens")
