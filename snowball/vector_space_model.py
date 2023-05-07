__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import codecs
import re
import sys

from gensim import corpora
from gensim.models import TfidfModel
from nltk import word_tokenize


class VectorSpaceModel:
    def __init__(self, sentences_file, stopwords):
        self.dictionary = None
        self.corpus = None
        f_sentences = codecs.open(sentences_file, encoding="utf-8")
        documents = list()
        count = 0
        print("Gathering sentences and removing stopwords")
        for line in f_sentences:
            line = re.sub("<[A-Z]+>[^<]+</[A-Z]+>", "", line)

            # remove stop words and tokenize
            document = [word for word in word_tokenize(line.lower()) if word not in stopwords]
            documents.append(document)
            count += 1
            if count % 10000 == 0:
                sys.stdout.write(".")

        f_sentences.close()

        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(text) for text in documents]
        self.tf_idf_model = TfidfModel(self.corpus)

        print(len(documents), "documents red")
        print(len(self.dictionary), " unique tokens")
