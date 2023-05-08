__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import numpy as np
from nltk import word_tokenize

from snowball.reverb_breds import Reverb


class SnowballTuple:
    # pylint: disable=too-many-instance-attributes
    """
    Tuple class:

    see: http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
    select everything except stopwords, ADJ and ADV

    # construct TF-IDF vectors with the words part of a ReVerb pattern
    # or if no ReVerb patterns with selected words from the contexts

    """

    filter_pos = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"]

    def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
        self.ent1 = _e1
        self.ent2 = _e2
        self.sentence = _sentence
        self.confidence = 0
        self.confidence_old = 0
        self.bef_words = _before
        self.bet_words = _between
        self.aft_words = _after
        self.config = config
        self.bef_vector = None
        self.bet_vector = None
        self.aft_vector = None
        self.bef_reverb_vector = None
        self.bet_reverb_vector = None
        self.aft_reverb_vector = None
        self.passive_voice = None

        if config.use_reverb == "yes":
            self.extract_patterns(config)

        elif config.use_reverb == "no":
            self.bef_vector = self.create_vector(self.bef_words)
            self.bet_vector = self.create_vector(self.bet_words)
            self.aft_vector = self.create_vector(self.aft_words)

    def __str__(self):
        return f"{self.bef_words}  {self.bet_words}  {self.aft_words}"

    def __eq__(self, other):
        return (
            self.ent1 == other.ent1
            and self.ent2 == other.ent2
            and self.bef_words == other.bef_words
            and self.bet_words == other.bet_words
            and self.aft_words == other.aft_words
        )

    def __hash__(self) -> int:
        return hash(self.ent1) ^ hash(self.ent2)

    def get_vector(self, context):
        """
        Return the vector for the given context
        """
        if context == "bef":
            return self.bef_vector
        if context == "bet":
            return self.bet_vector
        # ToDo: can only be "aft" here
        return self.aft_vector

    def create_vector(self, text: str) -> np.array:
        """
        Create a TF-IDF vector for the given text
        """
        vect_ids = self.config.vsm.dictionary.doc2bow(self.tokenize(text))
        return self.config.vsm.tf_idf_model[vect_ids]

    def tokenize(self, text):
        """
        Tokenize text and remove stopwords
        """
        return [word for word in word_tokenize(text.lower()) if word not in self.config.stopwords]

    def construct_pattern_vector(self, pattern_tags, config):  # pylint: disable=inconsistent-return-statements
        """
        Construct TF-IDF representation for each context
        """
        pattern = [t[0] for t in pattern_tags if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]

        if len(pattern) >= 1:
            vect_ids = self.config.vsm.dictionary.doc2bow(pattern)
            return self.config.vsm.tf_idf_model[vect_ids]

    def construct_words_vectors(self, words, config):
        """
        Construct TF-IDF representation for each context
        """
        tokens, tags = zip(*words)
        pattern = [
            token
            for token, tag in zip(tokens, tags)
            if token.lower() not in config.stopwords and tag not in self.filter_pos
        ]
        vect_ids = self.config.vsm.dictionary.doc2bow(pattern)
        return self.config.vsm.tf_idf_model[vect_ids]

    def extract_patterns(self, config) -> None:
        """
        If a ReVerb pattern is found in the BET context it constructs a TF-IDF vector with the words part of the
        pattern, otherwise uses all words filtering stopwords, ADJ and ADV.

        For the BEF and AFT contexts it uses all words filtering stopwords, ADJ and ADV.
        """

        if patterns_bet_tags := Reverb.extract_reverb_patterns_tagged_ptb(self.bet_words):
            self.passive_voice = self.config.reverb.detect_passive_voice(patterns_bet_tags)
            # 's_ is always wrongly tagged as VBZ, if the first word is 's' ignore it
            if patterns_bet_tags[0][0] == "'s":
                self.bet_vector = self.construct_words_vectors(self.bet_words, config)
            else:
                self.bet_vector = self.construct_pattern_vector(patterns_bet_tags, config)
        else:
            self.bet_vector = self.construct_words_vectors(self.bet_words, config)

        # extract two words before the first entity, and two words after the second entity
        if len(self.bef_words) > 0:
            self.bef_vector = self.construct_words_vectors(self.bef_words, config)

        if len(self.aft_words) > 0:
            self.aft_vector = self.construct_words_vectors(self.aft_words, config)
