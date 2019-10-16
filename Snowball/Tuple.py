#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys

from nltk import pos_tag, word_tokenize
from Snowball.ReVerb import Reverb


class Tuple(object):
    # see: http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
    # select everything except stopwords, ADJ and ADV

    filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']

    def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
        self.e1 = _e1
        self.e2 = _e2
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

        if config.use_reverb == 'yes':
            # construct TF-IDF vectors with the words part of a ReVerb pattern
            # or if no ReVerb patterns with selected words from the contexts
            self.extract_patterns(config)

        elif config.use_reverb == 'no':
            self.bef_vector = self.create_vector(self.bef_words)
            self.bet_vector = self.create_vector(self.bet_words)
            self.aft_vector = self.create_vector(self.aft_words)

        else:
            print("use_reverb configuration parameter not set")
            sys.exit(0)

    def __str__(self):
        return str(self.bef_words.encode("utf8") + ' ' + self.bet_words.encode("utf8") + ' ' +
                   self.aft_words.encode("utf8"))

    def __eq__(self, other):
        return (
                    self.e1 == other.e1 and self.e2 == other.e2 and self.bef_words == other.bef_words and
                    self.bet_words == other.bet_words and self.aft_words == other.aft_words)

    def get_vector(self, context):
        if context == "bef":
            return self.bef_vector
        elif context == "bet":
            return self.bet_vector
        elif context == "aft":
            return self.aft_vector
        else:
            print("Error, vector must be 'bef', 'bet' or 'aft'")
            sys.exit(0)

    def create_vector(self, text):
        vect_ids = self.config.vsm.dictionary.doc2bow(self.tokenize(text))
        return self.config.vsm.tf_idf_model[vect_ids]

    def tokenize(self, text):
        return [word for word in word_tokenize(text.lower())
                if word not in self.config.stopwords]

    def construct_pattern_vector(self, pattern_tags, config):
        # construct TF-IDF representation for each context
        pattern = [t[0] for t in pattern_tags if
                   t[0].lower() not in config.stopwords and t[1] not in
                   self.filter_pos]

        if len(pattern) >= 1:
            vect_ids = self.config.vsm.dictionary.doc2bow(pattern)
            return self.config.vsm.tf_idf_model[vect_ids]

    def construct_words_vectors(self, words, config):
        # split text into tokens and tag them using NLTK's default English tagger
        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        text_tokens = word_tokenize(words)
        tags_ptb = pos_tag(text_tokens)
        pattern = [t[0] for t in tags_ptb if
                   t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
        if len(pattern) >= 1:
            vect_ids = self.config.vsm.dictionary.doc2bow(pattern)
            return self.config.vsm.tf_idf_model[vect_ids]

    def extract_patterns(self, config):

        # extract ReVerb pattern and detect the presence of the passive voice
        patterns_bet_tags = Reverb.extract_reverb_patterns_ptb(self.bet_words)
        if len(patterns_bet_tags) > 0:
            self.passive_voice = self.config.reverb.detect_passive_voice(patterns_bet_tags)
            # forced hack since _'s_ is always tagged as VBZ, (u"'s", 'VBZ') and
            # causes ReVerb to identify a pattern which is wrong, if this happens, ignore
            # that a pattern was extracted
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
