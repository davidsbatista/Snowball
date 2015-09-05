#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys

from copy import deepcopy
from math import log


class Pattern(object):

    def __init__(self, t=None):
        self.positive = 0
        self.negative = 0
        self.unknown = 0
        self.confidence = 0
        self.tuples = list()
        self.tuple_patterns = set()
        self.centroid_bef = list()
        self.centroid_bet = list()
        self.centroid_aft = list()
        if tuple is not None:
            self.tuples.append(t)
            self.centroid_bef = t.bef_vector
            self.centroid_bet = t.bet_vector
            self.centroid_aft = t.aft_vector

    def __str__(self):
        output = ''
        for t in self.tuples:
            output += str(t)+'|'
        return output

    def __eq__(self, other):
        if set(self.tuples) == set(other.tuples):
            return True
        else:
            return False

    def update_confidence_2003(self, config):
        if self.positive > 0:
            self.confidence = log(float(self.positive), 2) * (float(self.positive) / float(self.positive + self.unknown
                                                                                           * config.wUnk + self.negative
                                                                                           * config.wNeg))
        elif self.positive == 0:
            self.confidence = 0

    def update_confidence(self):
        if self.positive > 0 or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, t):
        self.tuples.append(t)
        self.centroid(self)

    def update_selectivity(self, t, config):
        for s in config.seed_tuples:
            if s.e1 == t.e1 or s.e1.strip() == t.e1.strip():
                if s.e2 == t.e2.strip() or s.e2.strip() == t.e2.strip():
                    self.positive += 1
                else:
                    self.negative += 1
            else:
                for n in config.negative_seed_tuples:
                    if n.e1 == t.e1 or n.e1.strip() == t.e1.strip():
                        if n.e2 == t.e2.strip() or n.e2.strip() == t.e2.strip():
                            self.negative += 1
                self.unknown += 1

        #self.update_confidence()
        self.update_confidence_2003(config)

    def merge_tuple_patterns(self):
        #fazer o merge tendo em consideração todos os contextos
        for t in self.tuples:
            self.tuple_patterns.add(t.bet_words)

    @staticmethod
    def centroid(self):
        # it there just one tuple associated with this pattern centroid is the tuple
        if len(self.tuples) == 1:
            t = self.tuples[0]
            self.centroid_bef = t.bef_vector
            self.centroid_bet = t.bet_vector
            self.centroid_aft = t.aft_vector
        else:
            # if there are more tuples associated, calculate the average over all vectors
            self.centroid_bef = self.calculate_centroid(self, "bef")
            self.centroid_bet = self.calculate_centroid(self, "bet")
            self.centroid_aft = self.calculate_centroid(self, "aft")

    @staticmethod
    def calculate_centroid(self, context):
        centroid = deepcopy(self.tuples[0].get_vector(context))
        if centroid is not None:
            # add all other words from other tuples
            for t in range(1, len(self.tuples), 1):
                current_words = [e[0] for e in centroid]
                words = self.tuples[t].get_vector(context)
                if words is not None:
                    for word in words:
                        # if word already exists in centroid, update its tf-idf
                        if word[0] in current_words:
                            # get the current tf-idf for this word in the centroid
                            for i in range(0, len(centroid), 1):
                                if centroid[i][0] == word[0]:
                                    current_tf_idf = centroid[i][1]
                                    # sum the tf-idf from the tuple to the current tf_idf
                                    current_tf_idf += word[1]
                                    # update (w,tf-idf) in the centroid
                                    w_new = list(centroid[i])
                                    w_new[1] = current_tf_idf
                                    centroid[i] = tuple(w_new)
                                    break
                        # if it is not in the centroid, added it with the associated tf-idf score
                        else:
                            centroid.append(word)

            # divide tf-idf score of tuple (w,tf-idf), by the number of vectors
            for i in range(0, len(centroid), 1):
                tmp = list(centroid[i])
                tmp[1] /= len(self.tuples)
                # assure that the tf-idf values are still normalized
                try:
                    assert tmp[1] <= 1.0
                    assert tmp[1] >= 0.0
                except AssertionError:
                    "Error calculating extraction pattern centroid"
                    sys.exit(0)
                centroid[i] = tuple(tmp)

        return centroid