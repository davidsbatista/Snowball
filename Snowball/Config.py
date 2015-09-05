#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Snowball import VectorSpaceModel

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import fileinput
import os
import cPickle

from nltk.corpus import stopwords
from Snowball.Seed import Seed
from Snowball.ReVerb import Reverb


class Config(object):

    def __init__(self, config_file, seeds_file, negative_seeds, sentences_file, similarity, confidance):

        self.seed_tuples = set()
        self.negative_seed_tuples = set()
        self.e1_type = None
        self.e2_type = None
        self.stopwords = stopwords.words('english')
        self.threshold_similarity = similarity
        self.instance_confidance = confidance
        self.reverb = Reverb()

        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.wUpdt = float(line.split("=")[1])

            if line.startswith("wUnk"):
                self.wUnk = float(line.split("=")[1])

            if line.startswith("wNeg"):
                self.wNeg = float(line.split("=")[1])

            if line.startswith("number_iterations"):
                self.number_iterations = int(line.split("=")[1])

            if line.startswith("use_RlogF"):
                self.use_RlogF = bool(line.split("=")[1])

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("use_reverb"):
                self.use_reverb = line.split("=")[1].strip()

            if line.startswith("alpha"):
                self.alpha = float(line.split("=")[1])

            if line.startswith("beta"):
                self.beta = float(line.split("=")[1])

            if line.startswith("gamma"):
                self.gamma = float(line.split("=")[1])

        assert self.alpha+self.beta+self.gamma == 1

        self.read_seeds(seeds_file)
        self.read_negative_seeds(negative_seeds)
        fileinput.close()

        print "\nConfiguration parameters"
        print "========================"
        print "Relationship Representation"
        print "e1 type              :", self.e1_type
        print "e2 type              :", self.e2_type
        print "context window       :", self.context_window_size
        print "max tokens away      :", self.max_tokens_away
        print "min tokens away      :", self.min_tokens_away
        print "use ReVerb           :", self.use_reverb

        print "\nVectors"
        print "alpha                :", self.alpha
        print "beta                 :", self.beta
        print "gamma                :", self.gamma

        print "\nSeeds:"
        print "positive seeds       :", len(self.seed_tuples)
        print "negative seeds       :", len(self.negative_seed_tuples)
        print "negative seeds wNeg  :", self.wNeg
        print "unknown seeds wUnk   :", self.wUnk

        print "\nParameters and Thresholds"
        print "threshold_similarity :", self.threshold_similarity
        print "instance confidence  :", self.instance_confidance
        print "min_pattern_support  :", self.min_pattern_support
        print "iterations           :", self.number_iterations
        print "iteration wUpdt      :", self.wUpdt
        print "\n"

        try:
            os.path.isfile("vsm.pkl")
            f = open("vsm.pkl", "r")
            print "\nLoading tf-idf model from disk..."
            self.vsm = cPickle.load(f)
            f.close()

        except IOError:
            print "\nGenerating tf-idf model from sentences..."
            self.vsm = VectorSpaceModel(sentences_file, self.stopwords)
            print "\nWriting generated model to disk..."
            f = open("vsm.pkl", "wb")
            cPickle.dump(self.vsm, f)
            f.close()

    def read_seeds(self, seeds_file):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e2 = line.split(";")[1].strip()
                seed = Seed(e1, e2)
                self.seed_tuples.add(seed)

    def read_negative_seeds(self, negative_seeds):
        for line in fileinput.input(negative_seeds):
                if line.startswith("#") or len(line) == 1:
                    continue
                if line.startswith("e1"):
                    self.e1_type = line.split(":")[1].strip()
                elif line.startswith("e2"):
                    self.e2_type = line.split(":")[1].strip()
                else:
                    e1 = line.split(";")[0].strip()
                    e2 = line.split(";")[1].strip()
                    seed = Seed(e1, e2)
                    self.negative_seed_tuples.add(seed)