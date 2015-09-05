#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import re
from nltk import word_tokenize

# regex for simple tags, e.g.:
# <PER>Bill Gates</PER>
regex_simple = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)

# regex for wikipedia linked tags e.g.:
# <PER url=http://en.wikipedia.org/wiki/Mark_Zuckerberg>Mark Elliot Zuckerberg</PER>
regex_linked = re.compile('<[A-Z]+ url=[^>]+>[^<]+</[A-Z]+>', re.U)


class Relationship:
    def __init__(self, _sentence, _before=None, _between=None, _after=None, _ent1=None, _ent2=None, _arg1type=None,
                 _arg2type=None, _type=None):

        self.sentence = _sentence
        self.rel_type = _type
        self.before = _before
        self.between = _between
        self.after = _after
        self.ent1 = _ent1
        self.ent2 = _ent2
        self.arg1type = _arg1type
        self.arg2type = _arg2type

        if _before is None and _between is None and _after is None and _sentence is not None:
            matches = []
            for m in re.finditer(regex_linked, self.sentence):
                matches.append(m)

            for x in range(0, len(matches) - 1):
                if x == 0:
                    start = 0
                if x > 0:
                    start = matches[x - 1].end()
                try:
                    end = matches[x + 2].init_bootstrapp()
                except IndexError:
                    end = len(self.sentence) - 1

                self.before = self.sentence[start:matches[x].init_bootstrapp()]
                self.between = self.sentence[matches[x].end():matches[x + 1].init_bootstrapp()]
                self.after = self.sentence[matches[x + 1].end(): end]
                self.ent1 = matches[x].group()
                self.ent2 = matches[x + 1].group()
                arg1match = re.match("<[A-Z]+>", self.ent1)
                arg2match = re.match("<[A-Z]+>", self.ent2)
                self.arg1type = arg1match.group()[1:-1]
                self.arg2type = arg2match.group()[1:-1]

    def __eq__(self, other):
        if self.ent1 == other.ent1 and self.before == other.before and self.between == other.between \
                and self.after == other.after:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.ent1) ^ hash(self.ent2) ^ hash(self.before) ^ hash(self.between) ^ hash(self.after)


class Sentence:

    def __init__(self, _sentence, e1_type, e2_type, max_tokens, min_tokens, window_size):
        self.relationships = set()
        self.sentence = _sentence
        matches = []

        #TODO: regex to used depends on Config.tags_type
        #for m in re.finditer(regex_linked, self.sentence):
        for m in re.finditer(regex_simple, self.sentence):
            matches.append(m)

        if len(matches) >= 2:
            for x in range(0, len(matches) - 1):
                if x == 0:
                    start = 0
                if x > 0:
                    start = matches[x - 1].end()
                try:
                    end = matches[x + 2].start()
                except IndexError:
                    end = len(self.sentence) - 1

                before = self.sentence[start:matches[x].start()]
                between = self.sentence[matches[x].end():matches[x + 1].start()]
                after = self.sentence[matches[x + 1].end(): end]

                # select 'window_size' tokens from left and right context
                before = word_tokenize(before)[-window_size:]
                after = word_tokenize(after)[:window_size]
                before = ' '.join(before)
                after = ' '.join(after)

                # only consider relationships where the distance between the two entities
                # is less than 'max_tokens' and greater than 'min_tokens'
                number_bet_tokens = len(word_tokenize(between))
                if not number_bet_tokens > max_tokens and not number_bet_tokens < min_tokens:

                    #TODO: run code according to Config.tags_type
                    # simple tags
                    ent1 = matches[x].group()
                    ent2 = matches[x + 1].group()
                    arg1match = re.match("<[A-Z]+>", ent1)
                    arg2match = re.match("<[A-Z]+>", ent2)
                    ent1 = re.sub("</?[A-Z]+>", "", ent1, count=2, flags=0)
                    ent2 = re.sub("</?[A-Z]+>", "", ent2, count=2, flags=0)
                    arg1type = arg1match.group()[1:-1]
                    arg2type = arg2match.group()[1:-1]

                    """
                    # linked tags
                    ent1 = re.findall('url=([^>]+)', matches[x].group())[0]
                    ent2 = re.findall('url=([^>]+)', matches[x+1].group())[0]
                    arg1type = re.findall('<([A-Z]+)', matches[x].group())[0]
                    arg2type = re.findall('<([A-Z]+)', matches[x+1].group())[0]
                    """

                    #DEBUG
                    """
                    print _sentence
                    print matches[x].group()
                    print matches[x+1].group()
                    print "BEF", before
                    print "BET", between
                    print "AFT", after
                    print "ent1", ent1, arg1type
                    print "ent2", ent2, arg2type
                    print "==========================================\n"
                    """

                    if ent1 == ent2:
                        continue

                    if e1_type is not None and e2_type is not None:
                        # restrict relationships by the arguments semantic types
                        if arg1type == e1_type and arg2type == e2_type:

                            rel = Relationship(_sentence, before, between, after, ent1, ent2, arg1type, arg2type,
                                               _type=None)
                            self.relationships.add(rel)

                    elif e1_type is None and e2_type is None:
                        # create all possible relationship types
                        rel = Relationship(_sentence, before, between, after, ent1, ent2, arg1type, arg2type,
                                           _type=None)
                        self.relationships.add(rel)


class SentenceParser:

    def __init__(self, _sentence, e1_type, e2_type):
        self.relationships = set()
        self.sentence = _sentence
        self.entities = list()
        self.valid = False
        self.tree = None
        self.deps = None

        for m in re.finditer(regex, self.sentence):
            self.entities.append(m.group())

        for e1 in self.entities:
            for e2 in self.entities:
                if e1 == e2:
                    continue
                arg1match = re.match("<([A-Z]+)>", e1)
                arg2match = re.match("<([A-Z]+)>", e2)
                if arg1match.group(1) == e1_type and arg2match.group(1) == e2_type:
                    self.valid = True
                    break;