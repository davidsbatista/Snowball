#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import fileinput
import StringIO
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.mapping import map_tag
from nltk.tokenize.punkt import PunktWordTokenizer
from SentenceBreds import Sentence


class Reverb(object):

    def __init__(self):
        self.lmtzr = WordNetLemmatizer()
        self.aux_verbs = ['be']

    @staticmethod
    def extract_reverb_patterns(text):
        """
        Extract ReVerb relational patterns
        http://homes.cs.washington.edu/~afader/bib_pdf/emnlp11.pdf

        VERB - verbs (all tenses and modes)
        NOUN - nouns (common and proper)
        PRON - pronouns
        ADJ - adjectives
        ADV - adverbs
        ADP - adpositions (prepositions and postpositions)
        CONJ - conjunctions
        DET - determiners
        NUM - cardinal numbers
        PRT - particles or other function words
        X - other: foreign words, typos, abbreviations
        . - punctuation

        # extract ReVerb patterns:
        # V | V P | V W*P
        # V = verb particle? adv?
        # W = (noun | adj | adv | pron | det)
        # P = (prep | particle | inf. marker)
        """

        # split text into tokens
        text_tokens = PunktWordTokenizer().tokenize(text)

        # tag the sentence, using the default NTLK English tagger
        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        tags_ptb = pos_tag(text_tokens)

        # convert the tags to reduced tagset (Petrov et al. 2012)
        # http://arxiv.org/pdf/1104.2086.pdf
        tags = []
        for t in tags_ptb:
            tag = map_tag('en-ptb', 'universal', t[1])
            tags.append((t[0], tag))

        patterns = []
        patterns_tags = []
        i = 0
        limit = len(tags)-1

        while i <= limit:
            tmp = StringIO.StringIO()
            tmp_tags = []

            # a ReVerb pattern always starts with a verb
            if tags[i][1] == 'VERB':
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

                # V = verb particle? adv? (also capture auxiliary verbs)
                while i <= limit and tags[i][1] in ['VERB', 'PRT', 'ADV']:
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # W = (noun | adj | adv | pron | det)
                while i <= limit and tags[i][1] in ['NOUN', 'ADJ', 'ADV', 'PRON', 'DET']:
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # P = (prep | particle | inf. marker)
                while i <= limit and tags[i][1] in ['ADP', 'PRT']:
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1
                # add the build pattern to the list collected patterns
                patterns.append(tmp.getvalue())
                patterns_tags.append(tmp_tags)
            i += 1

        return patterns, patterns_tags

    @staticmethod
    def extract_reverb_patterns_tagged_ptb(tagged_text):
        """
        Extract ReVerb relational patterns
        http://homes.cs.washington.edu/~afader/bib_pdf/emnlp11.pdf
        """

        # The pattern limits the relation to be a verb (e.g., invented), a verb followed immediately by
        # a preposition (e.g., located in), or a verb followed by nouns, adjectives, or adverbs ending in a preposition
        # (e.g., has an atomic weight of).

        # V | V P | V W*P
        # V = verb particle? adv?
        # W = (noun | adj | adv | pron | det)
        # P = (prep | particle | inf. marker)

        patterns = []
        patterns_tags = []
        i = 0
        limit = len(tagged_text)-1
        tags = tagged_text

        verb = ['VB', 'VBD', 'VBD|VBN', 'VBG', 'VBG|NN', 'VBN', 'VBP', 'VBP|TO', 'VBZ', 'VP']
        adverb = ['RB', 'RBR', 'RBS', 'RB|RP', 'RB|VBG', 'WRB']
        particule = ['POS', 'PRT', 'TO', 'RP']
        noun = ['NN', 'NNP', 'NNPS', 'NNS', 'NN|NNS', 'NN|SYM', 'NN|VBG', 'NP']
        adjectiv = ['JJ', 'JJR', 'JJRJR', 'JJS', 'JJ|RB', 'JJ|VBG']
        pronoun = ['WP', 'WP$', 'PRP', 'PRP$', 'PRP|VBP']
        determiner = ['DT', 'EX', 'PDT', 'WDT']
        adp = ['IN', 'IN|RP']

        # TODO: detect negations
        # ('rejected', 'VBD'), ('a', 'DT'), ('takeover', 'NN')

        # TODO: If there are multiple possible matches in a sentence for a single verb, the longest possible
        # match is chosen.

        while i <= limit:
            tmp = StringIO.StringIO()
            tmp_tags = []

            # a ReVerb pattern always starts with a verb
            if tags[i][1] in verb:
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

                # V = verb particle? adv? (also capture auxiliary verbs)
                while i <= limit and (tags[i][1] in verb or tags[i][1] in adverb or tags[i][1] in particule):
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # W = (noun | adj | adv | pron | det)
                while i <= limit and (tags[i][1] in noun or tags[i][1] in adjectiv or tags[i][1] in adverb or
                                      tags[i][1] in pronoun or tags[i][1] in determiner):
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # P = (prep | particle | inf. marker)
                while i <= limit and (tags[i][1] in adp or tags[i][1] in particule):
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # add the build pattern to the list collected patterns
                patterns.append(tmp.getvalue())
                patterns_tags.append(tmp_tags)
            i += 1

        # Finally, if the pattern matches multiple adjacent sequences, we merge them into a single relation phrase
        # (e.g.,wants to extend). This refinement enables the model to readily handle relation phrases containing
        # multiple verbs.

        merged_patterns_tags = [item for sublist in patterns_tags for item in sublist]
        return merged_patterns_tags

    @staticmethod
    def extract_reverb_patterns_ptb(text):
        """
        Extract ReVerb relational patterns
        http://homes.cs.washington.edu/~afader/bib_pdf/emnlp11.pdf
        """

        # The pattern limits the relation to be a verb (e.g., invented), a verb followed immediately by
        # a preposition (e.g., located in), or a verb followed by nouns, adjectives, or adverbs ending in a preposition
        # (e.g., has an atomic weight of).

        # V | V P | V W*P
        # V = verb particle? adv?
        # W = (noun | adj | adv | pron | det)
        # P = (prep | particle | inf. marker)

        # split text into tokens
        text_tokens = PunktWordTokenizer().tokenize(text)

        # tag the sentence, using the default NTLK English tagger
        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        tags_ptb = pos_tag(text_tokens)
        patterns = []
        patterns_tags = []
        i = 0
        limit = len(tags_ptb)-1
        tags = tags_ptb

        verb = ['VB', 'VBD', 'VBD|VBN', 'VBG', 'VBG|NN', 'VBN', 'VBP', 'VBP|TO', 'VBZ', 'VP']
        adverb = ['RB', 'RBR', 'RBS', 'RB|RP', 'RB|VBG', 'WRB']
        particule = ['POS', 'PRT', 'TO', 'RP']
        noun = ['NN', 'NNP', 'NNPS', 'NNS', 'NN|NNS', 'NN|SYM', 'NN|VBG', 'NP']
        adjectiv = ['JJ', 'JJR', 'JJRJR', 'JJS', 'JJ|RB', 'JJ|VBG']
        pronoun = ['WP', 'WP$', 'PRP', 'PRP$', 'PRP|VBP']
        determiner = ['DT', 'EX', 'PDT', 'WDT']
        adp = ['IN', 'IN|RP']

        # match is chosen.

        while i <= limit:
            tmp = StringIO.StringIO()
            tmp_tags = []

            # a ReVerb pattern always starts with a verb
            if tags[i][1] in verb:
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

                # V = verb particle? adv? (also capture auxiliary verbs)
                while i <= limit and (tags[i][1] in verb or tags[i][1] in adverb or tags[i][1] in particule):
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # W = (noun | adj | adv | pron | det)
                while i <= limit and (tags[i][1] in noun or tags[i][1] in adjectiv or tags[i][1] in adverb or
                                      tags[i][1] in pronoun or tags[i][1] in determiner):
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # P = (prep | particle | inf. marker)
                while i <= limit and (tags[i][1] in adp or tags[i][1] in particule):
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # add the build pattern to the list collected patterns
                patterns.append(tmp.getvalue())
                patterns_tags.append(tmp_tags)
            i += 1

        # Finally, if the pattern matches multiple adjacent sequences, we merge them into a single relation phrase
        # (e.g.,wants to extend). This refinement enables the model to readily handle relation phrases containing
        # multiple verbs.

        merged_patterns_tags = [item for sublist in patterns_tags for item in sublist]
        return merged_patterns_tags

    def detect_passive_voice(self, pattern):
        passive_voice = False
        #TODO: há casos mais complexos, adjectivos ou adverbios pelo meio, por exemplo:
            # ORG1 'which is being bought by' ORG2
            #
            # "ENT1 was first put forward by ENT2
            #
            # <ORG>Sun Microsystems</ORG> , which was acquired instead by business software giant <ORG>Oracle</ORG> .
            #
            # This placement allowed <ORG>AOL</ORG> to draw users very quickly and gave Explorer prominence over rival
            # <ORG>Netscape</ORG> , which was later bought by <ORG>AOL</ORG>.

        # to be + past verb + by
        if len(pattern) >= 3:
            if pattern[0][1].startswith('V'):
                verb = self.lmtzr.lemmatize(pattern[0][0], 'v')
                if verb in self.aux_verbs:
                    if (pattern[1][1] == 'VBN' or pattern[1][1] == 'VBD') and pattern[-1][0] == 'by':
                        passive_voice = True

                    # past verb + by
                    elif (pattern[-2][1] == 'VBN' or pattern[-2][1] == 'VBD') and pattern[-1][0] == 'by':
                        passive_voice = True

                # past verb + by
                elif (pattern[-2][1] == 'VBN' or pattern[-2][1] == 'VBD') and pattern[-1][0] == 'by':
                        passive_voice = True

        # past verb + by
        elif len(pattern) >= 2:
            if (pattern[-2][1] == 'VBN' or pattern[-2][1] == 'VBD') and pattern[-1][0] == 'by':
                passive_voice = True

        return passive_voice


def main():
    # for testing, it extracts PER-ORG relationships from a file, where each line is a sentence with
    # the named-entities tagged
    reverb = Reverb()
    for line in fileinput.input():
        sentence = Sentence(line, "ORG", "ORG", 6, 1, 2, None)
        for r in sentence.relationships:
            pattern_tags = reverb.extract_reverb_patterns_tagged_ptb(r.between)
            # simple passive voice
            # auxiliary verb be + main verb past participle + 'by'
            print r.ent1, '\t', r.ent2
            print r.sentence
            print pattern_tags
            if reverb.detect_passive_voice(pattern_tags):
                print "Passive Voice: True"
            else:
                print "Passive Voice: False"
            print "\n"
    fileinput.close()

if __name__ == "__main__":
    main()