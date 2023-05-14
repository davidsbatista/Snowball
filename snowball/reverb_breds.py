__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import io
from typing import Any, List, Tuple

from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.mapping import map_tag


class Reverb:
    """
    An implementation of:

    "Identifying Relations for Open Information Extraction" (Anthony Fader, Stephen Soderland, and Oren Etzioni)

    https://aclanthology.org/D11-1142.pdf
    """

    def __init__(self) -> None:
        self.lmtzr = WordNetLemmatizer()
        self.aux_verbs = ["be"]

    @staticmethod
    def extract_reverb_patterns(text: str) -> Tuple[List[str], List[List[Tuple[Any, Any]]]]:
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
        text_tokens = word_tokenize(text)

        # tag the sentence, using the default NTLK English tagger
        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        tags_ptb = pos_tag(text_tokens)

        # convert the tags to reduced tag set (Petrov et al. 2012) http://arxiv.org/pdf/1104.2086.pdf
        tags = []
        for tmp_tag in tags_ptb:
            tag = map_tag("en-ptb", "universal", tmp_tag[1])
            tags.append((tmp_tag[0], tag))

        patterns = []
        patterns_tags = []
        i = 0
        limit = len(tags) - 1

        while i <= limit:
            tmp = io.StringIO()
            tmp_tags = []

            # a ReVerb pattern always starts with a verb
            if tags[i][1] == "VERB":
                tmp.write(tags[i][0] + " ")
                tmp_tag = (tags[i][0], tags[i][1])
                tmp_tags.append(tmp_tag)
                i += 1

                # V = verb particle? adv? (also capture auxiliary verbs)
                while i <= limit and tags[i][1] in ["VERB", "PRT", "ADV"]:
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1

                # W = (noun | adj | adv | pron | det)
                while i <= limit and tags[i][1] in ["NOUN", "ADJ", "ADV", "PRON", "DET"]:
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1

                # P = (prep | particle | inf. marker)
                while i <= limit and tags[i][1] in ["ADP", "PRT"]:
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1
                # add the build pattern to the list collected patterns
                patterns.append(tmp.getvalue())
                patterns_tags.append(tmp_tags)
            i += 1

        return patterns, patterns_tags

    @staticmethod
    def extract_reverb_patterns_tagged_ptb(tagged_text: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        # pylint: disable=too-many-locals
        """
        Extract ReVerb relational patterns
        http://homes.cs.washington.edu/~afader/bib_pdf/emnlp11.pdf

        The pattern limits the relation to be:
            a verb (e.g., invented),
            a verb followed immediately by a preposition (e.g., located in),
            or a verb followed by nouns, adjectives, or adverbs ending in a preposition (e.g., has an atomic weight of).

        V | V P | V W*P
        V = verb particle? adv?
        W = (noun | adj | adv | pron | det)
        P = (prep | particle | inf. marker)
        """

        patterns = []
        patterns_tags = []
        i = 0
        limit = len(tagged_text) - 1
        tags = tagged_text

        verb = ["VB", "VBD", "VBD|VBN", "VBG", "VBG|NN", "VBN", "VBP", "VBP|TO", "VBZ", "VP"]
        adverb = ["RB", "RBR", "RBS", "RB|RP", "RB|VBG", "WRB"]
        particule = ["POS", "PRT", "TO", "RP"]
        noun = ["NN", "NNP", "NNPS", "NNS", "NN|NNS", "NN|SYM", "NN|VBG", "NP"]
        adjective = ["JJ", "JJR", "JJRJR", "JJS", "JJ|RB", "JJ|VBG"]
        pronoun = ["WP", "WP$", "PRP", "PRP$", "PRP|VBP"]
        determiner = ["DT", "EX", "PDT", "WDT"]
        adp = ["IN", "IN|RP"]

        # TODO: detect negations
        # ('rejected', 'VBD'), ('a', 'DT'), ('takeover', 'NN')

        while i <= limit:
            tmp = io.StringIO()
            tmp_tags = []

            # a ReVerb pattern always starts with a verb
            if tags[i][1] in verb:
                tmp.write(tags[i][0] + " ")
                tmp_tag = (tags[i][0], tags[i][1])
                tmp_tags.append(tmp_tag)
                i += 1

                # V = verb particle? adv? (also capture auxiliary verbs)
                while i <= limit and (tags[i][1] in verb or tags[i][1] in adverb or tags[i][1] in particule):
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1

                # W = (noun | adj | adv | pron | det)
                while i <= limit and (
                    tags[i][1] in noun
                    or tags[i][1] in adjective
                    or tags[i][1] in adverb
                    or tags[i][1] in pronoun
                    or tags[i][1] in determiner
                ):
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1

                # P = (prep | particle | inf. marker)
                while i <= limit and (tags[i][1] in adp or tags[i][1] in particule):
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1

                # add the build pattern to the list collected patterns
                patterns.append(tmp.getvalue())
                patterns_tags.append(tmp_tags)
            i += 1

        # if the pattern matches multiple adjacent sequences merge them into a single relation phrase, e.g.:
        # "wants to extend", enabling the model to readily handle relation phrases containing multiple verbs.
        merged_patterns_tags = [item for sublist in patterns_tags for item in sublist]
        return merged_patterns_tags

    @staticmethod
    def extract_reverb_patterns_ptb(text: str) -> List[Tuple[Any, Any]]:  # pylint: disable=too-many-locals
        """
        Extract ReVerb relational patterns from raw text.

        Part-of-speech tagging is performed using the default NTLK English tagger.
        """

        # The pattern limits the relation to be a verb (e.g., invented),
        # a verb followed immediately by a preposition (e.g., located in),
        # or a verb followed by nouns, adjectives, or adverbs ending in a
        # preposition (e.g., has an atomic weight of).

        # V | V P | V W*P
        # V = verb particle? adv?
        # W = (noun | adj | adv | pron | det)
        # P = (prep | particle | inf. marker)

        # split text into tokens
        text_tokens = word_tokenize(text)

        # tag the sentence, using the default NTLK English tagger
        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        tags_ptb = pos_tag(text_tokens)
        patterns = []
        patterns_tags = []
        i = 0
        limit = len(tags_ptb) - 1
        tags = tags_ptb

        verb = ["VB", "VBD", "VBD|VBN", "VBG", "VBG|NN", "VBN", "VBP", "VBP|TO", "VBZ", "VP"]
        adverb = ["RB", "RBR", "RBS", "RB|RP", "RB|VBG", "WRB"]
        particule = ["POS", "PRT", "TO", "RP"]
        noun = ["NN", "NNP", "NNPS", "NNS", "NN|NNS", "NN|SYM", "NN|VBG", "NP"]
        adjectiv = ["JJ", "JJR", "JJRJR", "JJS", "JJ|RB", "JJ|VBG"]
        pronoun = ["WP", "WP$", "PRP", "PRP$", "PRP|VBP"]
        determiner = ["DT", "EX", "PDT", "WDT"]
        adp = ["IN", "IN|RP"]

        # match is chosen.

        while i <= limit:
            tmp = io.StringIO()
            tmp_tags = []

            # a ReVerb pattern always starts with a verb
            if tags[i][1] in verb:
                tmp.write(tags[i][0] + " ")
                tmp_tag = (tags[i][0], tags[i][1])
                tmp_tags.append(tmp_tag)
                i += 1

                # V = verb particle? adv? (also capture auxiliary verbs)
                while i <= limit and (tags[i][1] in verb or tags[i][1] in adverb or tags[i][1] in particule):
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1

                # W = (noun | adj | adv | pron | det)
                while i <= limit and (
                    tags[i][1] in noun
                    or tags[i][1] in adjectiv
                    or tags[i][1] in adverb
                    or tags[i][1] in pronoun
                    or tags[i][1] in determiner
                ):
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1

                # P = (prep | particle | inf. marker)
                while i <= limit and (tags[i][1] in adp or tags[i][1] in particule):
                    tmp.write(tags[i][0] + " ")
                    tmp_tag = (tags[i][0], tags[i][1])
                    tmp_tags.append(tmp_tag)
                    i += 1

                # add the build pattern to the list collected patterns
                patterns.append(tmp.getvalue())
                patterns_tags.append(tmp_tags)
            i += 1

        # Finally, if the pattern matches multiple adjacent sequences, we merge
        # them into a single relation phrase (e.g.,wants to extend).
        # This refinement enables the model to readily handle relation
        # phrases containing multiple verbs.

        merged_patterns_tags = [item for sublist in patterns_tags for item in sublist]
        return merged_patterns_tags

    def detect_passive_voice(self, pattern: List[Tuple[Any, Any]]) -> bool:
        """Detect if the passive voice is present in a pattern"""
        passive_voice = False

        # TODO: there more complex exceptions, adjectives or adverbs in between
        # (to be) + (adj|adv) + past_verb + by
        # to be + past verb + by

        if len(pattern) >= 3:
            if pattern[0][1].startswith("V"):
                verb = self.lmtzr.lemmatize(pattern[0][0], "v")
                if verb in self.aux_verbs:
                    if (pattern[1][1] == "VBN" or pattern[1][1] == "VBD") and pattern[-1][0] == "by":
                        passive_voice = True

                    # past verb + by
                    elif (pattern[-2][1] == "VBN" or pattern[-2][1] == "VBD") and pattern[-1][0] == "by":
                        passive_voice = True

                # past verb + by
                elif (pattern[-2][1] == "VBN" or pattern[-2][1] == "VBD") and pattern[-1][0] == "by":
                    passive_voice = True

        # past verb + by
        elif len(pattern) >= 2:
            if (pattern[-2][1] == "VBN" or pattern[-2][1] == "VBD") and pattern[-1][0] == "by":
                passive_voice = True

        return passive_voice
