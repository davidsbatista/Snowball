__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

from typing import Any, List, Optional, Tuple

from snowball.reverb_breds import Reverb


class SnowballTuple:
    # pylint: disable=too-many-instance-attributes, too-many-arguments
    """
    Tuple class:

    see: http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
    select everything except stopwords, ADJ and ADV

    # construct TF-IDF vectors with the words part of a ReVerb pattern
    # or if no ReVerb patterns with selected words from the contexts

    """

    filter_pos = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"]

    def __init__(self, ent1: str, ent2: str, sentence: str, before: str, between: str, after: str, config: Any) -> None:
        self.ent1 = ent1
        self.ent2 = ent2
        self.sentence = sentence
        self.confidence = 0
        self.confidence_old = 0
        self.bef_words = before
        self.bet_words = between
        self.aft_words = after
        self.config = config
        self.bef_vector = None
        self.bet_vector = None
        self.aft_vector = None
        self.bef_reverb_vector = None
        self.bet_reverb_vector = None
        self.aft_reverb_vector = None
        self.passive_voice = None
        if config.use_reverb == "no":
            self.bef_vector = self.create_vector(self.bef_words) if self.bef_words else []
            self.bet_vector = self.create_vector(self.bet_words) if self.bet_words else []
            self.aft_vector = self.create_vector(self.aft_words) if self.aft_words else []
        else:
            self.extract_patterns()

    def __str__(self) -> str:
        return f"{self.bef_words}  {self.bet_words}  {self.aft_words}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SnowballTuple):
            return False
        return (
            self.ent1 == other.ent1
            and self.ent2 == other.ent2
            and self.bef_words == other.bef_words
            and self.bet_words == other.bet_words
            and self.aft_words == other.aft_words
        )

    def __hash__(self) -> int:
        return hash(self.ent1) ^ hash(self.ent2)

    def get_vector(self, context: str) -> Optional[List[Tuple[int, float]]]:
        """
        Return the vector for the given context
        """
        if context == "bef":
            return self.bef_vector
        if context == "bet":
            return self.bet_vector
        # ToDo: can only be "aft" here
        return self.aft_vector

    def create_vector(self, text: str) -> List[Tuple[int, float]]:
        """
        Create a TF-IDF vector for the given text, this is only applies when ReVerb is not used to extract patterns.
        """
        words, _ = zip(*text)
        tokens = [word.lower() for word in words if word not in self.config.stopwords]
        vect_ids = self.config.vsm.dictionary.doc2bow(tokens)
        return self.config.vsm.tf_idf_model[vect_ids]

    def construct_pattern_vector(self, pattern_tags: List[Tuple[str, str]]) -> List[Tuple[int, float]]:
        """
        Construct TF-IDF representation for each context
        """
        pattern = [
            t[0] for t in pattern_tags if t[0].lower() not in self.config.stopwords and t[1] not in self.filter_pos
        ]
        vect_ids = self.config.vsm.dictionary.doc2bow(pattern)
        return self.config.vsm.tf_idf_model[vect_ids]

    def construct_words_vectors(self, words):
        """
        Construct TF-IDF representation for each context
        """
        tokens, tags = zip(*words)
        pattern = [
            token
            for token, tag in zip(tokens, tags)
            if token.lower() not in self.config.stopwords and tag not in self.filter_pos
        ]
        vect_ids = self.config.vsm.dictionary.doc2bow(pattern)
        return self.config.vsm.tf_idf_model[vect_ids]

    def extract_patterns(self) -> None:
        """
        If a ReVerb pattern is found in the BET context it constructs a TF-IDF vector with the words part of the
        pattern, otherwise uses all words filtering stopwords, ADJ and ADV.

        For the BEF and AFT contexts it uses all words filtering stopwords, ADJ and ADV.
        """

        if patterns_bet_tags := Reverb.extract_reverb_patterns_tagged_ptb(self.bet_words):
            self.passive_voice = self.config.reverb.detect_passive_voice(patterns_bet_tags)
            # 's_ is always wrongly tagged as VBZ, if the first word is 's' ignore it
            if patterns_bet_tags[0][0] == "'s":
                self.bet_vector = self.construct_words_vectors(self.bet_words)
            else:
                self.bet_vector = self.construct_pattern_vector(patterns_bet_tags)
        else:
            self.bet_vector = self.construct_words_vectors(self.bet_words)

        # extract two words before the first entity, and two words after the second entity
        if len(self.bef_words) > 0:
            self.bef_vector = self.construct_words_vectors(self.bef_words)

        if len(self.aft_words) > 0:
            self.aft_vector = self.construct_words_vectors(self.aft_words)
