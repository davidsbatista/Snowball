__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import sys
from copy import deepcopy
from math import log
from typing import Any, List, Optional, Set, Tuple

from snowball.config import Config
from snowball.snowball_tuple import SnowballTuple


class Pattern:
    # pylint: disable=too-many-instance-attributes
    """
    A pattern is a set of tuples that is used to extract relationships between named-entities.
    """

    def __init__(self, tpl: Optional[SnowballTuple]) -> None:
        self.positive: int = 0
        self.negative: int = 0
        self.unknown: int = 0
        self.confidence_old: float = 0.0
        self.confidence: float = 0.0
        self.tuples: List[SnowballTuple] = []
        self.tuple_patterns: Set[Any] = set()
        self.centroid_bef: Optional[List[Tuple[int, float]]] = []
        self.centroid_bet: Optional[List[Tuple[int, float]]] = []
        self.centroid_aft: Optional[List[Tuple[int, float]]] = []
        if tpl is not None:
            self.tuples.append(tpl)
            self.centroid_bef = tpl.bef_vector
            self.centroid_bet = tpl.bet_vector
            self.centroid_aft = tpl.aft_vector

    def __str__(self) -> str:
        output = ""
        for tpl in self.tuples:
            output += str(tpl) + "|"
        return output

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Pattern):
            return False
        return set(self.tuples) == set(other.tuples)

    def update_confidence_2003(self, config: "Config") -> None:
        """
        Update the confidence of the pattern
        """
        if self.positive > 0:
            self.confidence = log(float(self.positive), 2) * (
                float(self.positive) / float(self.positive + self.unknown * config.w_unk + self.negative * config.w_neg)
            )
        elif self.positive == 0:
            self.confidence = 0

    def update_confidence(self) -> None:
        """
        Update the confidence of the pattern
        """
        if self.positive > 0 or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, tpl: SnowballTuple) -> None:
        """
        Add another tuple to be used to generate the pattern
        """
        self.tuples.append(tpl)
        self.updated_centroid()

    def update_selectivity(self, tpl: SnowballTuple, config: Config) -> None:
        """
        Update the selectivity of the pattern
        """
        for seed in config.positive_seeds:
            if seed.ent1 == tpl.ent1 or seed.ent1.strip() == tpl.ent1.strip():
                if seed.ent2 == tpl.ent2.strip() or seed.ent2.strip() == tpl.ent2.strip():
                    self.positive += 1
                else:
                    self.negative += 1
            else:
                for neg_seed in config.negative_seeds:
                    if neg_seed.ent1 == tpl.ent1 or neg_seed.ent1.strip() == tpl.ent1.strip():
                        if neg_seed.ent2 == tpl.ent2.strip() or neg_seed.ent2.strip() == tpl.ent2.strip():
                            self.negative += 1
                self.unknown += 1

        # self.update_confidence()
        self.update_confidence_2003(config)

    def merge_tuple_patterns(self) -> None:
        """
        Merge all tuple patterns into one
        """
        # ToDo: fazer o merge tendo em consideração todos os contextos
        for tpl in self.tuples:
            self.tuple_patterns.add(tpl.bet_words)

    def updated_centroid(self) -> None:
        """
        Calculate the centroid of a pattern, based on the tuples associated with it.

        If there is just one tuple associated with this pattern, the centroid is the tuple itself. Otherwise,
        the centroid is the average of all tuples associated with this pattern.
        """
        if len(self.tuples) == 1:
            self.centroid_bef = self.tuples[0].bef_vector
            self.centroid_bet = self.tuples[0].bet_vector
            self.centroid_aft = self.tuples[0].aft_vector
        else:
            self.centroid_bef = self.calculate_centroid("bef")
            self.centroid_bet = self.calculate_centroid("bet")
            self.centroid_aft = self.calculate_centroid("aft")

    def calculate_centroid(self, context: str) -> Any:  # noqa: C901
        # pylint: disable=too-many-nested-blocks
        """
        Calculate the centroid of a pattern
        """
        # ToDo: refactor this method
        centroid = deepcopy(self.tuples[0].get_vector(context))
        if centroid is not None:
            # add all other words from other tuples
            for tpl in range(1, len(self.tuples), 1):
                current_words = [e[0] for e in centroid]
                words = self.tuples[tpl].get_vector(context)
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
                                    centroid[i] = tuple(w_new)  # type: ignore
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
                    print("Error calculating extraction pattern centroid")
                    sys.exit(0)
                centroid[i] = tuple(tmp)  # type: ignore

        return centroid
