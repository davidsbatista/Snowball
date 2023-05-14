__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import operator
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from gensim.matutils import cossim
from nltk.data import load
from tqdm import tqdm

from snowball.commons import blocks
from snowball.config import Config
from snowball.pattern import Pattern
from snowball.seed import Seed
from snowball.sentence import Sentence
from snowball.snowball_tuple import SnowballTuple

PRINT_PATTERNS = False


class Snowball:
    def __init__(
        self,
        config_file: str,
        seeds_file: str,
        negative_seeds: str,
        sentences_file: str,
        similarity: float,
        confidence: float,
        n_iterations: int,
    ):
        # pylint: disable=too-many-arguments
        self.current_iteration: int = 0
        self.patterns: List[Pattern] = []
        self.processed_tuples: List[SnowballTuple] = []
        self.candidate_tuples: Dict[SnowballTuple, List[Tuple[Pattern, float]]] = defaultdict(list)
        self.config = Config(
            config_file, seeds_file, negative_seeds, sentences_file, similarity, confidence, n_iterations
        )

    def write_relationships_to_disk(self) -> None:
        """Write extracted relationships to disk"""
        # ToDo: write this in JSON format
        print("\nWriting extracted relationships to disk")
        with open("relationships.txt", "wt", encoding="utf8") as f_output:
            tmp = sorted(self.candidate_tuples, key=lambda out_tpl: out_tpl.confidence, reverse=True)
            for tpl in tmp:
                f_output.write("instance: " + tpl.ent1 + "\t" + tpl.ent2 + "\tscore:" + str(tpl.confidence) + "\n")
                f_output.write("sentence: " + tpl.sentence + "\n")
                if tpl.passive_voice is False or tpl.passive_voice is None:
                    f_output.write("passive voice: False\n")
                elif tpl.passive_voice is True:
                    f_output.write("passive voice: True\n")
                f_output.write("\n")

    def debug_patterns(self) -> None:
        """
        Print patterns to stdout
        """
        if PRINT_PATTERNS is True:
            print("\nPatterns:")
            for pattern in self.patterns:
                pattern.merge_tuple_patterns()
                print("Patterns:", len(pattern.tuples))
                print("Positive", pattern.positive)
                print("Negative", pattern.negative)
                print("Unknown", pattern.unknown)
                print("Tuples", len(pattern.tuples))
                print("Pattern Confidence", pattern.confidence)
                print("\n")

    def similarity(self, tpl: SnowballTuple, extraction_pattern: Pattern) -> float:
        """
        Calculate the similarity between a tuple and an extraction pattern
        """
        (bef, bet, aft) = (0, 0, 0)

        if tpl.bef_vector is not None and extraction_pattern.centroid_bef is not None:
            bef = cossim(tpl.bef_vector, extraction_pattern.centroid_bef)

        if tpl.bet_vector is not None and extraction_pattern.centroid_bet is not None:
            bet = cossim(tpl.bet_vector, extraction_pattern.centroid_bet)

        if tpl.aft_vector is not None and extraction_pattern.centroid_aft is not None:
            aft = cossim(tpl.aft_vector, extraction_pattern.centroid_aft)

        return self.config.alpha * bef + self.config.beta * bet + self.config.gamma * aft

    def cluster_tuples(self, matched_tuples: List[SnowballTuple]) -> None:
        """
        Cluster the matched instances: generate patterns/update patterns.
        Applies a single-pass clustering algorithm to cluster the matched instances.
        """
        start = 0
        # initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            self.patterns.append(Pattern(matched_tuples[0]))
            start = 1

        # compute the similarity between an instance with each pattern go through all tuples
        for i in range(start, len(matched_tuples), 1):
            tpl = matched_tuples[i]
            max_similarity: float = 0.0
            max_similarity_cluster_index: int = 0

            # go through all patterns(clusters of tuples) and find the one with the highest similarity score
            for pattern_idx in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[pattern_idx]
                score = self.similarity(tpl, extraction_pattern)
                if score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = pattern_idx

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                self.patterns.append(Pattern(tpl))

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(tpl)

    def _normalize_confidence(self) -> None:
        """
        Normalize patterns confidence, find the maximum value of confidence and divide all by the maximum value.
        """
        max_confidence: float = 0.0
        for pattern in self.patterns:
            if pattern.confidence > max_confidence:
                max_confidence = pattern.confidence
        if max_confidence > 0:
            for pattern in self.patterns:
                pattern.confidence = float(pattern.confidence) / float(max_confidence)

    def match_seeds_tuples(self) -> Tuple[Dict[Tuple[str, str], int], List[SnowballTuple]]:
        """
        Looks for sentences matching the seed instances, checks if an extracted tuple matches seeds tuples.
        """
        matched_tuples: List[SnowballTuple] = []
        count_matches: Dict[Tuple[str, str], int] = defaultdict(int)
        for tpl in self.processed_tuples:
            for seed in self.config.positive_seeds:
                if tpl.ent1 == seed.ent1 and tpl.ent2 == seed.ent2:
                    matched_tuples.append(tpl)
                    count_matches[(tpl.ent1, tpl.ent2)] += 1

        return count_matches, matched_tuples

    def generate_tuples(self, sentences_file: str) -> None:
        """
        Generate tuples instances from a text file with sentences where named entities are already tagged
        """
        if os.path.exists("processed_tuples.pkl"):
            with open("processed_tuples.pkl", "rb") as f_in:
                print("\nLoading processed tuples from disk...")
                self.processed_tuples = pickle.load(f_in)
                print(len(self.processed_tuples), "tuples loaded")
        else:
            print("\nGenerating relationship instances from sentences")
            tagger = load("taggers/maxent_treebank_pos_tagger/english.pickle")

            with open(sentences_file, "r", encoding="utf8") as f_in:
                total = sum(bl.count("\n") for bl in blocks(f_in))

            with open(sentences_file, encoding="utf-8") as f_sentences:
                for line in tqdm(f_sentences, total=total):
                    sentence = Sentence(
                        line.strip(),
                        self.config.e1_type,
                        self.config.e2_type,
                        self.config.max_tokens_away,
                        self.config.min_tokens_away,
                        self.config.context_window_size,
                        tagger,
                    )

                    for rel in sentence.relationships:
                        if rel.e1_type == self.config.e1_type and rel.e2_type == self.config.e2_type:
                            tpl = SnowballTuple(
                                rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config
                            )
                            self.processed_tuples.append(tpl)

            print(f"\n{len(self.processed_tuples)} relationships generated")
            print("Dumping relationships to file")
            with open("processed_tuples.pkl", "wb") as f_out:
                pickle.dump(self.processed_tuples, f_out)

    def _update_seeds(self) -> None:
        """
        Update seed set of tuples to use in next iteration:
            seeds = { T | Conf(T) > min_tuple_confidence }
        """
        if self.current_iteration + 1 < self.config.number_iterations:
            print("Adding tuples to seed with confidence =>" + str(self.config.instance_confidence))
            for seed_tpl in self.candidate_tuples.keys():
                if seed_tpl.confidence >= self.config.instance_confidence:
                    seed = Seed(seed_tpl.ent1, seed_tpl.ent2)
                    self.config.positive_seeds.add(seed)

    def init_bootstrap(self, tuples: Optional[str]) -> None:  # noqa: C901
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        """
        Starts a bootstrap iteration
        """
        if tuples is not None:
            with open(tuples, "rb") as f_in:
                print("Loading pre-processed sentences", tuples)
                self.processed_tuples = pickle.load(f_in)
                print(len(self.processed_tuples), "tuples loaded")

        while self.current_iteration <= self.config.number_iterations:
            print("\n=============================================")
            print("\nStarting iteration", self.current_iteration)
            print("\nLooking for seed matches of:")
            for seed in self.config.positive_seeds:
                print(f"{seed.ent1}\t{seed.ent2}")

            count_matches, matched_tuples = self.match_seeds_tuples()

            if len(matched_tuples) == 0:
                print("\nNo seed matches found")
                sys.exit(0)

            print("\nNumber of seed matches found")
            for tpl in sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True):
                print(f"{tpl[0][0]}\t{tpl[0][1]} {tpl[1]}")

            print("\nClustering matched instances to generate patterns")
            self.cluster_tuples(matched_tuples)

            # eliminate patterns supported by less than 'min_pattern_support' tuples
            self.patterns = [p for p in self.patterns if len(p.tuples) >= self.config.min_pattern_support]
            print("\n", len(self.patterns), "patterns generated")
            if self.current_iteration == 0 and len(self.patterns) == 0:
                print("No patterns generated")
                sys.exit(0)

            # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
            # This was already collect, and it's stored in: self.processed_tuples
            #
            # Measure the similarity of each occurrence with each extraction pattern
            # and store each pattern that has a similarity higher than a given threshold
            #
            # Each candidate tuple will then have a number of patterns that helped generate it,
            # each with an associated degree of match.
            print("\nCollecting instances based on extraction patterns")
            pattern_best = None

            for processed_tpl in tqdm(self.processed_tuples):
                sim_best: float = 0.0

                for extraction_pattern in self.patterns:
                    score = self.similarity(processed_tpl, extraction_pattern)
                    if score > self.config.threshold_similarity:
                        extraction_pattern.update_selectivity(processed_tpl, self.config)
                    if score > sim_best:
                        sim_best = score
                        pattern_best = extraction_pattern

                if sim_best >= self.config.threshold_similarity:
                    # if this instance was already extracted, check if it was by this extraction pattern
                    patterns = self.candidate_tuples[processed_tpl]
                    if patterns is not None and pattern_best not in [x[0] for x in patterns]:
                        self.candidate_tuples[processed_tpl].append((pattern_best, sim_best))  # type: ignore

                    # if this instance was not extracted before, associate this extraction pattern with the instance
                    # and the similarity score
                    else:
                        self.candidate_tuples[processed_tpl].append((pattern_best, sim_best))

                # update extraction pattern confidence
                extraction_pattern.confidence_old = (  # pylint: disable=undefined-loop-variable
                    extraction_pattern.confidence  # pylint: disable=undefined-loop-variable
                )
                extraction_pattern.update_confidence()  # pylint: disable=undefined-loop-variable

            self._normalize_confidence()
            self.debug_patterns()

            # update tuple confidence based on patterns confidence
            print("\nCalculating tuples confidence")
            for candidate_tpl in list(self.candidate_tuples.keys()):
                confidence: float = 1.0
                candidate_tpl.confidence_old = candidate_tpl.confidence
                for candidate_pattern_score in self.candidate_tuples[candidate_tpl]:
                    pattern = candidate_pattern_score[0]
                    score = candidate_pattern_score[1]
                    confidence *= 1 - (pattern.confidence * score)
                candidate_tpl.confidence = 1 - confidence

                # use past confidence values to calculate new confidence
                # if parameter Wupdt < 0.5 the system trusts new examples less on each iteration
                # which will lead to more conservative patterns and have a damping effect.
                if self.current_iteration > 0:
                    candidate_tpl.confidence = (
                        candidate_tpl.confidence * self.config.w_updt
                        + candidate_tpl.confidence_old * (1 - self.config.w_updt)
                    )

            self._update_seeds()

            # increment the number of iterations
            self.current_iteration += 1

        self.write_relationships_to_disk()
