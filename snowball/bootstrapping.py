__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import codecs
import operator
import os
import pickle
import sys
from collections import defaultdict

from gensim.matutils import cossim
from nltk import word_tokenize

from snowball.config import Config
from snowball.pattern import Pattern
from snowball.seed import Seed
from snowball.sentence import Sentence
from snowball.snowball_tuple import SnowballTuple

PRINT_PATTERNS = False


class Snowball:
    def __init__(self, config_file, seeds_file, negative_seeds, sentences_file, similarity, confidence):
        # pylint: disable=too-many-arguments
        self.patterns = []
        self.processed_tuples = []
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file, negative_seeds, sentences_file, similarity, confidence)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences where named entities are already tagged
        """
        try:
            os.path.isfile("processed_tuples.pkl")
            with open("processed_tuples.pkl", "rb") as f:
                print("\nLoading processed tuples from disk...")
                self.processed_tuples = pickle.load(f)
                print(len(self.processed_tuples), "tuples loaded")

            # f = open("processed_tuples.pkl", "rb")
            # print("\nLoading processed tuples from disk...")
            # self.processed_tuples = pickle.load(f)
            # f.close()
            # print(len(self.processed_tuples), "tuples loaded")

        except IOError:
            print("\nGenerating relationship instances from sentences")
            with codecs.open(sentences_file, encoding="utf-8") as f_sentences:
                for line in f_sentences:
                    sentence = Sentence(
                        line.strip(),
                        self.config.e1_type,
                        self.config.e2_type,
                        self.config.max_tokens_away,
                        self.config.min_tokens_away,
                        self.config.context_window_size,
                    )

                    for rel in sentence.relationships:
                        if rel.arg1type == self.config.e1_type and rel.arg2type == self.config.e2_type:
                            bef_tokens = word_tokenize(rel.before)
                            bet_tokens = word_tokenize(rel.between)
                            aft_tokens = word_tokenize(rel.after)
                            if not (bef_tokens == 0 and bet_tokens == 0 and aft_tokens == 0):
                                t = SnowballTuple(
                                    rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config
                                )
                                self.processed_tuples.append(t)

            print("\n", len(self.processed_tuples), "relationships generated")
            print("Dumping relationships to file")
            with open("processed_tuples.pkl", "wb") as f:
                pickle.dump(self.processed_tuples, f)

    def init_bootstrap(self, tuples):
        # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-branches, too-many-statements
        """
        Starts a bootstrap iteration
        """
        if tuples is not None:
            with open(tuples, "rb") as f:
                print("Loading pre-processed sentences", tuples)
                self.processed_tuples = pickle.load(f)
                print(len(self.processed_tuples), "tuples loaded")

        i = 0
        while i <= self.config.number_iterations:
            print("\n=============================================")
            print("\nStarting iteration", i)
            print("\nLooking for seed matches of:")
            for s in self.config.seed_tuples:
                print(s.ent1, "\t", s.ent2)

            # Looks for sentences matching the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples(self)

            if len(matched_tuples) == 0:
                print("\nNo seed matches found")
                sys.exit(0)

            else:
                print("\nNumber of seed matches found")
                sorted_counts = sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True)
                for t in sorted_counts:
                    print(t[0][0], "\t", t[0][1], t[1])

                # Cluster the matched instances: generate patterns/update patterns
                print("\nClustering matched instances to generate patterns")
                self.cluster_tuples(self, matched_tuples)

                # Eliminate patterns supported by less than 'min_pattern_support' tuples
                new_patterns = [p for p in self.patterns if len(p.tuples) >= self.config.min_pattern_support]
                self.patterns = new_patterns
                print("\n", len(self.patterns), "patterns generated")
                if i == 0 and len(self.patterns) == 0:
                    print("No patterns generated")
                    sys.exit(0)

                # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
                # This was already collect and its stored in: self.processed_tuples
                #
                # Measure the similarity of each occurrence with each extraction pattern
                # and store each pattern that has a similarity higher than a given threshold
                #
                # Each candidate tuple will then have a number of patterns that helped generate it,
                # each with an associated de gree of match. Snowball uses this infor
                print("\nCollecting instances based on extraction patterns")
                count = 0
                pattern_best = None
                for t in self.processed_tuples:
                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    sim_best = 0
                    for extraction_pattern in self.patterns:
                        score = self.similarity(t, extraction_pattern)
                        if score > self.config.threshold_similarity:
                            extraction_pattern.update_selectivity(t, self.config)
                        if score > sim_best:
                            sim_best = score
                            pattern_best = extraction_pattern

                    if sim_best >= self.config.threshold_similarity:
                        # if this instance was already extracted, check if it was by this extraction pattern
                        patterns = self.candidate_tuples[t]
                        if patterns is not None:
                            if pattern_best not in [x[0] for x in patterns]:
                                self.candidate_tuples[t].append((pattern_best, sim_best))

                        # if this instance was not extracted before, associate this extraction pattern with the instance
                        # and the similarity score
                        else:
                            self.candidate_tuples[t].append((pattern_best, sim_best))

                    # update extraction pattern confidence
                    extraction_pattern.confidence_old = extraction_pattern.confidence
                    extraction_pattern.update_confidence()

                # normalize patterns confidence
                # find the maximum value of confidence and divide all by the maximum
                max_confidence = 0
                for p in self.patterns:
                    if p.confidence > max_confidence:
                        max_confidence = p.confidence

                if max_confidence > 0:
                    for p in self.patterns:
                        p.confidence = float(p.confidence) / float(max_confidence)

                if PRINT_PATTERNS is True:
                    print("\nPatterns:")
                    for p in self.patterns:
                        p.merge_tuple_patterns()
                        print("Patterns:", len(p.tuples))
                        print("Positive", p.positive)
                        print("Negative", p.negative)
                        print("Unknown", p.unknown)
                        print("Tuples", len(p.tuples))
                        print("Pattern Confidence", p.confidence)
                        print("\n")

                # update tuple confidence based on patterns confidence
                print("\nCalculating tuples confidence")
                for t in self.candidate_tuples.keys():
                    confidence = 1
                    t.confidence_old = t.confidence
                    for p in self.candidate_tuples.get(t):
                        confidence *= 1 - (p[0].confidence * p[1])
                    t.confidence = 1 - confidence

                    # use past confidence values to calculate new confidence
                    # if parameter Wupdt < 0.5 the system trusts new examples less on each iteration
                    # which will lead to more conservative patterns and have a damping effect.
                    if i > 0:
                        t.confidence = t.confidence * self.config.w_updt + t.confidence_old * (1 - self.config.w_updt)

                # update seed set of tuples to use in next iteration
                # seeds = { T | Conf(T) > min_tuple_confidence }
                if i + 1 < self.config.number_iterations:
                    print("Adding tuples to seed with confidence =>" + str(self.config.instance_confidence))
                    for t in self.candidate_tuples.keys():
                        if t.confidence >= self.config.instance_confidence:
                            seed = Seed(t.ent1, t.ent2)
                            self.config.seed_tuples.add(seed)

                # increment the number of iterations
                i += 1

        print("\nWriting extracted relationships to disk")
        with open("relationships.txt", "wt", encoding="utf8") as f_output:
            tmp = sorted(self.candidate_tuples, key=lambda tpl: tpl.confidence, reverse=True)
            for t in tmp:
                f_output.write("instance: " + t.ent1 + "\t" + t.ent2 + "\tscore:" + str(t.confidence) + "\n")
                f_output.write("sentence: " + t.sentence + "\n")
                if t.passive_voice is False or t.passive_voice is None:
                    f_output.write("passive voice: False\n")
                elif t.passive_voice is True:
                    f_output.write("passive voice: True\n")
                f_output.write("\n")

    def similarity(self, tpl, extraction_pattern):
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

    @staticmethod
    def cluster_tuples(self, matched_tuples):
        """
        single-pass clustering
        """
        start = 0
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)
            start = 1

        # Compute the similarity between an instance with each pattern go through all tuples
        for i in range(start, len(matched_tuples), 1):
            t = matched_tuples[i]
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the
            # highest similarity score
            for w in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[w]
                score = self.similarity(t, extraction_pattern)
                if score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = w

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)

    @staticmethod
    def match_seeds_tuples(self):
        """
        checks if an extracted tuple matches seeds tuples
        """
        matched_tuples = []
        count_matches = {}
        for t in self.processed_tuples:
            for s in self.config.seed_tuples:
                if t.ent1 == s.ent1 and t.ent2 == s.ent2:
                    matched_tuples.append(t)
                    try:
                        count_matches[(t.ent1, t.ent2)] += 1
                    except KeyError:
                        count_matches[(t.ent1, t.ent2)] = 1

        return count_matches, matched_tuples


def main():
    if len(sys.argv) != 7:
        print(
            "\nSnowball.py paramters.cfg sentences_file seeds_file_positive seeds_file_negative similarity_threshold"
            " confidance_threshold\n"
        )
        sys.exit(0)
    else:
        configuration = sys.argv[1]
        sentences_file = sys.argv[2]
        seeds_file = sys.argv[3]
        negative_seeds = sys.argv[4]
        similarity = sys.argv[5]
        confidence = sys.argv[6]

        snowball = Snowball(
            configuration, seeds_file, negative_seeds, sentences_file, float(similarity), float(confidence)
        )
        if sentences_file.endswith(".pkl"):
            snowball.init_bootstrap(tuples=sentences_file)
        else:
            snowball.generate_tuples(sentences_file)
            snowball.init_bootstrap(tuples=None)


if __name__ == "__main__":
    main()
