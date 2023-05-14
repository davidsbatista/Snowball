__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import fileinput
import os
import pickle
from typing import Any, Set

from nltk.corpus import stopwords

from snowball.reverb_breds import Reverb
from snowball.seed import Seed
from snowball.vector_space_model import VectorSpaceModel


class Config:
    # pylint: disable=too-many-instance-attributes
    """
    Configuration class
    """

    def __init__(  # noqa: C901
        self,
        config_file: str,
        positive_seeds: str,
        negative_seeds: str,
        sentences_file: str,
        similarity: float,
        confidence: float,
        n_iterations: int,
    ) -> None:  # noqa: C901
        # pylint: disable=too-many-arguments, too-many-statements
        if config_file is None:
            self.context_window_size: int = 2
            self.min_tokens_away: int = 1
            self.max_tokens_away: int = 6
            self.similarity: float = 0.6
            self.alpha: float = 0.0
            self.beta: float = 1.0
            self.gamma: float = 0.0
            self.min_pattern_support: int = 4
            self.w_neg: float = 2
            self.w_unk: float = 0.0
            self.w_updt: float = 0.5
            self.use_reverb: bool = True
        else:
            self.read_config(config_file)
        self.positive_seeds: Set[Seed] = set()
        self.negative_seeds: Set[Seed] = set()
        self.e1_type: str
        self.e2_type: str
        self.stopwords: Set[str] = set(stopwords.words("english"))
        self.threshold_similarity: float = similarity
        self.instance_confidence: float = confidence
        self.reverb: "Reverb" = Reverb()
        self.number_iterations = n_iterations
        self.read_seeds(positive_seeds, self.positive_seeds)
        if negative_seeds:
            self.read_seeds(negative_seeds, self.negative_seed_tuples)

        print("\nConfiguration parameters")
        print("========================")
        print("e1 type              :", self.e1_type)
        print("e2 type              :", self.e2_type)
        print("context window       :", self.context_window_size)
        print("max tokens away      :", self.max_tokens_away)
        print("min tokens away      :", self.min_tokens_away)
        print("use ReVerb           :", self.use_reverb)
        print("")
        print("alpha                :", self.alpha)
        print("beta                 :", self.beta)
        print("gamma                :", self.gamma)
        print("")
        print("positive seeds       :", len(self.positive_seeds))
        print("negative seeds       :", len(self.negative_seeds))
        print("negative seeds wNeg  :", self.w_neg)
        print("unknown seeds wUnk   :", self.w_unk)
        print("")
        print("threshold_similarity :", self.threshold_similarity)
        print("instance confidence  :", self.instance_confidence)
        print("min_pattern_support  :", self.min_pattern_support)
        print("iterations           :", self.number_iterations)
        print("iteration wUpdt      :", self.w_updt)
        print("\n")

        if os.path.exists("vsm.pkl"):
            print("\nLoading TF-IDF model from disk...")
            with open("vsm.pkl", "rb") as f_in:
                self.vsm = pickle.load(f_in)
        else:
            print("\nGenerating tf-idf model from sentences...")
            self.vsm = VectorSpaceModel(sentences_file, self.stopwords)
            with open("vsm.pkl", "wb") as f_out:
                pickle.dump(self.vsm, f_out)

    def read_seeds(self, seeds_file: str, holder: Set[Any]) -> None:
        """
        Reads the seeds file and adds the seeds to the holder.
        """
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                ent1 = line.split(";")[0].strip()
                ent2 = line.split(";")[1].strip()
                seed = Seed(ent1, ent2)
                holder.add(seed)

    def read_config(self, config_file: str) -> None:  # noqa: C901
        # pylint: disable=too-many-branches
        """
        Reads the configuration file and sets the parameters.
        """

        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.w_updt = float(line.split("=")[1])

            if line.startswith("wUnk"):
                self.w_unk = float(line.split("=")[1])

            if line.startswith("wNeg"):
                self.w_neg = float(line.split("=")[1])

            if line.startswith("use_reverb"):
                self.use_reverb = True

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("similarity"):
                self.similarity = float(line.split("=")[1].strip())

            if line.startswith("alpha"):
                self.alpha = float(line.split("=")[1])

            if line.startswith("beta"):
                self.beta = float(line.split("=")[1])

            if line.startswith("gamma"):
                self.gamma = float(line.split("=")[1])

        fileinput.close()
        if self.alpha + self.beta + self.gamma != 1:
            raise (ValueError(print("alpha + beta + gamma != 1")))
