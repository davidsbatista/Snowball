__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import re
from typing import Any, Dict, List, Set, Tuple

from nltk import word_tokenize
from nltk.corpus import stopwords

# tokens between entities which do not represent relationships
bad_tokens = [",", "(", ")", ";", "''", "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords = stopwords.words("english")
not_valid = bad_tokens + stopwords
regex_clean_tags = re.compile("</?[A-Z]+>", re.U)


def tokenize_entity(entity: str) -> List[str]:
    """Simple poor man's tokenization of an entity string"""
    parts = word_tokenize(entity)
    if parts[-1] == ".":
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(entity_string: str, text_tokens: List[str]) -> Tuple[List[str], List[int]]:
    """Find the locations of an entity in a text."""
    locations = []
    ent_parts = tokenize_entity(entity_string)
    for idx in range(len(text_tokens)):
        if text_tokens[idx : idx + len(ent_parts)] == ent_parts:
            locations.append(idx)
    return ent_parts, locations


class Entity:
    """Entity class to hold information about an entity extracted from a sentence."""

    def __init__(
        self, surface_string: str, surface_string_parts: List[str], ent_type: str, locations: List[int]
    ) -> None:
        self.string = surface_string
        self.parts = surface_string_parts
        self.type = ent_type
        self.locations = locations

    def __hash__(self) -> int:
        return hash(self.string) ^ hash(self.type)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.string == other.string and self.type == other.type


class Relationship:  # pylint: disable=too-many-arguments, too-many-instance-attributes
    """Relationship class to hold information about a relationship extracted from a sentence."""

    def __init__(
        self,
        sentence: str,
        before: List[Tuple[str, str]],
        between: List[Tuple[str, str]],
        after: List[Tuple[str, str]],
        ent1_str: str,
        ent2_str: str,
        e1_type: str,
        e2_type: str,
    ) -> None:
        self.sentence = sentence
        self.before = before
        self.between = between
        self.after = after
        self.ent1 = ent1_str
        self.ent2 = ent2_str
        self.e1_type = e1_type
        self.e2_type = e2_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relationship):
            return NotImplemented
        return (
            self.ent1 == other.ent1
            and self.before == other.before
            and self.between == other.between
            and self.after == other.after
        )

    def __hash__(self) -> int:
        return hash(self.ent1) ^ hash(self.ent2) ^ hash(self.before) ^ hash(self.between) ^ hash(self.after)


class Sentence:  # pylint: disable=too-few-public-methods, too-many-locals, too-many-arguments
    """Holds information about a sentence extracted from a document."""

    def __init__(
        self,
        sentence: str,
        e1_type: str,
        e2_type: str,
        max_tokens: int,
        min_tokens: int,
        window_size: int,
        pos_tagger: Any = None,
    ):  # noqa: C901
        self.relationships = []
        self.tagged_text = None
        self.entities_regex = re.compile("<[A-Z]+>[^<]+</[A-Z]+>", re.U)
        entities = list(re.finditer(self.entities_regex, sentence))

        if len(entities) >= 2:
            sentence_no_tags = re.sub(regex_clean_tags, "", sentence)  # clean tags from text
            text_tokens = word_tokenize(sentence_no_tags)

            # extract information about the entity, create an Entity instance
            # and store in a structure to hold information collected about
            # all the entities in the sentence
            entities_info: Set[Entity] = set()
            for ent in entities:
                entity = ent.group()
                e_string = re.findall("<[A-Z]+>([^<]+)</[A-Z]+>", entity)[0]
                e_type = re.findall("<([A-Z]+)", entity)[0]
                e_parts, ent_locations = find_locations(e_string, text_tokens)
                entities_info.add(Entity(e_string, e_parts, e_type, ent_locations))

            # create a hash table:
            #   key: is the starting index in the tokenized sentence of an entity
            #   value: the corresponding Entity instance
            locations: Dict[int, Entity] = {
                start: entity_obj for entity_obj in entities_info for start in entity_obj.locations
            }

            # look for a pair of entities such that:
            # the distance between the two entities is less than 'max_tokens'
            # and greater than 'min_tokens'
            # the arguments match the seeds semantic types
            sorted_keys = list(sorted(locations))

            for i in range(len(sorted_keys) - 1):
                distance = sorted_keys[i + 1] - sorted_keys[i]
                ent1 = locations[sorted_keys[i]]
                ent2 = locations[sorted_keys[i + 1]]

                # ignore relationships between the same entity
                if max_tokens >= distance >= min_tokens and ent1.type == e1_type and ent2.type == e2_type:
                    if ent1.string == ent2.string:
                        continue

                    # run PoS-tagger over the sentence only once
                    if self.tagged_text is None:
                        # split text into tokens and tag them using NLTK's default English tagger
                        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/
                        # english.pickle'
                        self.tagged_text = pos_tagger.tag(text_tokens)

                    before = self.tagged_text[: sorted_keys[i]]
                    before = before[-window_size:]
                    between = self.tagged_text[sorted_keys[i] + len(ent1.parts) : sorted_keys[i + 1]]
                    after = self.tagged_text[sorted_keys[i + 1] + len(ent2.parts) :]
                    after = after[:window_size]

                    # ignore relationships where BET context is only stopwords or other invalid words
                    if all(x in not_valid for x in text_tokens[sorted_keys[i] + len(ent1.parts) : sorted_keys[i + 1]]):
                        continue

                    rel = Relationship(sentence, before, between, after, ent1.string, ent2.string, e1_type, ent2.type)
                    self.relationships.append(rel)
