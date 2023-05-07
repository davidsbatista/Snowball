__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

from typing import Any


class Seed:
    def __init__(self, ent1: str, ent2: str) -> None:
        self.ent1 = ent1
        self.ent2 = ent2

    def __hash__(self) -> int:
        return hash(self.ent1) ^ hash(self.ent2) ^ hash((self.ent1, self.ent2))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Seed):
            return NotImplemented
        return self.ent1 == other.ent1 and self.ent2 == other.ent2
