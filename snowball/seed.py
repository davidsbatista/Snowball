__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"


class Seed(object):
    def __init__(self, _e1, _e2):
        self.ent1 = _e1
        self.ent2 = _e2

    def __hash__(self):
        return hash(self.ent1) ^ hash(self.ent2)

    def __eq__(self, other):
        return self.ent1 == other.ent1 and self.ent2 == other.ent2
