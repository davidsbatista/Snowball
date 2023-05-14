from snowball.seed import Seed


def test_seed_creation():
    s = Seed("a", "b")
    assert s.ent1 == "a"
    assert s.ent2 == "b"


def test_seed_equality():
    s1 = Seed("a", "b")
    s2 = Seed("a", "b")
    s3 = Seed("b", "a")
    assert s1 == s2
    assert not s1 == s3
    assert s1 != 1


def test_seed_hashing():
    s1 = Seed("a", "b")
    s2 = Seed("a", "b")
    s3 = Seed("b", "a")
    assert hash(s1) == hash(s2)
    assert hash(s1) != hash(s3)
