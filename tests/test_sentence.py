from snowball.sentence import Entity


def test_entity_hashing():
    entity1 = Entity("example", ["example"], "ORG", [0, 1])
    entity2 = Entity("example", ["example"], "ORG", [0, 1])
    assert hash(entity1) == hash(entity2)


def test_entity_equality():
    entity1 = Entity("example", ["example"], "LOC", [0, 1])
    entity2 = Entity("example", ["example"], "LOC", [0, 1])
    assert entity1 == entity2


def test_entity_inequality():
    entity1 = Entity("example", ["example"], "ORG", [0, 1])
    entity2 = Entity("example", ["example"], "LOC", [0, 1])
    assert entity1 != entity2
