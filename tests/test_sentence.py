import pytest

from snowball.sentence import Entity, Relationship, tokenize_entity


@pytest.fixture
def mock_word_tokenize(monkeypatch):
    def mock_tokenize(entity):
        return entity.split()  # Simulate word_tokenize behavior with simple splitting

    monkeypatch.setattr("snowball.sentence.word_tokenize", mock_tokenize)


def test_tokenize_entity(mock_word_tokenize):
    entity = "Hello, world!"
    expected_tokens = ["Hello,", "world!"]
    assert tokenize_entity(entity) == expected_tokens


def test_tokenize_entity_with_period(mock_word_tokenize):
    entity = "I am."
    expected_tokens = ["I", "am."]
    assert tokenize_entity(entity) == expected_tokens


def test_tokenize_entity_with_multiple_spaces(mock_word_tokenize):
    entity = "This    is      a     test."
    expected_tokens = ["This", "is", "a", "test."]
    assert tokenize_entity(entity) == expected_tokens


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


def test_relationship():
    sentence = "The tech company Soundcloud is based in Berlin, capital of Germany."
    before = [("The", "POS"), ("tech", "POS"), ("company", "POS")]
    between = [("is", "POS"), ("based", "POS"), ("in", "POS")]
    after = [("capital", "POS"), ("of", "POS"), ("Germany", "POS")]
    ent1_str = "Soundcloud"
    ent2_str = "Berlin"
    e1_type = "ORG"
    e2_type = "LOC"
    rel1 = Relationship(sentence, before, between, after, ent1_str, ent2_str, e1_type, e2_type)
    assert rel1 == rel1

    rel2 = Relationship(sentence, before, between, after, ent1_str, ent2_str, "LOC", "LOC")
    assert not rel1 == rel2
