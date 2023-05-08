from unittest.mock import Mock, patch

import pytest

from snowball.snowball_tuple import SnowballTuple


@pytest.fixture()
def mock_config():
    config = Mock()
    config.use_reverb = "no"
    config.stopwords = []
    config.vsm = Mock()
    config.vsm.dictionary.doc2bow.return_value = [(0, 1)]
    config.vsm.tf_idf_model.__getitem__.return_value = [(0, 0.5)]
    return config


@pytest.fixture()
def mock_reverb():
    reverb = Mock()
    reverb.extract_reverb_patterns_tagged_ptb.return_value = [("loves", "VBZ"), ("dogs", "NNS")]
    reverb.detect_passive_voice.return_value = False
    return reverb


@pytest.fixture()
def mock_words():
    return [("The", "DT"), ("quick", "JJ"), ("brown", "JJ"), ("fox", "NN")]


def test_create_vector(mock_config):
    text = "Volkswagen is a car manufacturer based in Wolfsburg."
    snowball_tuple = SnowballTuple(
        "Volkswagen", "Wolfsburg", text, "", "is a car manufacturer based in", "", mock_config
    )
    vector = snowball_tuple.create_vector(text)
    assert len(vector) == 1
    assert vector[0][0] == 0
    assert vector[0][1] == 0.5


def test_tokenize(mock_config):
    text = "The quick brown fox"
    snowball_tuple = SnowballTuple(None, None, None, None, None, None, mock_config)
    tokens = snowball_tuple.tokenize(text)
    assert len(tokens) == 4
    assert tokens == ["quick", "brown", "fox"]


def test_construct_pattern_vector(mock_config, mock_reverb):
    pattern_tags = [("loves", "VBZ"), ("dogs", "NNS")]
    snowball_tuple = SnowballTuple(None, None, None, None, None, None, mock_config)
    vector = snowball_tuple.construct_pattern_vector(pattern_tags, mock_config)
    assert len(vector) == 1
    assert vector[0][0] == 0
    assert vector[0][1] == 0.5


def test_construct_words_vectors(mock_config, mock_words):
    snowball_tuple = SnowballTuple(None, None, None, None, None, None, mock_config)
    vector = snowball_tuple.construct_words_vectors(mock_words, mock_config)
    assert len(vector) == 1
    assert vector[0][0] == 0
    assert vector[0][1] == 0.5


def test_extract_patterns_no_reverb(mock_config, mock_words):
    snowball_tuple = SnowballTuple(None, None, None, mock_words[:2], [], mock_words[2:], mock_config)
    snowball_tuple.extract_patterns(mock_config)
    assert snowball_tuple.bef_vector is not None
    assert snowball_tuple.bet_vector is None
    assert snowball_tuple.aft_vector is not None


def test_extract_patterns_with_reverb(mock_config, mock_reverb, mock_words):
    snowball_tuple = SnowballTuple(None, None, None, mock_words[:2], mock_words[2:3], mock_words[3:], mock_config)
    with patch("snowball.snowball_tuple.Reverb", return_value=mock_reverb):
        snowball_tuple.extract_patterns(mock_config)
    assert snowball_tuple.bef_vector is not None
    assert snowball_tuple.bet_vector is not None
    assert snowball_tuple.aft_vector is not None
    assert snowball_tuple.passive
