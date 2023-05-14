"""
@pytest.fixture
def test_config():
    config = Config()
    config.positive_seeds = [
        {'ent1': 'person', 'ent2': 'city'},
        {'ent1': 'person', 'ent2': 'country'},
    ]
    config.negative_seeds = [
        {'ent1': 'person', 'ent2': 'company'},
        {'ent1': 'person', 'ent2': 'organization'},
    ]
    config.w_unk = 1.0
    config.w_neg = 1.0
    return config

@pytest.fixture
def test_pattern():
    pattern = Pattern()
    return pattern


def test_add_tuple(test_pattern):
    tpl1 = ('person', 'lives', 'city')
    tpl2 = ('person', 'lives', 'country')
    test_pattern.add_tuple(tpl1)
    assert test_pattern.tuples == [tpl1]
    test_pattern.add_tuple(tpl2)
    assert test_pattern.tuples == [tpl1, tpl2]


def test_merge_tuple_patterns(test_pattern):
    tpl1 = ('person', 'lives', 'city')
    tpl2 = ('person', 'works for', 'company')
    tpl3 = ('person', 'lives', 'country')
    test_pattern.add_tuple(tpl1)
    test_pattern.add_tuple(tpl2)
    test_pattern.add_tuple(tpl3)
    test_pattern.merge_tuple_patterns()
    assert len(test_pattern.tuple_patterns) == 2
    assert {'lives'} in test_pattern.tuple_patterns
    assert {'works for'} in test_pattern.tuple_patterns


def test_updated_centroid(test_pattern):
    tpl1 = ('person', 'lives', 'city')
    tpl2 = ('person', 'lives', 'country')
    test_pattern.add_tuple(tpl1)
    test_pattern.updated_centroid()
    assert test_pattern.centroid_bef == [('person', 1.0), ('lives', 1.0)]
    assert test_pattern.centroid_bet == [('city', 1.0)]
    assert test_pattern.centroid_aft == []
    test_pattern.add_tuple(tpl2)
    test_pattern.updated_centroid()
    assert test_pattern.centroid_bef == [('person', 1.0), ('lives', 1.0)]
    assert test_pattern.centroid_bet == [('city', 1.0), ('country', 1.0)]
    assert test_pattern.centroid_aft == []


def test_update_selectivity(test_pattern, test_config):
    tpl1 = ('person', 'lives', 'city')
    tpl2 = ('person', 'works for', 'company')
    tpl3 = ('company', 'is located in', 'city')
    test_pattern.update_selectivity(tpl1, test_config)
    assert test_pattern.positive == 1
    assert test_pattern.negative == 0
    assert test_pattern.unknown == 1
    test_pattern.update_selectivity(tpl2, test_config)
    assert test_pattern.positive == 1
    assert test_pattern.negative == 1
    assert test_pattern.unknown == 1
    test_pattern.update_selectivity(tpl3, test_config)
    assert test_pattern.positive == 1
    assert test_pattern.negative == 2
"""
