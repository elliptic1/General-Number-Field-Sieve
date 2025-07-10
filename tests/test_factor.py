from gnfs.factor import gnfs_factor


def test_gnfs_factor_even_number():
    assert gnfs_factor(10) == [2, 5]
