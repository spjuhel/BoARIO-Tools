import pytest

import pymrio

from boario_tools.mriot import lexico_reindex


@pytest.fixture
def example_mriot():
    # Create an example pymrio.IOSystem for testing
    # You should adjust this based on the actual structure of your data
    return pymrio.load_test()


def test_lexico_reindex(example_mriot):
    with pytest.raises(ValueError):
        lexico_reindex(example_mriot)

    example_mriot.calc_all()

    # Ensure lexico_reindex sorts matrices Z, Y, x, and A lexicographically
    sorted_mriot = lexico_reindex(example_mriot)

    # Check if matrices Z, Y, x, and A are sorted
    assert sorted_mriot.Z.columns.is_monotonic_increasing
    assert sorted_mriot.x.columns.is_monotonic_increasing
    assert sorted_mriot.Y.columns.is_monotonic_increasing
    assert sorted_mriot.A.columns.is_monotonic_increasing

    # Check if matrices Z, Y, x, and A are lexicographically sorted
    assert sorted_mriot.Z.index.is_monotonic_increasing
    assert sorted_mriot.x.index.is_monotonic_increasing
    assert sorted_mriot.Y.index.is_monotonic_increasing
    assert sorted_mriot.A.index.is_monotonic_increasing
