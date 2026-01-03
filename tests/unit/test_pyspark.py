import pytest
from labchain.utils.pyspark import PySparkMapReduce
from labchain.base import XYData


@pytest.fixture
def pyspark_map_reduce(request):
    pyspark_mr = PySparkMapReduce(app_name="test_app")

    def cleanup():
        pyspark_mr.stop()

    request.addfinalizer(cleanup)
    return pyspark_mr


def test_map_parallelization(pyspark_map_reduce):
    test_data = XYData.mock(["a", "b", "c"])
    map_function = lambda x: (x[0], len(x[0]))

    result = pyspark_map_reduce.map(test_data.value, map_function)

    assert result.count() == 3
    assert set(result.collect()) == {("a", 1), ("b", 1), ("c", 1)}


def test_map_function_application(pyspark_map_reduce):
    test_data = XYData.mock([(1, "apple"), (2, "banana"), (3, "cherry")])
    map_function = lambda x: (x[0], x[1].upper())

    result = pyspark_map_reduce.map(test_data.value, map_function)
    aux = result.collect()
    print(aux)
    assert result.count() == 3
    assert set(aux) == {(1, "APPLE"), (2, "BANANA"), (3, "CHERRY")}


def test_map_empty_input(pyspark_map_reduce):
    empty_data = XYData.mock([])
    map_function = lambda x: [(x[0], x[1])]

    result = pyspark_map_reduce.map(empty_data.value, map_function)

    assert result.count() == 0
    assert result.collect() == []


def test_reduce_function_application(pyspark_map_reduce):
    test_data = XYData.mock([("a", 1), ("b", 2), ("a", 3), ("b", 4), ("c", 5)])
    map_function = lambda x: (x[0], x[1])
    reduce_function = lambda x, y: x + y

    mapped_result = pyspark_map_reduce.map(test_data.value, map_function)
    reduced_result = pyspark_map_reduce.reduce(reduce_function)

    assert isinstance(reduced_result, tuple)
    assert reduced_result == ("a", 1, "b", 2, "a", 3, "b", 4, "c", 5)


def test_reduce_empty_mapped_rdd(pyspark_map_reduce):
    empty_data = XYData.mock([])
    map_function = lambda x: (x[0], x[1])
    reduce_function = lambda x, y: x + y

    mapped_result = pyspark_map_reduce.map(empty_data.value, map_function)
    with pytest.raises(ValueError):
        pyspark_map_reduce.reduce(reduce_function)


def test_complex_data_types(pyspark_map_reduce):
    complex_data = XYData.mock(
        [
            (1, {"fruits": ["apple", "banana"], "count": 2}),
            (2, {"fruits": ["cherry", "date"], "count": 2}),
            (1, {"fruits": ["elderberry"], "count": 1}),
        ]
    )

    def map_function(x):
        return (x[0], x[1]["fruits"])

    def reduce_function(x, y):
        return x + y

    mapped_result = pyspark_map_reduce.map(complex_data.value, map_function)
    reduced_result = pyspark_map_reduce.reduce(reduce_function)

    assert isinstance(reduced_result, tuple)
    assert reduced_result == (
        1,
        ["apple", "banana"],
        2,
        ["cherry", "date"],
        1,
        ["elderberry"],
    )


def test_maintain_key_value_structure(pyspark_map_reduce):
    test_data = XYData.mock([("a", 1), ("b", 2), ("c", 3), ("a", 4), ("b", 5)])
    map_function = lambda x: (x[0], x[1] * 2)
    reduce_function = lambda x, y: x + y

    mapped_result = pyspark_map_reduce.map(test_data.value, map_function)
    reduced_result = pyspark_map_reduce.reduce(reduce_function)
    assert isinstance(reduced_result, tuple)
    assert reduced_result == ("a", 2, "b", 4, "c", 6, "a", 8, "b", 10)
