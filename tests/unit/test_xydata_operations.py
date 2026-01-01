import pytest
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from labchain.base.base_types import XYData, ensure_dim


def test_ensure_dim_numpy():
    # Test with 1D numpy array
    x = np.array([1, 2, 3])
    result = ensure_dim(x)
    assert result.shape == (3, 1)

    # Test with 2D numpy array
    x = np.array([[1, 2], [3, 4]])
    result = ensure_dim(x)
    assert result.shape == (2, 2)


def test_ensure_dim_torch():
    # Test with 1D torch tensor
    x = torch.tensor([1, 2, 3])
    result = ensure_dim(x)
    assert result.shape == (3, 1)

    # Test with 2D torch tensor
    x = torch.tensor([[1, 2], [3, 4]])
    result = ensure_dim(x)
    assert result.shape == (2, 2)


def test_ensure_dim_list():
    # Test with 1D list
    x = [1, 2, 3]
    result = ensure_dim(x)
    assert result.shape == (3, 1)

    # Test with 2D list
    x = [[1, 2], [3, 4]]
    result = ensure_dim(x)
    assert result.shape == (2, 2)


def test_concat_numpy():
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[5, 6], [7, 8]])
    result = XYData.concat([x1, x2], axis=0)
    assert isinstance(result, XYData)
    assert result.value.shape == (4, 2)


def test_concat_pandas():
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    result = XYData.concat([df1, df2], axis=0)
    assert isinstance(result, XYData)
    assert result.value.shape == (4, 2)


def test_concat_torch():
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])
    result = XYData.concat([t1, t2], axis=0)
    assert isinstance(result, XYData)
    assert result.value.shape == (4, 2)


def test_concat_sparse():
    s1 = csr_matrix([[1, 2], [3, 4]])
    s2 = csr_matrix([[5, 6], [7, 8]])
    result = XYData.concat([s1, s2], axis=0)
    assert isinstance(result, XYData)
    assert result.value.shape == (4, 2)


def test_ensure_dim_invalid_type():
    with pytest.raises(TypeError):
        ensure_dim("not a valid type")


def test_concat_invalid_type():
    with pytest.raises(TypeError):
        XYData.concat(["not a valid type"])
