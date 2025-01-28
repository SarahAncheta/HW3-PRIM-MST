import pytest
import numpy as np

from mst.graph import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    """
    An MST should have n - 1 edges (n is number of nodes).
    divide the total number of nonzeros by two, because the matrix is symmetric
    """
    assert (mst.shape[0] - 1)  == (np.count_nonzero(mst)/2)

    """
    We check if the mst is fully connected (they are always connected) using a linear algebra trick that I looked up
    https://math.stackexchange.com/questions/864604/checking-connectivity-of-adjacency-matrix
    we square the matrix to the power of the number of nodes
    if any of the rows or columns is zero, then that node is isolated
    """

    reachability = np.linalg.matrix_power(mst, mst.shape[0]) #we use a linear algebra trick that I looked up
    assert not (reachability == 0).all(axis=0).any()



def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_empty_data():
    """
    
    Unit test, should raise exception if given an empty array (adjacency matrix is empty)
    
    """
    g = Graph(np.array([]))
    with pytest.raises(ValueError):
        g.construct_mst()

def test_mst_strings():
    """
    
    Unit test, should raise exception if given an array that contains strings
    
    """

    words = np.array([
    [0, 1, "a"],
    [1, 0, 2],
    [2, 1, 0]])

    g = Graph(words)

    with pytest.raises(ValueError):
        g.construct_mst()

def test_mst_nans():

    """
    
    Unit test, should raise exception if given an array with NaNs
    
    """
    na_matrix = np.array([
    [0, 1, np.nan],
    [1, 0, 2],
    [2, 1, 0]])

    g = Graph(na_matrix)

    with pytest.raises(ValueError):
        g.construct_mst()
