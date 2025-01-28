import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """

        adj_mat = self.adj_mat

        if adj_mat.size == 0:
            raise ValueError("This adjacency matrix is empty")
        
        if np.any(np.vectorize(lambda x: isinstance(x, str))(adj_mat)):
            raise ValueError("This adjacency matrix contains strings")
        
        if np.isnan(adj_mat).any():
            raise ValueError("This adjacency matrix contains NaNs")
        

        S = set() #set of nodes
        T = [] #holds the minimum spanning tree node and parent node

        n = len(adj_mat[0]) #get the number of nodes in the square matrix
        start = 0 #we start with the first node, 0

        pred = n*[None] #initialize the parents of the nodes as None, as they are not connected yet
        pred[start] = -1 #set the parent of start node as itself, designated with -1

        cost = n*[np.inf] #initilize the cost of all nodes as inf
        cost[start]= 0 #except the start node, which has distance 0 to itself

        pq = []
        

        for node, costval in enumerate(cost): #we add each of the nodes and the inital cost value to the heapq
            heapq.heappush(pq, (costval, node))

        while pq: #while we have items in the list to through, we pop off the node/path with the lowest cost
            q = heapq.heappop(pq)
            u = q[1]

            if u in S: #if the node is not already in the set of nodes we have seen, then we will add it to S
                continue

            S.add(u)

            if pred[u] != -1: #if it isn't the start node, we add it to the path of the minimum spanning tree with its parent
                T.append((pred[u], u))

            for i in range(n): #we add the leaves of the node to the heap if the cost is less than the current cost
                if (u != i) and (i not in S) and (adj_mat[u][i] < cost[i]) and (adj_mat[u][i] > 0):
                    heapq.heappush(pq, (adj_mat[u][i], i))
                    cost[i] = adj_mat[u][i]
                    pred[i] = u

        mst = np.zeros((n, n)) #we build up the mst solution, based on the pairs of parent nodes and current nodes, and their weights
        for u, v in T:
            mst[u][v] = adj_mat[u][v]
            mst[v][u] = adj_mat[u][v]
        
        self.mst = mst

            
        


