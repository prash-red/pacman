"""
This file contains the algorithm for 6ix-pac to move around the grid.
"""
import random

# You can modify this file to implement your own algorithm

from sqlitedict import SqliteDict
from constants import *
from typing import Any, Optional
from graph import *
from collections import deque
import copy

from grid import grid

"""
You can use the following values from constants.py to check for the type of cell in the grid:
I = 1 -> Wall 
o = 2 -> Pellet (Small Dot)
e = 3 -> Empty
"""
GRID_WIDTH = 28
GRID_HEIGHT = 31


def load_graph(filename: str) -> Graph:
    """
    Load a graph from a file.

    Parameters:
    - filename (str): The name of the file to load the graph from.

    Returns:
    - Graph: The graph loaded from the file.
    """

    return GraphSerializer.deserialize(filename)


def get_next_coordinate(grid: list[list], location: tuple) -> Optional[list | tuple]:
    """
    Calculate the next coordinate for 6ix-pac to move to.
    Check if the next coordinate is a valid move.

    Parameters:
    - grid (list of lists): A 2D array representing the game board.
    - location (list): The current location of the 6ix-pac in the form (x, y).

    Returns:
    - list or tuple:
        - If the next coordinate is valid, return the next coordinate in the form (x, y) or [x,y].
        - If the next coordinate is invalid, return None.
    """
    algorithm = 2

    if algorithm == 1:
        nearest_pellet, path = bfs_nearest_pellet(grid, location, set())
        if nearest_pellet is None:
            return None
        else:
            return path[0]

    if algorithm == 2:
        path = load('path')
        if path is None or len(path) == 0 or path[0] is None:
            algo_graph = load('algo_graph')
            curr_vertex_coords = load('next_vertex')
            curr_vertex = algo_graph.get_vertex(curr_vertex_coords)
            next_vertex = fleury_next_move(algo_graph, curr_vertex, dict())
            next_vertex_coords = (next_vertex.col, next_vertex.row)
            path = deque(bfs(grid, curr_vertex_coords, next_vertex_coords))
            next_coord = path.popleft()
            save('path', path)
            save('next_vertex', next_vertex_coords)
            save('algo_graph', algo_graph)
            return next_coord

        else:
            next_coord = path.popleft()
            save('path', path)
            return next_coord


def bfs(grid: list[list], start: tuple, end: tuple) -> Optional[list]:
    """
    Find the shortest path from start to end using BFS.
    Check if the next coordinate is a valid move.

    Parameters:
    - grid (list of lists): A 2D array representing the game board.
    - start (list): The starting location of the 6ix-pac in the form (x, y).
    - end (list): The ending location of the 6ix-pac in the form (x, y).

    Returns:
    - list or tuple:
        - If the next coordinate is valid, return the next coordinate in the form (x, y) or [x,y].
        - If the next coordinate is invalid, return None.
    """
    queue = deque()
    queue.append((start, []))
    visited = set()
    visited.add(start)

    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x = current[0] + x
            next_y = current[1] + y
            if 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT and grid[next_x][next_y] != I and grid[next_x][
                next_y] != n and (next_x, next_y) not in visited:
                queue.append(((next_x, next_y), path + [(next_x, next_y)]))
                visited.add((next_x, next_y))


def bfs_nearest_pellet(grid: list[list], location: tuple, visited: set) -> Optional[tuple]:
    """
    Find the nearest pellet using BFS.
    Check if the next coordinate is a valid move.

    Parameters:
    - grid (list of lists): A 2D array representing the game board.
    - location (list): The current location of the 6ix-pac in the form (x, y).

    Returns:
    - list or tuple:
        - If the next coordinate is valid, return the next coordinate in the form (x, y) or [x,y].
        - If the next coordinate is invalid, return None.
    """
    queue = deque()
    queue.append((location, []))
    visited.add(location)

    while queue:
        current, path = queue.popleft()
        if grid[current[0]][current[1]] == o:
            return current, path
        for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x = current[0] + x
            next_y = current[1] + y
            if 0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT and grid[next_x][next_y] != I and grid[next_x][
                next_y] != n and (next_x, next_y) not in visited:
                queue.append(((next_x, next_y), path + [(next_x, next_y)]))
                visited.add((next_x, next_y))


def fleury_next_move(graph: Graph, curr_vertex: Vertex, bridge_dict: dict) -> Optional[Vertex]:
    """
    Find the next vertex using Fleury's Algorithm.

    Parameters:
    - graph (Graph): The graph representing the game board.
    - curr_vertex (Vertex): The current vertex of the 6ix-pac.

    Returns:
    - tuple:
        - If the next coordinate is valid, return the next coordinate in the form (x, y).
        - If the next coordinate is invalid, return None.
    """

    if graph is None or curr_vertex is None:
        return None

    if len(curr_vertex.edges) == 0:
        return None

    next_edge = curr_vertex.edges[0]
    for edge in curr_vertex.edges:
        edge_is_bridge = bridge_dict.get(edge, is_bridge(graph, edge))
        bridge_dict[edge] = edge_is_bridge
        if not edge_is_bridge:
            next_edge = edge
            break

    graph.remove_edge(next_edge)

    if next_edge.vertex1 == curr_vertex:
        return next_edge.vertex2
    else:
        return next_edge.vertex1


def is_bridge(graph: Graph, edge: Edge) -> bool:
    """
    Check if an edge is a bridge.

    Parameters:
    - graph (Graph): The graph representing the game board.
    - edge (Edge): The edge to check.

    Returns:
    - bool: True if the edge is a bridge, False otherwise.
    """
    if graph is None or edge is None:
        return False
    graph.remove_edge(edge)
    edge_is_bridge = not dfs_is_connected(graph, edge.vertex1, edge.vertex2, set())
    graph.add_edge(edge)
    return edge_is_bridge


def dfs_is_connected(graph: Graph, vertex1: Vertex, vertex2: Vertex, visited: set) -> bool:
    """
    Check if an vertex1 is connected to vertex2.

    Parameters:
    - graph (Graph): The graph representing the game board.
    - vertex1 (Vertex): The first vertex to check.
    - vertex2 (Vertex): The second vertex to check.
    - visited (set): The set of visited vertices.

    Returns:
    - bool: True if the vertex1 is connected to vertex2, False otherwise.
    """
    if graph is None or vertex1 is None or vertex2 is None:
        return False

    if vertex1 == vertex2:
        return True

    visited.add(vertex1)

    for edge in vertex1.edges:
        if edge.vertex1 == vertex1 and edge.vertex2 not in visited:
            if dfs_is_connected(graph, edge.vertex2, vertex2, visited):
                return True
        elif edge.vertex2 == vertex1 and edge.vertex1 not in visited:
            if dfs_is_connected(graph, edge.vertex1, vertex2, visited):
                return True

    return False


def save(key: hash, value: Any, cache_file="cache.sqlite3"):
    """
    Save the key-value pair to the cache file. If the key already exists, overwrite the value.

    Parameters:
    - key (str): The key to store the value under.
    - value (str): The value to store.
    - cache_file (str): The name of the cache file to store the key-value pair in.

    Returns:
    - None
    """
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value  # Using dict[key] to store
            mydict.commit()  # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)


def load(key: hash, cache_file="cache.sqlite3"):
    """
    Load the value stored under the key from the cache file.

    Parameters:
    - key (str): The key to load the value from.
    - cache_file (str): The name of the cache file to load the key-value pair from.

    Returns:
    - object: The object stored under the key.

    """
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict.get(key, None)  # No need to use commit(), since we are only loading data!
        return value
    except Exception as ex:
        print("Error during loading data:", ex)


if __name__ == "__main__":
    path = deque([(15, 7)])
    save('path', path)
    eulerian_graph = load('eulerian_graph')
    algo_graph = copy.deepcopy(eulerian_graph)
    save('algo_graph', algo_graph)
    next_vertex = (15, 7)
    save('next_vertex', next_vertex)
