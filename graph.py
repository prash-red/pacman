""" This module contains the function to create a graph from a grid."""
from __future__ import annotations

from collections import deque

from sqlitedict import SqliteDict
from typing import Optional
from typing import Any
from constants import *  # I, o, e, n
import json
import copy

grid = [[I, I, I, I, I, I, I, I, I, I, I, I, e, e, e, e, e, e, e, e, e, I, I, I, I, I, I, I, I, I, I],  # 0
        [I, o, o, o, o, I, I, o, o, o, o, I, e, e, e, e, e, e, e, e, e, I, o, o, o, o, o, o, o, o, I],
        [I, o, I, I, o, I, I, o, I, I, o, I, e, e, e, e, e, e, e, e, e, I, o, I, I, o, I, I, I, o, I],
        [I, o, I, I, o, o, o, o, I, I, o, I, e, e, e, e, e, e, e, e, e, I, o, I, I, o, I, e, I, o, I],
        [I, o, I, I, o, I, I, I, I, I, o, I, e, e, e, e, e, e, e, e, e, I, o, I, I, o, I, e, I, o, I],
        [I, o, I, I, o, I, I, I, I, I, o, I, I, I, I, I, I, I, I, I, I, I, o, I, I, o, I, I, I, o, I],  # 5
        [I, o, I, I, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, I],
        [I, o, I, I, I, I, I, o, I, I, o, I, I, I, I, I, e, I, I, I, I, I, I, I, I, o, I, I, I, o, I],
        [I, o, I, I, I, I, I, o, I, I, o, I, I, I, I, I, e, I, I, I, I, I, I, I, I, o, I, e, I, o, I],
        [I, o, I, I, o, o, o, o, I, I, o, e, e, e, e, e, e, e, e, e, I, I, o, o, o, o, I, e, I, o, I],
        [I, o, I, I, o, I, I, o, I, I, o, I, I, e, I, I, I, I, I, e, I, I, o, I, I, o, I, e, I, o, I],  # 10
        [I, o, I, I, o, I, I, o, I, I, o, I, I, e, I, n, n, n, I, e, I, I, o, I, I, o, I, I, I, o, I],
        [I, o, o, o, o, I, I, o, o, o, o, I, I, e, I, n, n, n, I, e, e, e, o, I, I, o, o, o, o, o, I],
        [I, o, I, I, I, I, I, e, I, I, I, I, I, e, I, n, n, n, n, e, I, I, I, I, I, o, I, I, I, I, I],
        [I, o, I, I, I, I, I, e, I, I, I, I, I, e, I, n, n, n, n, e, I, I, I, I, I, o, I, I, I, I, I],
        [I, o, o, o, o, I, I, o, o, o, o, I, I, e, I, n, n, n, I, e, e, e, o, I, I, o, o, o, o, o, I],  # 15
        [I, o, I, I, o, I, I, o, I, I, o, I, I, e, I, n, n, n, I, e, I, I, o, I, I, o, I, I, I, o, I],
        [I, o, I, I, o, I, I, o, I, I, o, I, I, e, I, I, I, I, I, e, I, I, o, I, I, o, I, e, I, o, I],
        [I, o, I, I, o, o, o, o, I, I, o, e, e, e, e, e, e, e, e, e, I, I, o, o, o, o, I, e, I, o, I],
        [I, o, I, I, I, I, I, o, I, I, o, I, I, I, I, I, e, I, I, I, I, I, I, I, I, o, I, e, I, o, I],
        [I, o, I, I, I, I, I, o, I, I, o, I, I, I, I, I, e, I, I, I, I, I, I, I, I, o, I, I, I, o, I],  # 20
        [I, o, I, I, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, I],
        [I, o, I, I, o, I, I, I, I, I, o, I, I, I, I, I, I, I, I, I, I, I, o, I, I, o, I, I, I, o, I],
        [I, o, I, I, o, I, I, I, I, I, o, I, e, e, e, e, e, e, e, e, e, I, o, I, I, o, I, e, I, o, I],
        [I, o, I, I, o, o, o, o, I, I, o, I, e, e, e, e, e, e, e, e, e, I, o, I, I, o, I, e, I, o, I],
        [I, o, I, I, o, I, I, o, I, I, o, I, e, e, e, e, e, e, e, e, e, I, o, I, I, o, I, I, I, o, I],  # 25
        [I, o, o, o, o, I, I, o, o, o, o, I, e, e, e, e, e, e, e, e, e, I, o, o, o, o, o, o, o, o, I],
        [I, I, I, I, I, I, I, I, I, I, I, I, e, e, e, e, e, e, e, e, e, I, I, I, I, I, I, I, I, I, I]]
#        |              |              |              |              |              |              |
#        0              5              10             15             20             25             30

restricted = {(col, row) for col in range(23, 28) for row in range(12, 21)}.union(
    {(col, row) for col in range(0, 5) for row in range(12, 21)},
    {(24, 27), (23, 27), (19, 27), (18, 27), (17, 27), (10, 27), (9, 27), (8, 27), (4, 27), (3, 27)})


class Vertex:
    """
    A vertex in a graph.

    Attributes:
        col (int): The column of the vertex.
        row (int): The row of the vertex.
        edges (list[Edge]): The edges connected to the vertex.
    """

    def __init__(self, col: int, row: int):
        self.col = col
        self.row = row
        self.edges = []

    def add_edge(self, edge):
        """
        Add an edge to the vertex.
        Args:
            edge: The edge to add.
        """
        self.edges.append(edge)

    def __repr__(self):
        return f"({self.col}, {self.row})"


class Edge:
    """
    An edge in a graph.

    Attributes:
        vertex1 (Vertex): The first vertex connected to the edge.
        vertex2 (Vertex): The second vertex connected to the edge.
        weight (int): The weight of the edge.
        pseudo (bool): Whether the edge is a pseudo edge.
    """

    def __init__(self, vertex1: Vertex, vertex2: Vertex, weight: int, pseudo: bool = False):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.weight = weight
        self.pseudo = pseudo

    def __repr__(self):
        return f"{self.vertex1} -- {self.vertex2}, weight: {self.weight}"


class Graph:
    """
    A graph.

    Attributes:
        vertices (set[Vertex]): The vertices in the graph.
        edges (set[Edge]): The edges in the graph.

    """

    def __init__(self, vertices: set[Vertex], edges: set[Edge]):
        self.vertices = vertices
        self.edges = edges

    def __repr__(self):
        return f"Vertices: {self.vertices}\nEdges: {self.edges}"

    def add_edge(self, edge: Edge):
        """
        Add an edge to the graph.
        Args:
            edge: The edge to add.
        """
        self.edges.add(edge)
        edge.vertex1.add_edge(edge)
        edge.vertex2.add_edge(edge)

    def remove_edge(self, edge: Edge):
        """
        Remove an edge from the graph.
        Args:
            edge: The edge to remove.
        """
        self.edges.remove(edge)
        edge.vertex1.edges.remove(edge)
        edge.vertex2.edges.remove(edge)

    def remove_vertex(self, vertex: tuple[int, int]):
        """
        Remove a vertex from the graph.
        Args:
            vertex: a tuple of the vertex to remove.
        """
        vertex = self.get_vertex(vertex)
        self.vertices.remove(vertex)
        for edge in vertex.edges:
            self.remove_edge(edge)

    def get_vertex(self, vertex: tuple[int, int]) -> Vertex:
        """
        Get the vertex with the given coordinates.
        Args:
            vertex: The coordinates of the vertex to get.

        Returns:
            Vertex: The vertex with the given coordinates.
        """
        for v in self.vertices:
            if v.col == vertex[0] and v.row == vertex[1]:
                return v

    def djikstra(self, start: Vertex, end: Vertex) -> list[Vertex]:
        """
        Find the shortest path between two vertices using Djikstra's algorithm.
        Args:
            start: The starting vertex.
            end: The ending vertex.

        Returns:
            list[Vertex]: The shortest path between the two vertices.
        """
        visited = set()
        distances = {vertex: float("inf") for vertex in self.vertices}
        distances[start] = 0
        previous = {vertex: None for vertex in self.vertices}

        while len(visited) < len(self.vertices):
            current = min((vertex for vertex in self.vertices if vertex not in visited), key=lambda v: distances[v])
            visited.add(current)

            for edge in self.edges:
                neighbour = edge.vertex1 if edge.vertex1 != current else edge.vertex2
                if neighbour not in visited:
                    new_distance = distances[current] + edge.weight
                    if new_distance < distances[neighbour]:
                        distances[neighbour] = new_distance
                        previous[neighbour] = current

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()
        return path

    def is_eulerian(self) -> bool:
        """
        Check if the graph is eulerian.
        Returns:
            bool: True if the graph is eulerian, False otherwise.
        """
        for vertex in self.vertices:
            if len(vertex.edges) % 2 == 1:
                return False
        return True


def valid_coordinate(grid: list[list], coordinate: tuple) -> bool:
    """
    Check if the coordinate is a valid move.
    Args:
        grid: The grid to check the coordinate on.
        coordinate: The coordinate to check.

    Returns:
        bool: True if the coordinate is a valid move, False otherwise.
    """
    x, y = coordinate
    return 0 <= x < len(grid) and 0 <= y < len(grid[x]) and grid[x][y] != I and grid[x][y] != n


def create_graph(grid: list[list[I, 0, o, e, n, c]]) -> Graph:
    """
    Create a graph from the grid.
    The nodes are the all the points in the graph where pacman can change direction
    The edges are the paths between the nodes

    Parameters:
    - grid (list of lists): A 2D array representing the game board.

    Returns:
    - Graph: A graph.
    """

    graph = Graph(set(), set())
    for col in range(len(grid)):
        for row in range(len(grid[col])):
            # if we can turn from horizontal direction to vertical direction or vice-versa at the current cell,
            # we add it as a vertex
            directions = set()
            for x, y in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                if 0 <= col + x < len(grid) and 0 <= row + y < len(grid[col]) and grid[col + x][row + y] != I and \
                        grid[col + x][row + y] != n:
                    directions.add((x, y))
            if (col, row) not in restricted and grid[col][row] != I and grid[col][row] != n and len(directions) > 0:
                if len(directions) >= 3:
                    graph.vertices.add(Vertex(col, row))

                elif len(directions) == 2:
                    horizontal_moves = []
                    vertical_moves = []
                    for x, y in directions:
                        if x == 0:
                            vertical_moves.append((x, y))
                        else:
                            horizontal_moves.append((x, y))

                    if len(horizontal_moves) == len(vertical_moves) == 1:
                        graph.vertices.add(Vertex(col, row))

    # code to remove the extra vertices that are not needed

    extra_vertices = {(9, 10), (18, 10), (9, 13), (18, 13), (6, 16), (21, 16), (9, 16), (18, 16), (9, 19), (18, 19),
                      (12, 19), (15, 19)}
    for vertex in extra_vertices:
        graph.remove_vertex(vertex)

    # code to connect the vertices, such that vertices are connected if and only if they are adjacent to each other

    vertex_set = {(vertex.col, vertex.row) for vertex in graph.vertices}

    for vertex in graph.vertices:
        directions = {(0, 1), (1, 0), (-1, 0), (0, -1)}

        scale = 1
        while directions:
            direction_found = set()
            for x, y in directions:
                next_x = vertex.col + x * scale
                next_y = vertex.row + y * scale

                if (next_x, next_y) in vertex_set:
                    if vertex.col == next_x:
                        distance = abs(vertex.row - next_y)
                    else:
                        distance = abs(vertex.col - next_x)

                    adjacent_vertex = graph.get_vertex((next_x, next_y))
                    # check if edge already exists
                    if any(
                            edge.vertex1 == vertex and edge.vertex2 == adjacent_vertex or edge.vertex1 == adjacent_vertex and edge.vertex2 == vertex
                            for edge in graph.edges):
                        direction_found.add((x, y))
                        continue
                    edge = Edge(vertex, graph.get_vertex((next_x, next_y)), distance)
                    graph.add_edge(edge)
                    direction_found.add((x, y))
                else:
                    if not valid_coordinate(grid, (next_x, next_y)):
                        direction_found.add((x, y))
            for direction in direction_found:
                directions.remove(direction)
            scale += 1
    return graph


def create_eulerian(graph: Graph) -> Graph:
    """
    Create a eulerian graph from the graph.
    The nodes are the all the points in the graph where pacman can change direction
    The edges are the paths between the nodes

    Parameters:
    - graph (Graph): The graph to create a eulerian graph from.

    Returns:
    - Graph: A eulerian graph.
    """
    new_graph = copy.deepcopy(graph)
    odd_vertices = set()

    for vertex in new_graph.vertices:
        if len(vertex.edges) % 2 == 1:
            odd_vertices.add(vertex)

    if len(odd_vertices) == 0:
        return new_graph
    else:
        # pair up the odd vertices with the shortest path and add a pseudo edge
        if len(odd_vertices) % 2 == 1:
            odd_vertices.pop()

        while len(odd_vertices) > 0:
            vertex1 = odd_vertices.pop()
            vertex2 = min(odd_vertices, key=lambda v: len(new_graph.djikstra(vertex1, v)))
            odd_vertices.remove(vertex2)
            path = new_graph.djikstra(vertex1, vertex2)
            edge = Edge(vertex1, vertex2, len(path), True)
            new_graph.add_edge(edge)

        return new_graph


class GraphSerializer:
    """
    A class to serialize a graph.
    """

    @staticmethod
    def serialize(graph: Graph) -> str:
        """
        Serialize a graph to a JSON string.
        Args:
            graph: The graph to serialize.

        Returns:
            str: The serialized graph.
        """
        serialized_graph = {"vertices": [], "edges": []}

        # Serialize vertices
        for vertex in graph.vertices:
            serialized_vertex = {"col": vertex.col, "row": vertex.row}
            serialized_graph["vertices"].append(serialized_vertex)

        # Serialize edges
        for edge in graph.edges:
            serialized_edge = {
                "vertex1": (edge.vertex1.col, edge.vertex1.row),
                "vertex2": (edge.vertex2.col, edge.vertex2.row),
                "weight": edge.weight,
                "pseudo": edge.pseudo
            }
            serialized_graph["edges"].append(serialized_edge)

        # Convert to JSON string
        json_str = json.dumps(serialized_graph, indent=4)
        return json_str

    @staticmethod
    def save_to_file(graph: Graph, filename: str):
        """
        Save the graph to a file.
        Args:
            graph: The graph to save.
            filename: The name of the file to save the graph to.
        """
        # Serialize the graph to JSON
        serialized_graph = GraphSerializer.serialize(graph)

        try:
            # Open the file in write mode and write the JSON data
            with open(filename, "w") as file:
                file.write(serialized_graph)
            print(f"Graph saved to {filename}")
        except Exception as e:
            print(f"Error saving the graph to {filename}: {str(e)}")

    @staticmethod
    def deserialize(filename: str) -> Optional[Graph]:
        """
        Deserialize a graph from a file.
        Args:
            filename: The name of the file to deserialize the graph from.

        Returns:
            dict: The deserialized graph.
        """
        try:
            # Open the file in read mode and load the JSON data
            with open(filename, "r") as file:
                serialized_graph = json.load(file)

            vertices = set()
            edges = set()

            # Deserialize vertices
            for serialized_vertex in serialized_graph["vertices"]:
                vertex = Vertex(serialized_vertex["col"], serialized_vertex["row"])
                vertices.add(vertex)

            # Deserialize edges
            for serialized_edge in serialized_graph["edges"]:
                vertex1 = Vertex(*serialized_edge["vertex1"])
                vertex2 = Vertex(*serialized_edge["vertex2"])
                weight = serialized_edge["weight"]
                pseudo = serialized_edge["pseudo"]
                edge = Edge(vertex1, vertex2, weight, pseudo)
                edges.add(edge)

            # Construct the graph
            graph = Graph(vertices, edges)
            return graph
        except Exception as e:
            print(f"Error deserializing the graph from {filename}: {str(e)}")
            return None


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
            value = mydict[key]  # No need to use commit(), since we are only loading data!
        return value
    except Exception as ex:
        print("Error during loading data:", ex)


if __name__ == "__main__":
    graph = create_graph(grid)

    GraphSerializer.save_to_file(graph, "graph.json")
    save("graph", graph)

    eulerian_graph = create_eulerian(graph)

    GraphSerializer.save_to_file(eulerian_graph, "eulerian_graph.json")
    save("eulerian_graph", eulerian_graph)
