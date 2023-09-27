# Pac-Man Pathfinding Project

## Introduction

This project is an assignment by the UTRA club to recruit members into the PacBots subteam. The assignment involves creating a pathfinding algorithm for Pac-Man to collect all the pellets on the game board in the shortest time possible. To accomplish this, two different algorithms have been implemented: Breadth-First Search (BFS) and Fleury's Algorithm.

## Project Overview

### Breadth-First Search (BFS) Algorithm
The BFS algorithm is designed to find the nearest pellet from Pac-Man's current location and generate a path to reach it. It explores the game board in a breadth-first manner, ensuring that the first pellet found is the closest one. This algorithm is efficient for finding the optimal path to individual pellets.

### Fleury's Algorithm
Fleury's Algorithm takes a different approach. It begins by creating a graph representation of the game board, where all the direction-changing points (vertices) are considered as vertices, and the paths between them (valid game board paths) are represented as edges. This graph is then converted into an Eulerian graph. Fleury's Algorithm can be run on an Eulerian graph to find a path that visits each edge only once. In the context of Pac-Man, this translates to a path that covers all the turning points on the game board while collecting pellets in an optimized way.

## Usage

### Running the BFS Algorithm
To use the BFS algorithm, simply provide Pac-Man's current position and the game board as input. The algorithm will find the nearest pellet and generate a path to reach it.
Set 'algorithm = 1' in the 'get_next_coordinate()' function in algo.py to run the BFS algorithm.

### Running Fleury's Algorithm
To use Fleury's Algorithm, set 'algorithm = 2' in the 'get_next_coordinate()' function in algo.py to run Fleurry's algorithm. 
>  ⚠️ Make sure to run algo.py before each time you run game.py to ensure the cachel.sqlite3 file is configured properly for fleurry's algorithm.


## Getting Started

To get started with this project, clone the repository and explore the code and algorithms. Run game.py to visualize the algorithm.

## Acknowledgments

We would like to thank the UTRA club for providing this exciting assignment and opportunity to work on Pac-Man pathfinding algorithms.
