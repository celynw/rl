#!/usr/bin/env python3
from queue import PriorityQueue
import numpy as np
from rich import print, inspect

# ==================================================================================================
class PathPlanner:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, maze: np.ndarray):
		"""Return a list of tuples as a path from start to goal using A* algorithm."""
		# self.maze = (1 - maze).transpose()
		self.maze = maze.transpose()

	# ----------------------------------------------------------------------------------------------
	"""Define the heuristic function"""
	def heuristic(self, a, b):
		return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

	# ----------------------------------------------------------------------------------------------
	def get_path(self, start: tuple[int, int], goal: tuple[int, int]):
		# Initialize the open and closed lists
		not_visited = PriorityQueue()
		visited = set()

		not_visited.put((0, start))

		# Define the cost dictionary
		cost = {}
		cost[start] = 0

		# Define the parent dictionary
		parent = {}
		parent[start] = None

		while not not_visited.empty():
			# Get the node with the lowest f-score
			current = not_visited.get()[1]

			# If we have reached the goal, construct the path and return it
			if current == goal:
				path = []
				while current is not None:
					path.append(current)
					current = parent[current]
				path.reverse()

				return path

			# Mark the current node as closed
			visited.add(current)

			# Get the neighbors of the current node
			neighbors = [(current[0] + 1, current[1]), (current[0] - 1, current[1]),
						 (current[0], current[1] + 1), (current[0], current[1] - 1)]

			# Check each neighbor
			for neighbor in neighbors:
				# Check if the neighbor is a valid location
				if 0 <= neighbor[0] < self.maze.shape[0] and 0 <= neighbor[1] < self.maze.shape[1]:
					# Check if the neighbor is an obstacle or has already been visited
					if self.maze[neighbor[0]][neighbor[1]] == 0 or neighbor in visited:
						continue

					# Calculate the tentative g-score
					tentative_cost = cost[current] + 1

					# If the neighbor has not been visited or the tentative g-score is lower than the current g-score
					if neighbor not in cost or tentative_cost < cost[neighbor]:

						# Update the cost and parent dictionaries
						cost[neighbor] = tentative_cost
						parent[neighbor] = current

						# Calculate the f-score and add the neighbor to the open list
						f_score = tentative_cost + self.heuristic(neighbor, goal)
						not_visited.put((f_score, neighbor))

		# If we have exhausted all possible paths and haven't found the goal, return None
		return []
