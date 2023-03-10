#!/usr/bin/env python3
# From https://github.com/janjilecek/pacman_python_pygame/
import argparse
import random
from enum import Enum
from math import prod
from typing import Optional, Any, SupportsFloat

import pygame
import numpy as np
import tcod
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from rich import print, inspect

unified_size = 8
move_speed = 4
assert unified_size % move_speed == 0

# ==================================================================================================
class Direction(Enum):
	DOWN = -90
	RIGHT = 0
	UP = 90
	LEFT = 180
	NONE = 360


# ==================================================================================================
class ScoreType(Enum):
	COOKIE = 10
	POWERUP = 50
	GHOST = 400


# ==================================================================================================
class GhostBehaviour(Enum):
	CHASE = 1
	SCATTER = 2


# ==================================================================================================
def translate_screen_to_maze(coords: tuple[int, int], size: int = unified_size) -> tuple[int, int]:
	return int(coords[0] / size), int(coords[1] / size)


# ==================================================================================================
def translate_maze_to_screen(coords: tuple[float, float], size: int = unified_size) -> tuple[int, int]:
	return int(coords[0] * size), int(coords[1] * size)
	# return coords[0] * size, coords[1] * size


# ==================================================================================================
class Renderer:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, game: "GameController", auto: bool = False):
		pygame.init()
		self.game = game
		self.width = self.game.width * unified_size
		self.height = self.game.height * unified_size
		# self.screen = pygame.display.set_mode((self.width, self.height)) # For interactive
		# self.surface = self.screen # For interactive
		self.surface = pygame.Surface((self.width, self.height)) # For headless
		self.clock = pygame.time.Clock()
		self.done = False
		self.won = False
		self.game_objects: list[GameObject] = []
		self.walls: list[Wall] = []
		self.cookies: list[Cookie] = []
		self.powerups: list[Powerup] = []
		self.ghosts: list[Ghost] = []
		self.hero: Optional[Hero | HeroAuto] = None
		self.lives = 1
		self.score = 0
		self.kokoro_active = False # powerup, special ability
		self.current_mode = GhostBehaviour.SCATTER
		self.mode_switch_event = pygame.USEREVENT + 1 # custom event
		self.kokoro_end_event = pygame.USEREVENT + 2
		self.pakupaku_event = pygame.USEREVENT + 3
		self.modes = [
			(7, 20),
			(7, 20),
			(5, 20),
			(5, 999999) # 'infinite' chase seconds
		]
		self.current_phase = 0

		for y, row in enumerate(self.game.maze):
			for x, column in enumerate(row):
				if column == 0:
					self.add_wall(Wall(self, x, y, unified_size))

		for cookie_space in self.game.cookie_spaces:
			translated = translate_maze_to_screen(cookie_space)
			cookie = Cookie(self, int(translated[0] + unified_size / 2), int(translated[1] + unified_size / 2))
			self.add_cookie(cookie)

		for powerup_space in self.game.powerup_spaces:
			translated = translate_maze_to_screen(powerup_space)
			powerup = Powerup(self, int(translated[0] + unified_size / 2), int(translated[1] + unified_size / 2))
			self.add_powerup(powerup)

		for i, ghost_spawn in enumerate(self.game.ghost_spawns):
			translated = translate_maze_to_screen(ghost_spawn)
			ghost = Ghost(self, translated[0], translated[1], int(unified_size * 0.75), self.game.ghost_colours[i % 4])
			self.add_ghost(ghost)

		translated = translate_maze_to_screen(self.game.pacman_spawn)
		herotype = HeroAuto if auto else Hero
		self.add_hero(herotype(self, translated[0], translated[1], unified_size))

	# ----------------------------------------------------------------------------------------------
	def tick(self, fps: int) -> None:
		black = (0, 0, 0)
		self.handle_mode_switch()
		pygame.time.set_timer(self.pakupaku_event, 200) # open close mouth
		while not self.done:
			for game_object in self.game_objects:
				game_object.tick()
				game_object.draw()

			# self.display_text(f"[Score: {self.score}]  [Lives: {self.lives}]")

			# if self.hero is None:
			# 	self.display_text("YOU DIED", (self.width / 2 - 256, self.height / 2 - 256), 100)
			# if self.get_won():
			# 	self.display_text("YOU WON", (self.width / 2 - 256, self.height / 2 - 256), 100)
			pygame.display.flip()
			self.clock.tick(fps)
			self.surface.fill(black)
			self.handle_events()
		# print("Game over")

	# ----------------------------------------------------------------------------------------------
	def handle_mode_switch(self) -> None:
		current_phase_timings = self.modes[self.current_phase]
		# print(f"Current phase: {str(self.current_phase)}, current_phase_timings: {str(current_phase_timings)}")
		scatter_timing = current_phase_timings[0]
		chase_timing = current_phase_timings[1]

		if self.current_mode == GhostBehaviour.CHASE:
			self.current_phase += 1
			self.current_mode = GhostBehaviour.SCATTER
		else:
			self.current_mode = GhostBehaviour.CHASE

		used_timing = scatter_timing if self.current_mode == GhostBehaviour.SCATTER else chase_timing
		pygame.time.set_timer(self.mode_switch_event, used_timing * 1000)

	# ----------------------------------------------------------------------------------------------
	def start_kokoro_timeout(self) -> None:
		pygame.time.set_timer(self.kokoro_end_event, 15000) # 15s

	# ----------------------------------------------------------------------------------------------
	def add_game_object(self, obj: "GameObject") -> None:
		self.game_objects.append(obj)

	# ----------------------------------------------------------------------------------------------
	def add_cookie(self, obj: "Cookie") -> None:
		self.game_objects.append(obj)
		self.cookies.append(obj)

	# ----------------------------------------------------------------------------------------------
	def add_ghost(self, obj: "Ghost") -> None:
		self.game_objects.append(obj)
		self.ghosts.append(obj)

	# ----------------------------------------------------------------------------------------------
	def add_powerup(self, obj: "Powerup") -> None:
		self.game_objects.append(obj)
		self.powerups.append(obj)

	# ----------------------------------------------------------------------------------------------
	def activate_kokoro(self) -> None:
		self.kokoro_active = True
		self.current_mode = GhostBehaviour.SCATTER
		self.start_kokoro_timeout()

	# ----------------------------------------------------------------------------------------------
	def add_score(self, score: ScoreType) -> None:
		self.score += score.value

	# ----------------------------------------------------------------------------------------------
	def get_hero_position(self) -> tuple[int, int]:
		return self.hero.get_position() if self.hero is not None else (0, 0)

	# ----------------------------------------------------------------------------------------------
	def end_game(self) -> None:
		if self.hero in self.game_objects:
			self.game_objects.remove(self.hero)
		self.hero = None

	# ----------------------------------------------------------------------------------------------
	def kill_pacman(self) -> None:
		self.lives -= 1
		# self.hero.set_position(32, 32)
		# self.hero.set_direction(Direction.NONE)
		# if self.lives == 0:
		# 	self.end_game()

	# # ----------------------------------------------------------------------------------------------
	# def display_text(self, text, position=(32, 0), size=30):
	# 	font = pygame.font.SysFont("Arial", size)
	# 	text_surface = font.render(text, False, (255, 255, 255))
	# 	self.screen.blit(text_surface, position)

	# ----------------------------------------------------------------------------------------------
	def add_wall(self, wall: "Wall") -> None:
		self.add_game_object(wall)
		self.walls.append(wall)

	# ----------------------------------------------------------------------------------------------
	def add_hero(self, hero: "Hero | HeroAuto") -> None:
		self.add_game_object(hero)
		self.hero = hero

	# ----------------------------------------------------------------------------------------------
	def handle_events(self) -> None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.done = True
			if event.type == self.mode_switch_event:
				self.handle_mode_switch()
			if event.type == self.kokoro_end_event:
				self.kokoro_active = False
			if event.type == self.pakupaku_event and self.hero is not None:
				self.hero.mouth_open = not self.hero.mouth_open

		if self.hero is None:
			return
		pressed = pygame.key.get_pressed()
		if pressed[pygame.K_UP]:
			self.hero.set_direction(Direction.UP)
		elif pressed[pygame.K_LEFT]:
			self.hero.set_direction(Direction.LEFT)
		elif pressed[pygame.K_DOWN]:
			self.hero.set_direction(Direction.DOWN)
		elif pressed[pygame.K_RIGHT]:
			self.hero.set_direction(Direction.RIGHT)



# ==================================================================================================
class GameObject:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, renderer: "Renderer", x: int, y: int, size: int, colour: tuple[int, int, int] = (255, 0, 0), is_circle: bool = False):
		self.size = size
		self.renderer = renderer
		self.game = self.renderer.game
		self.x = x
		self.y = y
		self.colour = colour
		self.circle = is_circle
		self.shape = pygame.Rect(self.x, self.y, size, size)

	# ----------------------------------------------------------------------------------------------
	def draw(self) -> None:
		if self.circle:
			pygame.draw.circle(self.renderer.surface, self.colour, (self.x, self.y), self.size)
		else:
			rect_object = pygame.Rect(self.x, self.y, self.size, self.size)
			pygame.draw.rect(self.renderer.surface, self.colour, rect_object, border_radius=0)

	# ----------------------------------------------------------------------------------------------
	def tick(self) -> None:
		pass

	# ----------------------------------------------------------------------------------------------
	def get_shape(self) -> pygame.Rect:
		return pygame.Rect(self.x, self.y, self.size, self.size)

	# ----------------------------------------------------------------------------------------------
	def set_position(self, x: int, y: int) -> None:
		self.x = x
		self.y = y

	# ----------------------------------------------------------------------------------------------
	def get_position(self) -> tuple[int, int]:
		return (self.x, self.y)


# ==================================================================================================
class Wall(GameObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, renderer: "Renderer", x: int, y: int, size: int, colour: tuple[int, int, int] = (0, 0, 255)):
		super().__init__(
			renderer=renderer,
			x=x * size,
			y=y * size,
			size=size,
			colour=colour
		)


# ==================================================================================================
class MovableObject(GameObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, renderer: Renderer, x: int, y: int, size: int, colour: tuple[int, int, int] = (255, 0, 0), is_circle: bool = False):
		super().__init__(
			renderer=renderer,
			x=x,
			y=y,
			size=size,
			colour=colour,
			is_circle=is_circle
		)
		self.current_direction = Direction.NONE
		self.direction_buffer = Direction.NONE
		self.last_working_direction = Direction.NONE
		self.location_queue: list[tuple[int, int]] = []
		self.next_target: Optional[tuple[int, int]] = None
		# self.image = pygame.image.load("images/ghost.png")

	# ----------------------------------------------------------------------------------------------
	def get_next_location(self) -> Optional[tuple[int, int]]:
		return None if len(self.location_queue) == 0 else self.location_queue.pop(0)

	# ----------------------------------------------------------------------------------------------
	def set_direction(self, direction: Direction) -> None:
		self.current_direction = direction
		self.direction_buffer = direction

	# ----------------------------------------------------------------------------------------------
	def collides_with_wall(self, position: tuple[int, int]) -> bool:
		collision_rect = pygame.Rect(position[0], position[1], self.size, self.size)
		collides = False
		for wall in self.renderer.walls:
			try:
				collides = collision_rect.colliderect(wall.get_shape())
			except AttributeError as e:
				inspect(wall, all=True)
				raise e
			if collides:
				break

		return collides

	# ----------------------------------------------------------------------------------------------
	def check_collision_direction(self, direction: Direction) -> tuple[bool, tuple[int, int]]:
		desired_position = (0, 0)
		if direction == Direction.NONE:
			return False, desired_position
		if direction == Direction.UP:
			desired_position = (self.x, self.y - 1)
		elif direction == Direction.DOWN:
			desired_position = (self.x, self.y + 1)
		elif direction == Direction.LEFT:
			desired_position = (self.x - 1, self.y)
		elif direction == Direction.RIGHT:
			desired_position = (self.x + 1, self.y)

		return self.collides_with_wall(desired_position), desired_position

	# ----------------------------------------------------------------------------------------------
	def automatic_move(self, direction: Direction) -> None:
		pass

	# ----------------------------------------------------------------------------------------------
	def tick(self) -> None:
		self.reached_target()
		self.automatic_move(self.current_direction)

	# ----------------------------------------------------------------------------------------------
	def reached_target(self) -> None:
		pass

	# # ----------------------------------------------------------------------------------------------
	# def draw(self):
	# 	self.image = pygame.transform.scale(self.image, (32, 32))
	# 	self.surface.blit(self.image, self.get_shape())


# ==================================================================================================
class Hero(MovableObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, renderer: Renderer, x: int, y: int, size: int):
		super().__init__(
			renderer=renderer,
			x=x,
			y=y,
			size=size,
			colour=(255, 255, 0)
		)
		self.last_non_colliding_position = (0, 0)
		# self.open = pygame.image.load("images/paku.png")
		# self.closed = pygame.image.load("images/man.png")
		# self.image = self.open
		self.mouth_open = True

	# ----------------------------------------------------------------------------------------------
	def tick(self) -> None:
		# TELEPORT
		if self.x < 0:
			self.x = self.renderer.width

		if self.x > self.renderer.width:
			self.x = 0

		self.last_non_colliding_position = self.get_position()

		if self.check_collision_direction(self.direction_buffer)[0]:
			self.automatic_move(self.current_direction)
		else:
			self.automatic_move(self.direction_buffer)
			self.current_direction = self.direction_buffer

		if self.collides_with_wall((self.x, self.y)):
			self.set_position(self.last_non_colliding_position[0], self.last_non_colliding_position[1])

		self.handle_cookie_pickup()
		self.handle_ghosts()

	# ----------------------------------------------------------------------------------------------
	def automatic_move(self, direction: Direction) -> None:
		collision_result = self.check_collision_direction(direction)

		desired_position_collides = collision_result[0]
		if not desired_position_collides:
			self.last_working_direction = self.current_direction
			desired_position = collision_result[1]
			self.set_position(desired_position[0], desired_position[1])
		else:
			self.current_direction = self.last_working_direction

	# ----------------------------------------------------------------------------------------------
	def handle_cookie_pickup(self) -> None:
		collision_rect = pygame.Rect(self.x, self.y, self.size, self.size)
		cookies = self.renderer.cookies
		powerups = self.renderer.powerups
		game_objects = self.renderer.game_objects
		cookie_to_remove = None
		for cookie in cookies:
			collides = collision_rect.colliderect(cookie.get_shape())
			if collides and cookie in game_objects:
				game_objects.remove(cookie)
				self.renderer.add_score(ScoreType.COOKIE)
				cookie_to_remove = cookie

		if cookie_to_remove is not None:
			cookies.remove(cookie_to_remove)

		if len(self.renderer.cookies) == 0:
			self.renderer.won = True

		for powerup in powerups:
			collides = collision_rect.colliderect(powerup.get_shape())
			if collides and powerup in game_objects:
				if not self.renderer.kokoro_active:
					game_objects.remove(powerup)
					self.renderer.add_score(ScoreType.POWERUP)
					self.renderer.activate_kokoro()

	# ----------------------------------------------------------------------------------------------
	def handle_ghosts(self) -> None:
		collision_rect = pygame.Rect(self.x, self.y, self.size, self.size)
		ghosts = self.renderer.ghosts
		game_objects = self.renderer.game_objects
		for ghost in ghosts:
			collides = collision_rect.colliderect(ghost.get_shape())
			if collides and ghost in game_objects:
				if self.renderer.kokoro_active:
					game_objects.remove(ghost)
					self.renderer.add_score(ScoreType.GHOST)
				else:
					if not self.renderer.won:
						self.renderer.kill_pacman()

	# ----------------------------------------------------------------------------------------------
	def draw(self) -> None:
		half_size = self.size / 2
		pygame.draw.circle(self.renderer.surface, self.colour, (self.x + half_size, self.y + half_size), half_size)

		# Draw collision box
		# rect = pygame.Rect(self.x, self.y, self.size, self.size)
		# pygame.draw.rect(self.surface, self.colour, rect, width=1)

		# half_size = self.size / 2
		# self.image = self.open if self.mouth_open else self.closed
		# self.image = pygame.transform.rotate(self.image, self.current_direction.value)
		# super(Hero, self).draw()


# ==================================================================================================
class HeroAuto(Hero, MovableObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, renderer: Renderer, x: int, y: int, size: int):
		super().__init__(
			renderer=renderer,
			x=x,
			y=y,
			size=size,
		)

	# ----------------------------------------------------------------------------------------------
	def reached_target(self) -> None:
		# # DEBUG
		# if self.next_target is None:
		# 	# self.request_path(random.choice(self.game.reachable_spaces))
		# 	self.request_path((random.randint(0, self.game.height), random.randint(0, self.game.width)))

		if (self.x, self.y) == self.next_target:
			self.next_target = self.get_next_location()
		self.current_direction = self.calculate_direction_to_next_target()

	# ----------------------------------------------------------------------------------------------
	def set_new_path(self, path: list[tuple[int, int]]) -> None:
		for item in path:
			self.location_queue.append(item)
		self.next_target = self.get_next_location()

	# ----------------------------------------------------------------------------------------------
	def calculate_direction_to_next_target(self) -> Direction:
		if self.next_target is None:
			return Direction.NONE

		diff_x = self.next_target[0] - self.x
		diff_y = self.next_target[1] - self.y
		if diff_x == 0:
			return Direction.DOWN if diff_y > 0 else Direction.UP
		if diff_y == 0:
			return Direction.LEFT if diff_x < 0 else Direction.RIGHT

		return Direction.NONE

	# ----------------------------------------------------------------------------------------------
	def request_path(self, coordinate: tuple[int, int]) -> None:
		player_position = translate_screen_to_maze(self.renderer.get_hero_position())
		path = self.renderer.game.pathfinder.get_path(
			player_position[1], player_position[0],
			coordinate[1], coordinate[0]
		)

		new_path = [translate_maze_to_screen(item) for item in path]
		self.location_queue = []
		self.set_new_path(new_path)

	# ----------------------------------------------------------------------------------------------
	def automatic_move(self, direction: Direction) -> None:
		if direction == Direction.UP:
			self.set_position(self.x, self.y - 1)
		elif direction == Direction.DOWN:
			self.set_position(self.x, self.y + 1)
		elif direction == Direction.LEFT:
			self.set_position(self.x - 1, self.y)
		elif direction == Direction.RIGHT:
			self.set_position(self.x + 1, self.y)

	# ----------------------------------------------------------------------------------------------
	def draw(self) -> None:
		super().draw()
		draw_path(self.renderer, self.location_queue) # DEBUG

	# ----------------------------------------------------------------------------------------------
	def tick(self) -> None:
		# # TELEPORT
		# if self.x < 0:
		# 	self.x = self.renderer.width

		# if self.x > self.renderer.width:
		# 	self.x = 0

		self.reached_target()
		self.automatic_move(self.current_direction)

		self.handle_cookie_pickup()
		self.handle_ghosts()

	# ----------------------------------------------------------------------------------------------
	def handle_cookie_pickup(self) -> None:
		collision_rect = pygame.Rect(self.x, self.y, self.size, self.size)
		cookies = self.renderer.cookies
		powerups = self.renderer.powerups
		game_objects = self.renderer.game_objects
		cookie_to_remove = None
		for cookie in cookies:
			collides = collision_rect.colliderect(cookie.get_shape())
			if collides and cookie in game_objects:
				game_objects.remove(cookie)
				self.renderer.add_score(ScoreType.COOKIE)
				cookie_to_remove = cookie

		if cookie_to_remove is not None:
			cookies.remove(cookie_to_remove)

		if len(self.renderer.cookies) == 0:
			self.renderer.won = True

		for powerup in powerups:
			collides = collision_rect.colliderect(powerup.get_shape())
			if collides and powerup in game_objects:
				if not self.renderer.kokoro_active:
					game_objects.remove(powerup)
					self.renderer.add_score(ScoreType.POWERUP)
					self.renderer.activate_kokoro()

	# ----------------------------------------------------------------------------------------------
	def handle_ghosts(self) -> None:
		collision_rect = pygame.Rect(self.x, self.y, self.size, self.size)
		ghosts = self.renderer.ghosts
		game_objects = self.renderer.game_objects
		for ghost in ghosts:
			collides = collision_rect.colliderect(ghost.get_shape())
			if collides and ghost in game_objects:
				if self.renderer.kokoro_active:
					game_objects.remove(ghost)
					self.renderer.add_score(ScoreType.GHOST)
				else:
					if not self.renderer.won:
						self.renderer.kill_pacman()


# ==================================================================================================
class Ghost(MovableObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, renderer: Renderer, x: int, y: int, size: int, colour: tuple[int, int, int] = (255, 0, 0)):
		super().__init__(
			renderer=renderer,
			x=x,
			y=y,
			size=size,
			colour=colour)
		# super().__init__(renderer, x, y, size)
		# self.sprite_normal = pygame.image.load(sprite_path)
		# self.sprite_fright = pygame.image.load("images/ghost_fright.png")
		self.padding = unified_size // 8

	# ----------------------------------------------------------------------------------------------
	def reached_target(self) -> None:
		if (self.x, self.y) == self.next_target:
			self.next_target = self.get_next_location()
		self.current_direction = self.calculate_direction_to_next_target()

	# ----------------------------------------------------------------------------------------------
	def set_new_path(self, path: list[tuple[int, int]]) -> None:
		for item in path:
			self.location_queue.append(item)
		self.next_target = self.get_next_location()

	# ----------------------------------------------------------------------------------------------
	def calculate_direction_to_next_target(self) -> Direction:
		if self.next_target is None:
			if self.renderer.current_mode == GhostBehaviour.CHASE and not self.renderer.kokoro_active:
				self.request_path_to_player()
			else:
				self.request_new_random_path()
			return Direction.NONE

		diff_x = self.next_target[0] - self.x
		diff_y = self.next_target[1] - self.y
		if diff_x == 0:
			return Direction.DOWN if diff_y > 0 else Direction.UP
		if diff_y == 0:
			return Direction.LEFT if diff_x < 0 else Direction.RIGHT

		if self.renderer.current_mode == GhostBehaviour.CHASE and not self.renderer.kokoro_active:
			self.request_path_to_player()
		else:
			self.request_new_random_path()

		return Direction.NONE

	# ----------------------------------------------------------------------------------------------
	def request_path_to_player(self) -> None:
		player_position = translate_screen_to_maze(self.renderer.get_hero_position())
		current_maze_coord = translate_screen_to_maze(self.get_position())
		path = self.game.pathfinder.get_path(
			current_maze_coord[1], current_maze_coord[0],
			player_position[1], player_position[0]
		)
		screen_path = [translate_maze_to_screen(item) for item in path]
		self.set_new_path(screen_path)

	# ----------------------------------------------------------------------------------------------
	def request_new_random_path(self):
		random_space = random.choice(self.game.reachable_spaces)
		current_maze_coord = translate_screen_to_maze(self.get_position())
		path = self.game.pathfinder.get_path(
			current_maze_coord[1], current_maze_coord[0],
			random_space[1], random_space[0]
		)
		screen_path = [translate_maze_to_screen(item) for item in path]
		self.set_new_path(screen_path)

	# ----------------------------------------------------------------------------------------------
	def automatic_move(self, direction: Direction) -> None:
		if direction == Direction.UP:
			self.set_position(self.x, self.y - 1)
		elif direction == Direction.DOWN:
			self.set_position(self.x, self.y + 1)
		elif direction == Direction.LEFT:
			self.set_position(self.x - 1, self.y)
		elif direction == Direction.RIGHT:
			self.set_position(self.x + 1, self.y)

	# ----------------------------------------------------------------------------------------------
	def draw(self):
		rect_object = pygame.Rect(self.x + self.padding, self.y + self.padding, self.size, self.size)
		colour = (0, 255, 255) if self.renderer.kokoro_active else self.colour
		pygame.draw.rect(self.renderer.surface, colour, rect_object, border_radius=0)
		# self.image = self.sprite_fright if self.renderer.is_kokoro_active() else self.sprite_normal
		# super(Ghost, self).draw()


# ==================================================================================================
class Cookie(GameObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, renderer: Renderer, x: int, y: int):
		super().__init__(
			renderer=renderer,
			x=x,
			y=y,
			size=unified_size // 8,
			colour=(255, 185, 175),
			is_circle=True
		)


# ==================================================================================================
class Powerup(GameObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, renderer: Renderer, x: int, y: int):
		super().__init__(
			renderer=renderer,
			x=x,
			y=y,
			size=unified_size // 2,
			colour=(255, 185, 175),
			is_circle=True
		)


# ==================================================================================================
class Pathfinder:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, arr: np.ndarray):
		cost = np.array(arr, dtype=np.bool_).tolist()
		self.pf = tcod.path.AStar(cost=cost, diagonal=0)

	# ----------------------------------------------------------------------------------------------
	def get_path(self, from_x: int, from_y: int, to_x: int, to_y: int) -> list[tuple[int, int]]:
		return [(sub[1], sub[0]) for sub in self.pf.get_path(from_x, from_y, to_x, to_y)]


# ==================================================================================================
class GameController:
	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		# [G]hosts and [P]acman take up more spaces than one letter (originally 2x3)
		self.ascii_maze = [
			"████████████████████████████",
			"█••••••••••••██••••••••••••█",
			"█•████•█████•██•█████•████•█",
			"█O████•█████•██•█████•████O█",
			"█•████•█████•██•█████•████•█",
			"█••••••••••••••••••••••••••█",
			"█•████•██•████████•██•████•█",
			"█•████•██•████████•██•████•█",
			"█••••••██••••██••••██••••••█",
			"██████•█████ ██ █████•██████",
			"██████•█████ ██ █████•██████",
			"██████•██    G     ██•██████",
			"██████•██ ████████ ██•██████",
			"██████•██ █      █ ██•██████",
			"      •   █G G G █   •      ",
			"██████•██ █      █ ██•██████",
			"██████•██ ████████ ██•██████",
			"██████•██          ██•██████",
			"██████•██ ████████ ██•██████",
			"██████•██ ████████ ██•██████",
			"█••••••••••••██••••••••••••█",
			"█•████•█████•██•█████•████•█",
			"█•████•█████•██•█████•████•█",
			"█O••██•••••••P •••••••██••O█",
			"███•██•██•████████•██•██•███",
			"███•██•██•████████•██•██•███",
			"█••••••██••••██••••██••••••█",
			"█•██████████•██•██████████•█",
			"█•██████████•██•██████████•█",
			"█••••••••••••••••••••••••••█",
			"████████████████████████████",
		]

		self.cookie_spaces = []
		self.powerup_spaces = []
		self.reachable_spaces = []
		self.ghost_spawns = []
		self.ghost_colours = [
			(255, 184, 255),
			(255, 0, 20),
			(0, 255, 255),
			(255, 184, 82),
			# "images/ghost.png",
			# "images/ghost_pink.png",
			# "images/ghost_orange.png",
			# "images/ghost_blue.png",
		]
		self.maze = self.convert_maze_to_numpy(self.ascii_maze)
		self.pathfinder = Pathfinder(self.maze)
		self.height, self.width = self.maze.shape

	# ----------------------------------------------------------------------------------------------
	def convert_maze_to_numpy(self, ascii_maze: list[str]) -> np.ndarray:
		maze = []
		for y, row in enumerate(ascii_maze):
			binary_row = []
			for x, column in enumerate(row):
				if column == "G":
					self.ghost_spawns.append((x + 0.5, y))
				elif column == "P":
					self.pacman_spawn = (x + 0.5, y)
				elif column == "•":
					self.cookie_spaces.append((x, y))
				elif column == "O":
					self.powerup_spaces.append((x, y))
				if column == "█":
					binary_row.append(0)
				else:
					binary_row.append(1)
					self.reachable_spaces.append((x, y))
			maze.append(binary_row)

		return np.array(maze)


# ==================================================================================================
def draw_path(renderer: Renderer, path_array: list[tuple[int, int]]):
	white = (255, 255, 255)
	half_size = unified_size / 2
	for i in range(len(path_array) - 1):
		start = [c + half_size for c in path_array[i]]
		end = [c + half_size for c in path_array[i + 1]]
		pygame.draw.line(renderer.surface, white, start, end)


# ==================================================================================================
class MyPacmanRGB(gym.Env):
	metadata = {
		"render_modes": ["human", "rgb_array"],
		"render_fps": 120,
	}
	actions = [Direction.UP, Direction.RIGHT, Direction.LEFT, Direction.DOWN]
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, render_mode: Optional[str] = None):
		"""
		RGB version of MountainCar environment.

		Args:
			args (argparse.Namespace): Parsed arguments, depends on which specific env we're using.
		"""
		# Don't know why I have to do this
		# When I look at MountainCar I don't think they do this
		# But otherwise it complains that "pygame is not initialized" when I did call pygame.init()
		import os
		# import sys
		os.environ["SDL_VIDEODRIVER"] = "dummy"

		self.screen = None
		self.game = GameController()
		self.renderer = Renderer(self.game)

		self.render_mode = "rgb_array"
		self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self.renderer.height, self.renderer.width, 3))
		self.action_space = spaces.Discrete(len(self.actions))

		self.last_score = 0

	# ----------------------------------------------------------------------------------------------
	def render(self) -> np.ndarray:
		self.renderer.surface.fill((0, 0, 0))
		for game_object in self.renderer.game_objects:
			game_object.draw()

		return np.transpose(np.array(pygame.surfarray.pixels3d(self.renderer.surface)), axes=(1, 0, 2))

	# ----------------------------------------------------------------------------------------------
	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[ObsType, dict[str, Any]]:
		super().reset(seed=seed)

		self.game = GameController()
		self.renderer = Renderer(self.game)
		self.last_score = 0

		self.renderer.handle_mode_switch()
		pygame.time.set_timer(self.renderer.pakupaku_event, 200) # open close mouth

		return self.render(), self.get_info()

	# ----------------------------------------------------------------------------------------------
	def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

		direction = self.actions[action]
		assert self.renderer.hero is not None
		self.renderer.hero.set_direction(direction)

		for game_object in self.renderer.game_objects:
			game_object.tick()
		self.render()
		self.renderer.clock.tick(self.metadata["render_fps"])
		self.renderer.handle_events()

		terminated = (self.renderer.lives == 0) or self.renderer.won
		reward = self.renderer.score - self.last_score
		self.last_score = self.renderer.score

		return self.render(), reward, terminated, False, self.get_info()

	# ----------------------------------------------------------------------------------------------
	def close(self) -> None:
		if self.screen is not None:
			import pygame

			pygame.display.quit()
			pygame.quit()
			self.isopen = False

	# ----------------------------------------------------------------------------------------------
	def get_info(self) -> dict[str, Any]:
		"""
		Return a created dictionary for the step info.

		Returns:
			dict: Key-value pairs for the step info.
		"""
		return {
			# "state": self.state, # Used later for bootstrap loss
			"state": 0, # TODO placeholder
		}


# ==================================================================================================
class MyPacmanRGBpp(MyPacmanRGB):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, render_mode: Optional[str] = None):
		super().__init__(args=args, render_mode=render_mode)
		# self.action_space = spaces.Tuple((spaces.Box(low=0, high=self.game.width), spaces.Box(low=0, high=self.game.height)))
		self.action_space = spaces.MultiDiscrete(np.array([self.game.height, self.game.width]))

		self.renderer = Renderer(self.game, auto=True)

	# ----------------------------------------------------------------------------------------------
	def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

		assert self.renderer.hero is not None
		self.renderer.hero.request_path(action)

		for game_object in self.renderer.game_objects:
			game_object.tick()
		self.render()
		self.renderer.clock.tick(self.metadata["render_fps"])
		self.renderer.handle_events()

		terminated = (self.renderer.lives == 0) or self.renderer.won
		reward = self.renderer.score - self.last_score
		self.last_score = self.renderer.score

		return self.render(), reward, terminated, False, self.get_info()

	# ----------------------------------------------------------------------------------------------
	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[ObsType, dict[str, Any]]:
		super().reset(seed=seed)

		self.game = GameController()
		self.renderer = Renderer(self.game, auto=True)
		self.last_score = 0

		self.renderer.handle_mode_switch()
		pygame.time.set_timer(self.renderer.pakupaku_event, 200) # open close mouth

		return self.render(), self.get_info()


# ==================================================================================================
if __name__ == "__main__":
	random.seed(0)

	game = GameController()
	# game_renderer = Renderer(game)
	game_renderer = Renderer(game, auto=True)

	game_renderer.tick(120)
