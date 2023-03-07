#!/usr/bin/env python3
import pygame
import numpy as np
import tcod
import gymnasium as gym
from gymnasium import spaces

unified_size = 8

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
def translate_screen_to_maze(in_coords, in_size=unified_size):
	return int(in_coords[0] / in_size), int(in_coords[1] / in_size)


# ==================================================================================================
def translate_maze_to_screen(in_coords, in_size=unified_size):
	return in_coords[0] * in_size, in_coords[1] * in_size


# ==================================================================================================
# Untested
def draw_path(path_array):
	white = (255, 255, 255)
	for path in path_array:
		game_renderer.add_game_object(Wall(game_renderer, path[0], path[1], unified_size, white))

	# Untested
	# _from = path_array[0]
	# _to = path_array[-1]
	# from_translated = translate_maze_to_screen(_from)
	# game_renderer.add_game_object(GameObject(game_renderer, from_translated[0], from_translated[1], unified_size, red))
	# to_translated = translate_maze_to_screen(_to)
	# game_renderer.add_game_object(GameObject(game_renderer, to_translated[0], to_translated[1], unified_size, green))


# ==================================================================================================
class GameObject:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_surface, x, y, in_size: int, in_color=(255, 0, 0), is_circle: bool = False):
		self._size = in_size
		self._renderer: GameRenderer = in_surface
		self._surface = in_surface._screen
		self.x = x
		self.y = y
		self._color = in_color
		self._circle = is_circle
		self._shape = pygame.Rect(self.x, self.y, in_size, in_size)

	# ----------------------------------------------------------------------------------------------
	def draw(self):
		if self._circle:
			pygame.draw.circle(self._surface, self._color, (self.x, self.y), self._size)
		else:
			rect_object = pygame.Rect(self.x, self.y, self._size, self._size)
			pygame.draw.rect(self._surface, self._color, rect_object, border_radius=0)

	# ----------------------------------------------------------------------------------------------
	def tick(self):
		pass

	# ----------------------------------------------------------------------------------------------
	def get_shape(self):
		return pygame.Rect(self.x, self.y, self._size, self._size)

	# ----------------------------------------------------------------------------------------------
	def set_position(self, in_x, in_y):
		self.x = in_x
		self.y = in_y

	# ----------------------------------------------------------------------------------------------
	def get_position(self):
		return (self.x, self.y)


# ==================================================================================================
class Wall(GameObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_surface, x, y, in_size: int, in_color=(0, 0, 255)):
		super().__init__(in_surface, x * in_size, y * in_size, in_size, in_color)


# ==================================================================================================
class GameRenderer:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_width: int, in_height: int):
		pygame.init()
		self._width = in_width
		self._height = in_height
		self._screen = pygame.display.set_mode((in_width, in_height))
		pygame.display.set_caption("Pacman")
		self._clock = pygame.time.Clock()
		self._done = False
		self._won = False
		self._game_objects = []
		self._walls = []
		self._cookies = []
		self._powerups = []
		self._ghosts = []
		self._hero: Hero = None
		self._lives = 3
		self._score = 0
		self._kokoro_active = False # powerup, special ability
		self._current_mode = GhostBehaviour.SCATTER
		self._mode_switch_event = pygame.USEREVENT + 1 # custom event
		self._kokoro_end_event = pygame.USEREVENT + 2
		self._pakupaku_event = pygame.USEREVENT + 3
		self._modes = [
			(7, 20),
			(7, 20),
			(5, 20),
			(5, 999999) # 'infinite' chase seconds
		]
		self._current_phase = 0

	# ----------------------------------------------------------------------------------------------
	def tick(self, in_fps: int):
		black = (0, 0, 0)
		self.handle_mode_switch()
		pygame.time.set_timer(self._pakupaku_event, 200) # open close mouth
		while not self._done:
			for game_object in self._game_objects:
				game_object.tick()
				game_object.draw()

			self.display_text(f"[Score: {self._score}]  [Lives: {self._lives}]")

			if self._hero is None:
				self.display_text("YOU DIED", (self._width / 2 - 256, self._height / 2 - 256), 100)
			if self.get_won():
				self.display_text("YOU WON", (self._width / 2 - 256, self._height / 2 - 256), 100)
			pygame.display.flip()
			self._clock.tick(in_fps)
			self._screen.fill(black)
			self._handle_events()
		print("Game over")

	# ----------------------------------------------------------------------------------------------
	def handle_mode_switch(self):
		current_phase_timings = self._modes[self._current_phase]
		print(f"Current phase: {str(self._current_phase)}, current_phase_timings: {str(current_phase_timings)}")
		scatter_timing = current_phase_timings[0]
		chase_timing = current_phase_timings[1]

		if self._current_mode == GhostBehaviour.CHASE:
			self._current_phase += 1
			self.set_current_mode(GhostBehaviour.SCATTER)
		else:
			self.set_current_mode(GhostBehaviour.CHASE)

		used_timing = scatter_timing if self._current_mode == GhostBehaviour.SCATTER else chase_timing
		pygame.time.set_timer(self._mode_switch_event, used_timing * 1000)

	# ----------------------------------------------------------------------------------------------
	def start_kokoro_timeout(self):
		pygame.time.set_timer(self._kokoro_end_event, 15000) # 15s

	# ----------------------------------------------------------------------------------------------
	def add_game_object(self, obj: GameObject):
		self._game_objects.append(obj)

	# ----------------------------------------------------------------------------------------------
	def add_cookie(self, obj: GameObject):
		self._game_objects.append(obj)
		self._cookies.append(obj)

	# ----------------------------------------------------------------------------------------------
	def add_ghost(self, obj: GameObject):
		self._game_objects.append(obj)
		self._ghosts.append(obj)

	# ----------------------------------------------------------------------------------------------
	def add_powerup(self, obj: GameObject):
		self._game_objects.append(obj)
		self._powerups.append(obj)

	# ----------------------------------------------------------------------------------------------
	def activate_kokoro(self):
		self._kokoro_active = True
		self.set_current_mode(GhostBehaviour.SCATTER)
		self.start_kokoro_timeout()

	# ----------------------------------------------------------------------------------------------
	def set_won(self):
		self._won = True

	# ----------------------------------------------------------------------------------------------
	def get_won(self):
		return self._won

	# ----------------------------------------------------------------------------------------------
	def add_score(self, in_score: ScoreType):
		self._score += in_score.value

	# ----------------------------------------------------------------------------------------------
	def get_hero_position(self):
		return self._hero.get_position() if self._hero != None else (0, 0)

	# ----------------------------------------------------------------------------------------------
	def set_current_mode(self, in_mode: GhostBehaviour):
		self._current_mode = in_mode

	# ----------------------------------------------------------------------------------------------
	def get_current_mode(self):
		return self._current_mode

	# ----------------------------------------------------------------------------------------------
	def end_game(self):
		if self._hero in self._game_objects:
			self._game_objects.remove(self._hero)
		self._hero = None

	# ----------------------------------------------------------------------------------------------
	def kill_pacman(self):
		self._lives -= 1
		self._hero.set_position(32, 32)
		self._hero.set_direction(Direction.NONE)
		if self._lives == 0: self.end_game()

	# ----------------------------------------------------------------------------------------------
	def display_text(self, text, in_position=(32, 0), in_size=30):
		font = pygame.font.SysFont('Arial', in_size)
		text_surface = font.render(text, False, (255, 255, 255))
		self._screen.blit(text_surface, in_position)

	# ----------------------------------------------------------------------------------------------
	def is_kokoro_active(self):
		return self._kokoro_active

	# ----------------------------------------------------------------------------------------------
	def add_wall(self, obj: Wall):
		self.add_game_object(obj)
		self._walls.append(obj)

	# ----------------------------------------------------------------------------------------------
	def get_walls(self):
		return self._walls

	# ----------------------------------------------------------------------------------------------
	def get_cookies(self):
		return self._cookies

	# ----------------------------------------------------------------------------------------------
	def get_ghosts(self):
		return self._ghosts

	# ----------------------------------------------------------------------------------------------
	def get_powerups(self):
		return self._powerups

	# ----------------------------------------------------------------------------------------------
	def get_game_objects(self):
		return self._game_objects

	# ----------------------------------------------------------------------------------------------
	def add_hero(self, in_hero):
		self.add_game_object(in_hero)
		self._hero = in_hero

	# ----------------------------------------------------------------------------------------------
	def _handle_events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self._done = True
			if event.type == self._mode_switch_event:
				self.handle_mode_switch()
			if event.type == self._kokoro_end_event:
				self._kokoro_active = False
			if event.type == self._pakupaku_event:
				if self._hero is None:
					break
				self._hero.mouth_open = not self._hero.mouth_open

		pressed = pygame.key.get_pressed()
		if pressed[pygame.K_UP]:
			self._hero.set_direction(Direction.UP)
		elif pressed[pygame.K_LEFT]:
			self._hero.set_direction(Direction.LEFT)
		elif pressed[pygame.K_DOWN]:
			self._hero.set_direction(Direction.DOWN)
		elif pressed[pygame.K_RIGHT]:
			self._hero.set_direction(Direction.RIGHT)


# ==================================================================================================
class MovableObject(GameObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_surface, x, y, in_size: int, in_color=(255, 0, 0), is_circle: bool = False):
		super().__init__(in_surface, x, y, in_size, in_color, is_circle)
		self.current_direction = Direction.NONE
		self.direction_buffer = Direction.NONE
		self.last_working_direction = Direction.NONE
		self.location_queue = []
		self.next_target = None
		# self.image = pygame.image.load("images/ghost.png")

	# ----------------------------------------------------------------------------------------------
	def get_next_location(self):
		return None if len(self.location_queue) == 0 else self.location_queue.pop(0)

	# ----------------------------------------------------------------------------------------------
	def set_direction(self, in_direction):
		self.current_direction = in_direction
		self.direction_buffer = in_direction

	# ----------------------------------------------------------------------------------------------
	def collides_with_wall(self, in_position):
		collision_rect = pygame.Rect(in_position[0], in_position[1], self._size, self._size)
		collides = False
		walls = self._renderer.get_walls()
		for wall in walls:
			collides = collision_rect.colliderect(wall.get_shape())
			if collides: break
		return collides

	# ----------------------------------------------------------------------------------------------
	def check_collision_in_direction(self, in_direction: Direction):
		desired_position = (0, 0)
		if in_direction == Direction.NONE: return False, desired_position
		if in_direction == Direction.UP:
			desired_position = (self.x, self.y - 1)
		elif in_direction == Direction.DOWN:
			desired_position = (self.x, self.y + 1)
		elif in_direction == Direction.LEFT:
			desired_position = (self.x - 1, self.y)
		elif in_direction == Direction.RIGHT:
			desired_position = (self.x + 1, self.y)

		return self.collides_with_wall(desired_position), desired_position

	# ----------------------------------------------------------------------------------------------
	def automatic_move(self, in_direction: Direction):
		pass

	# ----------------------------------------------------------------------------------------------
	def tick(self):
		self.reached_target()
		self.automatic_move(self.current_direction)

	# ----------------------------------------------------------------------------------------------
	def reached_target(self):
		pass

	# # ----------------------------------------------------------------------------------------------
	# def draw(self):
	# 	self.image = pygame.transform.scale(self.image, (32, 32))
	# 	self._surface.blit(self.image, self.get_shape())


# ==================================================================================================
class Hero(MovableObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_surface, x, y, in_size: int):
		super().__init__(in_surface, x, y, in_size, (255, 255, 0), False)
		self.last_non_colliding_position = (0, 0)
		# self.open = pygame.image.load("images/paku.png")
		# self.closed = pygame.image.load("images/man.png")
		# self.image = self.open
		self.mouth_open = True

	# ----------------------------------------------------------------------------------------------
	def tick(self):
		# TELEPORT
		if self.x < 0:
			self.x = self._renderer._width

		if self.x > self._renderer._width:
			self.x = 0

		self.last_non_colliding_position = self.get_position()

		if self.check_collision_in_direction(self.direction_buffer)[0]:
			self.automatic_move(self.current_direction)
		else:
			self.automatic_move(self.direction_buffer)
			self.current_direction = self.direction_buffer

		if self.collides_with_wall((self.x, self.y)):
			self.set_position(self.last_non_colliding_position[0], self.last_non_colliding_position[1])

		self.handle_cookie_pickup()
		# self.handle_powerup_pickup()
		# self.handle_ghost_collision()
		self.handle_ghosts()

	# ----------------------------------------------------------------------------------------------
	def automatic_move(self, in_direction: Direction):
		collision_result = self.check_collision_in_direction(in_direction)

		desired_position_collides = collision_result[0]
		if not desired_position_collides:
			self.last_working_direction = self.current_direction
			desired_position = collision_result[1]
			self.set_position(desired_position[0], desired_position[1])
		else:
			self.current_direction = self.last_working_direction

	# ----------------------------------------------------------------------------------------------
	def handle_cookie_pickup(self):
		collision_rect = pygame.Rect(self.x, self.y, self._size, self._size)
		cookies = self._renderer.get_cookies()
		powerups = self._renderer.get_powerups()
		game_objects = self._renderer.get_game_objects()
		cookie_to_remove = None
		for cookie in cookies:
			collides = collision_rect.colliderect(cookie.get_shape())
			if collides and cookie in game_objects:
				game_objects.remove(cookie)
				self._renderer.add_score(ScoreType.COOKIE)
				cookie_to_remove = cookie

		if cookie_to_remove is not None:
			cookies.remove(cookie_to_remove)

		if len(self._renderer.get_cookies()) == 0:
			self._renderer.set_won()

		for powerup in powerups:
			collides = collision_rect.colliderect(powerup.get_shape())
			if collides and powerup in game_objects:
				if not self._renderer.is_kokoro_active():
					game_objects.remove(powerup)
					self._renderer.add_score(ScoreType.POWERUP)
					self._renderer.activate_kokoro()

	# ----------------------------------------------------------------------------------------------
	def handle_ghosts(self):
		collision_rect = pygame.Rect(self.x, self.y, self._size, self._size)
		ghosts = self._renderer.get_ghosts()
		game_objects = self._renderer.get_game_objects()
		for ghost in ghosts:
			collides = collision_rect.colliderect(ghost.get_shape())
			if collides and ghost in game_objects:
				if self._renderer.is_kokoro_active():
					game_objects.remove(ghost)
					self._renderer.add_score(ScoreType.GHOST)
				else:
					if not self._renderer.get_won():
						self._renderer.kill_pacman()

	# ----------------------------------------------------------------------------------------------
	def draw(self):
		half_size = self._size / 2
		pygame.draw.circle(self._surface, self._color, (self.x + half_size, self.y + half_size), half_size)

		# Dra collision box
		# rect = pygame.Rect(self.x, self.y, self._size, self._size)
		# pygame.draw.rect(self._surface, self._color, rect, width=1)

		# half_size = self._size / 2
		# self.image = self.open if self.mouth_open else self.closed
		# self.image = pygame.transform.rotate(self.image, self.current_direction.value)
		# super(Hero, self).draw()


# ==================================================================================================
class Ghost(MovableObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_surface, x, y, in_size: int, in_game_controller, in_color=(255, 0, 0)):
	# def __init__(self, in_surface, x, y, in_size: int, in_game_controller, sprite_path="images/ghost_fright.png"):
		super().__init__(in_surface, x, y, in_size, in_color, False)
		# super().__init__(in_surface, x, y, in_size)
		self.game_controller = in_game_controller
		# self.sprite_normal = pygame.image.load(sprite_path)
		# self.sprite_fright = pygame.image.load("images/ghost_fright.png")
		self.padding = unified_size // 8

	# ----------------------------------------------------------------------------------------------
	def reached_target(self):
		if (self.x, self.y) == self.next_target:
			self.next_target = self.get_next_location()
		self.current_direction = self.calculate_direction_to_next_target()

	# ----------------------------------------------------------------------------------------------
	def set_new_path(self, in_path):
		for item in in_path:
			self.location_queue.append(item)
		self.next_target = self.get_next_location()

	# ----------------------------------------------------------------------------------------------
	def calculate_direction_to_next_target(self) -> Direction:
		if self.next_target is None:
			if self._renderer.get_current_mode() == GhostBehaviour.CHASE and not self._renderer.is_kokoro_active():
				self.request_path_to_player(self)
			else:
				self.game_controller.request_new_random_path(self)
			return Direction.NONE

		diff_x = self.next_target[0] - self.x
		diff_y = self.next_target[1] - self.y
		if diff_x == 0:
			return Direction.DOWN if diff_y > 0 else Direction.UP
		if diff_y == 0:
			return Direction.LEFT if diff_x < 0 else Direction.RIGHT

		if self._renderer.get_current_mode() == GhostBehaviour.CHASE and not self._renderer.is_kokoro_active():
			self.request_path_to_player(self)
		else:
			self.game_controller.request_new_random_path(self)

		return Direction.NONE

	# ----------------------------------------------------------------------------------------------
	def request_path_to_player(self, in_ghost):
		player_position = translate_screen_to_maze(in_ghost._renderer.get_hero_position())
		current_maze_coord = translate_screen_to_maze(in_ghost.get_position())
		path = self.game_controller.p.get_path(
			current_maze_coord[1], current_maze_coord[0],
			player_position[1], player_position[0]
		)

		new_path = [translate_maze_to_screen(item) for item in path]
		in_ghost.set_new_path(new_path)

	# ----------------------------------------------------------------------------------------------
	def automatic_move(self, in_direction: Direction):
		if in_direction == Direction.UP:
			self.set_position(self.x, self.y - 1)
		elif in_direction == Direction.DOWN:
			self.set_position(self.x, self.y + 1)
		elif in_direction == Direction.LEFT:
			self.set_position(self.x - 1, self.y)
		elif in_direction == Direction.RIGHT:
			self.set_position(self.x + 1, self.y)

	# ----------------------------------------------------------------------------------------------
	def draw(self):
		rect_object = pygame.Rect(self.x + self.padding, self.y + self.padding, self._size, self._size)
		color = (0, 255, 255) if self._renderer.is_kokoro_active() else self._color
		pygame.draw.rect(self._surface, color, rect_object, border_radius=0)
		# self.image = self.sprite_fright if self._renderer.is_kokoro_active() else self.sprite_normal
		# super(Ghost, self).draw()


# ==================================================================================================
class Cookie(GameObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_surface, x, y):
		super().__init__(in_surface, x, y, unified_size // 8, (255, 185, 175), True)


# ==================================================================================================
class Powerup(GameObject):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_surface, x, y):
		super().__init__(in_surface, x, y, unified_size // 2, (255, 185, 175), True)


# ==================================================================================================
class Pathfinder:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_arr):
		cost = np.array(in_arr, dtype=np.bool_).tolist()
		self.pf = tcod.path.AStar(cost=cost, diagonal=0)

	# ----------------------------------------------------------------------------------------------
	def get_path(self, from_x, from_y, to_x, to_y) -> object:
		res = self.pf.get_path(from_x, from_y, to_x, to_y)
		return [(sub[1], sub[0]) for sub in res]


# ==================================================================================================
class PacmanGameController:
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

		self.numpy_maze = []
		self.cookie_spaces = []
		self.powerup_spaces = []
		self.reachable_spaces = []
		self.ghost_spawns = []
		self.ghost_colors = [
			(255, 184, 255),
			(255, 0, 20),
			(0, 255, 255),
			(255, 184, 82),
			# "images/ghost.png",
			# "images/ghost_pink.png",
			# "images/ghost_orange.png",
			# "images/ghost_blue.png",
		]
		self.size = (0, 0)
		self.convert_maze_to_numpy()
		self.p = Pathfinder(self.numpy_maze)

	# ----------------------------------------------------------------------------------------------
	def request_new_random_path(self, in_ghost: Ghost):
		random_space = random.choice(self.reachable_spaces)
		current_maze_coord = translate_screen_to_maze(in_ghost.get_position())

		path = self.p.get_path(current_maze_coord[1], current_maze_coord[0], random_space[1], random_space[0])
		test_path = [translate_maze_to_screen(item) for item in path]
		in_ghost.set_new_path(test_path)

	# ----------------------------------------------------------------------------------------------
	def convert_maze_to_numpy(self):
		for y, row in enumerate(self.ascii_maze):
			self.size = (len(row), y + 1)
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
			self.numpy_maze.append(binary_row)


# ==================================================================================================
if __name__ == "__main__":
	random.seed(0)

	pacman_game = PacmanGameController()
	size = pacman_game.size
	game_renderer = GameRenderer(size[0] * unified_size, size[1] * unified_size)

	for y, row in enumerate(pacman_game.numpy_maze):
		for x, column in enumerate(row):
			if column == 0:
				game_renderer.add_wall(Wall(game_renderer, x, y, unified_size))

	for cookie_space in pacman_game.cookie_spaces:
		translated = translate_maze_to_screen(cookie_space)
		cookie = Cookie(game_renderer, translated[0] + unified_size / 2, translated[1] + unified_size / 2)
		game_renderer.add_cookie(cookie)

	for powerup_space in pacman_game.powerup_spaces:
		translated = translate_maze_to_screen(powerup_space)
		powerup = Powerup(game_renderer, translated[0] + unified_size / 2, translated[1] + unified_size / 2)
		game_renderer.add_powerup(powerup)

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
		pygame.init()
		self.game = GameController()
		self.renderer = Renderer(self.game)

		self.render_mode = "rgb_array"
		self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self.renderer.height, self.renderer.width, 3))
		self.action_space = spaces.Discrete(len(self.actions))

		self.last_score = 0

	# ----------------------------------------------------------------------------------------------
	def render(self):
		self.game_renderer.screen.fill((0,0,0))
		for game_object in self.game_renderer.game_objects:
			game_object.draw()

		return np.transpose(np.array(pygame.surfarray.pixels3d(self.game_renderer.screen)), axes=(1, 0, 2))

	# ----------------------------------------------------------------------------------------------
	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
		super().reset(seed=seed)

		self.game = GameController()
		self.game_renderer = Renderer(self.game)
		self.last_score = 0

		self.game_renderer.handle_mode_switch()
		pygame.time.set_timer(self.game_renderer.pakupaku_event, 200) # open close mouth

		return self.render(), self.get_info()

	# ----------------------------------------------------------------------------------------------
	def step(self, action: int):
		assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

		direction = self.actions[action]
		assert self.game_renderer.hero is not None
		self.game_renderer.hero.set_direction(direction)

		for game_object in self.game_renderer.game_objects:
			game_object.tick()
		self.render()
		self.game_renderer.clock.tick(self.metadata["render_fps"])
		self.game_renderer.handle_events()

		terminated = (self.game_renderer.lives == 0) or self.game_renderer.won
		reward = self.game_renderer.score - self.last_score
		self.last_score = self.game_renderer.score

		return self.render(), reward, terminated, False, self.get_info()

	# ----------------------------------------------------------------------------------------------
	def close(self):
		if self.screen is not None:
			import pygame

			pygame.display.quit()
			pygame.quit()
			self.isopen = False

	# ----------------------------------------------------------------------------------------------
	def get_info(self) -> dict:
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
if __name__ == "__main__":
	random.seed(0)

	game = GameController()
	game_renderer = Renderer(game)

	game_renderer.tick(120)
