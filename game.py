
# DO NOT MODIFY THIS FILE

import pygame
import sys
import math
from grid import grid
from algo import get_next_coordinate
from constants import BLACK, BLUE, RED, WHITE, YELLOW, o, e, I, O, c

GRID_SIZE = 20
GRID_WIDTH = len(grid)
GRID_HEIGHT = len(grid[0])

pygame.init()

WINDOW_WIDTH = GRID_WIDTH * GRID_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * GRID_SIZE
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("6ix-pac Assignment")

pacman_x = 14
pacman_y = 7

clock = pygame.time.Clock()

start_time = pygame.time.get_ticks()
score = 0

# Main game loop
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(BLACK)

    current_time = pygame.time.get_ticks()
    elapsed_time = (current_time - start_time) // 1000

    current_x, current_y = pacman_x, pacman_y
    path = get_next_coordinate(grid, (current_x, current_y))

    if path is None or len(path) == 0 or path[0] is None:
        print("Invalid Path. Try something else")
        running = False
        continue

    pacman_x, pacman_y = path

    orientation_angle = 0
    if pacman_x < current_x:
        orientation_angle = 180
    elif pacman_x > current_x:
        orientation_angle = 0
    elif pacman_y < current_y:
        orientation_angle = -90
    elif pacman_y > current_y:
        orientation_angle = 90


    if grid[pacman_x][pacman_y] == o:
        grid[pacman_x][pacman_y] = e
        score += 10

    # Draw the grid
    pellet = False
    for row in range(GRID_WIDTH):
        for col in range(GRID_HEIGHT):
            grid_unit = grid[row][col]
            cell_x = row * GRID_SIZE
            cell_y = col * GRID_SIZE
            if grid_unit == I:
                pygame.draw.rect(screen, BLUE, (cell_x, cell_y, 19, 19))
            elif grid_unit == e:
                pygame.draw.rect(screen, BLACK, (cell_x, cell_y, GRID_SIZE, GRID_SIZE))
            elif grid_unit == o:
                pellet = True
                pygame.draw.circle(screen, WHITE, (cell_x + GRID_SIZE // 2, cell_y + GRID_SIZE // 2), 5)
            elif grid_unit == O:
                pygame.draw.circle(screen, WHITE, (cell_x + GRID_SIZE // 2, cell_y + GRID_SIZE // 2), 10)
            elif grid_unit == c:
                pygame.draw.circle(screen, RED, (cell_x + GRID_SIZE // 2, cell_y + GRID_SIZE // 2), 5)

    if not pellet:
        running = False

   # Draw Pac-Man body
    pacman_radius = GRID_SIZE // 2 - 2
    pacman_rect = pygame.Rect(pacman_x * GRID_SIZE, pacman_y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    pygame.draw.circle(screen, YELLOW, (pacman_x * GRID_SIZE + GRID_SIZE // 2, pacman_y * GRID_SIZE + GRID_SIZE // 2), pacman_radius)

    # Draw Pac-Man mouth
    mouth_points = [(pacman_x * GRID_SIZE + GRID_SIZE // 2, pacman_y * GRID_SIZE + GRID_SIZE // 2)]
    mouth_angle = 40
    for angle in range(orientation_angle - mouth_angle, orientation_angle + mouth_angle + 1):
        x = pacman_x * GRID_SIZE + GRID_SIZE // 2 + pacman_radius * math.cos(math.radians(angle))
        y = pacman_y * GRID_SIZE + GRID_SIZE // 2 + pacman_radius * math.sin(math.radians(angle))
        mouth_points.append((x, y))

    mouth_points.append((pacman_x * GRID_SIZE + GRID_SIZE // 2, pacman_y * GRID_SIZE + GRID_SIZE // 2))
    pygame.draw.polygon(screen, BLACK, mouth_points)

    font = pygame.font.Font(None, 24)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (4, 16*17))

    time_text = font.render("Time:", True, WHITE)
    screen.blit(time_text, (4, 340))
    time_text = font.render(f"{elapsed_time} seconds", True, WHITE)
    screen.blit(time_text, (4, 370))

    pygame.display.flip()

    clock.tick(2)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
