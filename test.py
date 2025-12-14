import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from env.space_invaders_game.Code.Main import Game

game = Game(600, 600)
print("Game initialized successfully")
