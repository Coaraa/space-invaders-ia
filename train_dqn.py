from env.space_invaders_game.Code.Main import Game
from env.space_invaders_env import SpaceInvadersEnv
from dqn.dqn_agent import train_dqn

import pygame
pygame.init()

pygame.display.set_mode((1, 1))  
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

game = Game(SCREEN_WIDTH, SCREEN_HEIGHT)
env = SpaceInvadersEnv(game)
rewards = train_dqn(env, episodes=500)



