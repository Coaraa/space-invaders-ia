import dqn
from game_environnement.space_invaders_game.Code.Main import Game
from game_environnement.space_invaders_env import SpaceInvadersEnv
from dqn.dqn_agent import train_dqn
from dqn.dqn_agent import DQNAgent

import pygame
pygame.init()

pygame.display.set_mode((100, 100))  

game = Game(600, 600, screen=None)
env = SpaceInvadersEnv(game)


print("=== DÉBUT DE L'ENTRAINEMENT DQN ===")
rewards = train_dqn(env, episodes=500)
print("=== FIN DE L'ENTRAINEMENT DQN ===")




env.render_mode = True
dqn.agent.epsilon = 0.0  # politique pure

state = env.reset()

done = False
while not done:
    action = dqn.agent.act(state)
    state, reward, done = env.step(action)
    pygame.time.delay(30)
