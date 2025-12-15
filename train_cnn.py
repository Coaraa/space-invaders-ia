import torch
from cnn.cnn import CoordinateCNN, train_supervised
from data.generate_data import generate_expert_dataset
from game_environnement.space_invaders_game.Code.Main import Game
from game_environnement.space_invaders_env import SpaceInvadersEnv
import pygame


pygame.init()

pygame.display.set_mode((100, 100))  

game = Game(600, 600, screen=None)
env = SpaceInvadersEnv(game)



X_train, y_train = generate_expert_dataset(env, n_samples=10000)

supervised_model = CoordinateCNN(state_size=6, action_size=4)
supervised_model, history = train_supervised(supervised_model, X_train, y_train)

# Sauvegarde
torch.save(supervised_model.state_dict(), "supervised_cnn.pth")