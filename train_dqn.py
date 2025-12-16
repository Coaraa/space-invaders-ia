import dqn
from game_environnement.space_invaders_game.Code.Main import Game
from game_environnement.space_invaders_env import SpaceInvadersEnv
from dqn.dqn_agent import train_dqn
from dqn.dqn_agent import DQNAgent

import pygame
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

pygame.init()

pygame.display.set_mode((100, 100))

game = Game(600, 600, screen=None)
env = SpaceInvadersEnv(game)

# print("=== DÉBUT DE L'ENTRAINEMENT DQN ===")
# rewards = train_dqn(env, episodes=500)
# print("=== FIN DE L'ENTRAINEMENT DQN ===")



if "SDL_VIDEODRIVER" in os.environ:
    del os.environ["SDL_VIDEODRIVER"]
    print("Mode 'dummy' désactivé : La fenêtre devrait apparaître.")

try:
    pygame.quit()
except Exception:
    pass

# --- 1. CONFIGURATION ---
pygame.init()
screen_width = 600
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Test Agent DQN Space Invaders")
clock = pygame.time.Clock()

# --- 2. ENVIRONNEMENT ---
game = Game(screen_width, screen_height, screen=screen)
env = SpaceInvadersEnv(game)
env.test()

# --- 3. DÉTECTION TAILLE & AGENT ---
initial_state = env.reset()
#real_state_size = initial_state.shape[0]
real_state_size = 6
action_size = 4
agent = DQNAgent(real_state_size, action_size)

# --- 4. CHARGEMENT MODÈLE ---
model_path = "models/dqn_space_invaders.pth"
device = torch.device("cpu") # CPU obligatoire pour éviter les conflits
agent.device = device
agent.q_network.to(device)

try:
    print("Dossier courant :", os.getcwd())
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_size' not in checkpoint:
        agent.q_network.load_state_dict(checkpoint)
    else:
        agent.q_network.load_state_dict(checkpoint)
    agent.q_network.eval()
    print("Cerveau de l'IA chargé !")
except FileNotFoundError:
    print(f"Fichier '{model_path}' introuvable.")
    pygame.quit()
    raise SystemExit

# --- 5. BOUCLE DE JEU ---
agent.epsilon = 0.0
state = env.reset()
done = False
total_reward = 0

running = True
while running:
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Si le jeu n'est pas fini, faire un pas
    if not done:
        try:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
        except RuntimeError:
            print("Erreur de dimension état/modèle.")
            running = False

        # Rendu du jeu
        screen.fill((30, 30, 30))
        if game.player.sprite.lasers: game.player.sprite.lasers.draw(screen)
        if game.player: game.player.draw(screen)
        if game.blocks: game.blocks.draw(screen)
        if game.aliens: game.aliens.draw(screen)
        if game.alien_lasers: game.alien_lasers.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    # Si le jeu est fini, juste maintenir la fenêtre responsive
    else:
        screen.fill((30, 30, 30))
        pygame.display.flip()
        clock.tick(60)

# Après avoir fermé la fenêtre
pygame.display.quit()
pygame.quit()
print(f"Score final : {total_reward}")