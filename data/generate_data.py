import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def generate_expert_dataset(env, n_samples=5000):
    print("Génération du dataset expert...")
    inputs = []
    targets = []
    
    state = env.reset()
    while len(inputs) < n_samples:
        # --- LOGIQUE DE L'EXPERT (Le Professeur) ---
        # On triche un peu : on regarde directement les variables du jeu pour décider l'action parfaite
        player_x = env.game.player.sprite.rect.centerx
        aliens = env.game.aliens.sprites()
        
        action = 0 # Ne rien faire par défaut
        
        if aliens:
            # Trouver l'alien le plus proche
            nearest_alien = min(aliens, key=lambda a: abs(a.rect.centerx - player_x))
            
            # Si on est aligné (à 10 pixels près) -> TIRER (3)
            if abs(nearest_alien.rect.centerx - player_x) < 10:
                action = 3 
            # Si l'alien est à gauche -> GAUCHE (1)
            elif nearest_alien.rect.centerx < player_x:
                action = 1
            # Si l'alien est à droite -> DROITE (2)
            elif nearest_alien.rect.centerx > player_x:
                action = 2
        
        # --- ENREGISTREMENT ---
        # On enregistre l'état actuel (X) et l'action choisie (Y)
        inputs.append(state)
        targets.append(action)
        
        next_state, _, done = env.step(action)
        state = next_state
        
        if done:
            state = env.reset()
            
    print(f"Dataset généré : {len(inputs)} exemples.")
    
    # Conversion en Tenseurs PyTorch
    X = torch.tensor(np.array(inputs), dtype=torch.float32)
    y = torch.tensor(np.array(targets), dtype=torch.long)
    
    return X, y