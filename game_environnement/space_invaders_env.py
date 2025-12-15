import numpy as np
import pygame


class SpaceInvadersEnv:
    """
    Environnement RL pour Space Invaders (version symbolique).
    Actions :
        0 = ne rien faire
        1 = gauche
        2 = droite
        3 = tirer
    """

    def __init__(self, game, max_steps=2000):
        self.game = game
        self.action_size = 4
        self.state_size = 6 
        self.max_steps = max_steps

        # Variables RL
        self.steps = 0
        self.previous_score = 0
        self.previous_lives = game.lives
        self.previous_blocks = len(game.blocks)

    # ------------------------
    # RESET
    # ------------------------
    def reset(self):
        self.game.reset()

        self.steps = 0
        self.previous_score = self.game.score
        self.previous_lives = self.game.lives
        self.previous_blocks = len(self.game.blocks)

        return self._get_state()

    # ------------------------
    # STEP
    # ------------------------
    def step(self, action):
        self.steps += 1

        # Infos AVANT action (important pour la reward)
        state_before = self._get_state()
        clear_shot_available = state_before[4]
        player_was_ready = self.game.player.sprite.ready

        # Appliquer action
        done = self.game.step(action)

        # Reward
        reward = self._compute_reward(
            action=action,
            was_clear_shot=clear_shot_available,
            player_was_ready=player_was_ready
        )

        # Fin forcée
        if self.steps >= self.max_steps:
            done = True

        next_state = self._get_state()
        return next_state, reward, done

    # ------------------------
    # STATE REPRESENTATION
    # ------------------------
    def _get_state(self):
        """
        Etat symbolique :
        [pos_x_norm,
         nearest_alien_dx_norm,
         nearest_alien_dy_norm,
         danger_above,
         clear_shot_available,
         lasers_on_screen]
        """

        player = self.game.player.sprite
        aliens = self.game.aliens.sprites()

        # Position joueur normalisée
        pos_x = player.rect.centerx / self.game.screen_width

        # Alien le plus proche horizontalement
        if aliens:
            nearest = min(
                aliens,
                key=lambda a: abs(a.rect.centerx - player.rect.centerx)
            )
            dx = (nearest.rect.centerx - player.rect.centerx) / self.game.screen_width
            dy = (nearest.rect.centery - player.rect.centery) / self.game.screen_height
        else:
            dx, dy = 0.0, 0.0

        # Danger : laser alien au-dessus
        danger = 0.0
        for laser in self.game.alien_lasers:
            if abs(laser.rect.centerx - player.rect.centerx) < 20 and laser.rect.centery < player.rect.centery:
                danger = 1.0
                break

        # Tir clair : alien aligné verticalement
        clear_shot = 0.0
        for alien in aliens:
            if abs(alien.rect.centerx - player.rect.centerx) < 15:
                clear_shot = 1.0
                break

        lasers_on_screen = len(player.lasers) / 5.0


        return np.array([
            pos_x,
            dx,
            dy,
            danger,
            clear_shot,
            lasers_on_screen
        ], dtype=np.float32)

    # ------------------------
    # REWARD FUNCTION
    # ------------------------
    def _compute_reward(self, action, was_clear_shot, player_was_ready):
        reward = 0.0

        current_score = self.game.score
        current_lives = self.game.lives
        current_blocks = len(self.game.blocks)

        # Kill alien → signal principal
        if current_score > self.previous_score:
            reward += 13.0

        # Perte de vie → grosse pénalité
        if current_lives < self.previous_lives:
            reward -= 10.0

        # Tir intelligent
        if action == 3 and player_was_ready:
            if was_clear_shot == 1.0:
                reward += 0.5
            elif current_blocks < self.previous_blocks:
                reward -= 0.2
            else:
                reward -= 0.01

        # Bon placement sans tirer
        if was_clear_shot == 1.0 and action != 3:
            reward += 0.02

        # Victoire
        if not self.game.aliens:
            reward += 50.0

        # Pénalité de temps
        reward -= 0.01
        
        # Pénalité si les aliens descendent trop bas
        aliens = self.game.aliens.sprites()
        if aliens:
            lowest_alien_y = max(alien.rect.bottom for alien in aliens)
            height_ratio = lowest_alien_y / self.game.screen_height

            if height_ratio > 0.7:
                reward -= 0.05
            if height_ratio > 0.85:
                reward -= 0.2
            
                
        if action in [1, 2]:  # gauche ou droite
            reward += 0.002



        # Mise à jour mémoire
        self.previous_score = current_score
        self.previous_lives = current_lives
        self.previous_blocks = current_blocks

        return reward
