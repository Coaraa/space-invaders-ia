import numpy as np

class SpaceInvadersEnv:
    def __init__(self, game):
        self.game = game
        self.action_size = 4
        self.state_size = 6
        self.previous_score = 0
        self.max_steps = 2000

    def reset(self):
        self.game.reset()
        self.steps = 0
        self.previous_score = 0
        self.previous_lives = self.game.lives
        return self._get_state()

    def step(self, action):
        self.steps += 1

        done = self.game.step(action)

        reward = self._compute_reward()

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done



    def _get_state(self):
        player_x = self.game.player.sprite.rect.centerx
        player_y = self.game.player.sprite.rect.centery # Utile pour la distance

        # 1. Trouver l'alien le plus proche (le danger immédiat)
        closest_alien_x = 0
        closest_alien_y = 0
        min_dist = float('inf')

        if self.game.aliens:
            for alien in self.game.aliens.sprites():
                # Distance euclidienne simple ou distance Y
                dist = abs(alien.rect.centerx - player_x) + abs(alien.rect.centery - player_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_alien_x = alien.rect.centerx
                    closest_alien_y = alien.rect.centery
        
        # 2. Trouver le laser ennemi le plus proche (pour esquiver)
        closest_laser_x = 0
        closest_laser_y = 0
        min_laser_dist = float('inf')

        if self.game.alien_lasers:
            for laser in self.game.alien_lasers.sprites():
                dist = abs(laser.rect.centerx - player_x) + abs(laser.rect.centery - player_y)
                if dist < min_laser_dist:
                    min_laser_dist = dist
                    closest_laser_x = laser.rect.centerx
                    closest_laser_y = laser.rect.centery

        # Normalisation (très important pour les réseaux de neurones)
        return np.array([
            player_x / 600.0,
            (closest_alien_x - player_x) / 600.0, # Position relative X de l'alien
            closest_alien_y / 600.0,              # Position Y de l'alien
            (closest_laser_x - player_x) / 600.0, # Position relative X du laser
            closest_laser_y / 600.0,              # Position Y du laser
            1.0 if self.game.player.sprite.lasers else 0.0 # Balle prête ou non
        ], dtype=np.float32)

    def _compute_reward(self):
        reward = 0.0

        # + points si le score augmente (alien touché)
        if self.game.score > self.previous_score:
            reward += (self.game.score - self.previous_score) * 0.01

        # pénalité si on perd une vie
        if self.game.lives < self.previous_lives:
            reward -= 10.0

        # petite récompense de survie
        reward += 0.01

        self.previous_score = self.game.score
        self.previous_lives = self.game.lives

        return reward

