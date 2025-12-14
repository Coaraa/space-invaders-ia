import numpy as np

class SpaceInvadersEnv:
    def __init__(self, game):
        self.game = game
        self.action_size = 4
        self.state_size = 6
        self.previous_score = 0

    def reset(self):
        self.game.__init__()   # reset simple du jeu
        self.previous_score = 0
        return self._get_state()

    def step(self, action):
        done = self.game.step(action)
        next_state = self._get_state()
        reward = self._compute_reward(done)
        return next_state, reward, done

    def _get_state(self):
        player_x = self.game.player.sprite.rect.centerx

        if self.game.aliens:
            alien = self.game.aliens.sprites()[0]
            alien_x = alien.rect.centerx
            alien_y = alien.rect.centery
        else:
            alien_x = 0
            alien_y = 0

        bullet_active = 1 if self.game.player.sprite.lasers else 0

        if self.game.alien_lasers:
            laser_y = self.game.alien_lasers.sprites()[0].rect.y
        else:
            laser_y = 0

        score = self.game.score

        return np.array([
            player_x / 600,
            alien_x / 600,
            alien_y / 600,
            bullet_active,
            laser_y / 600,
            score / 1000
        ], dtype=np.float32)

    def _compute_reward(self, done):
        reward = 0

        if self.game.score > self.previous_score:
            reward += 5

        if self.game.lives < 3:
            reward -= 2

        if done and self.game.lives <= 0:
            reward -= 20

        if done and not self.game.aliens:
            reward += 50

        self.previous_score = self.game.score
        return reward
