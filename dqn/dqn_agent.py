import random

import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import ReplayBuffer
from .network import QNetwork

# ==============================
# Agent DQN
# ==============================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# ==============================
# Boucle d'entraînement
# ==============================

def train_dqn(env, episodes=500):
    agent = DQNAgent(env.state_size, env.action_size)
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward
            
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        agent.update_target_network()
        rewards_history.append(total_reward)

        print(f"Episode {episode+1}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    print("Entraînement terminé. Sauvegarde du modèle...")
    torch.save(agent.q_network.state_dict(), "dqn_space_invaders.pth")
    print("Modèle sauvegardé sous 'dqn_space_invaders.pth'")

    return rewards_history
