
import logging
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .agent import MADDPGAgent
from .buffer_replay import PrioritizedReplayBuffer, ReplayBuffer
from .network import MADDPGPolicy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
WEIGHT_DECAY = 2e-6   # L2 weight decay


LOGGER = logging.getLogger(__name__)


class MADDPGTrainer:
    def __init__(self, agent: MADDPGAgent, learning_rate_actor: float = 1e-3, learning_rate_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 1e-3, replaybuffer_type="base", filename='MADDPG.pth', filepath='./'):
        self.target_agent = agent
        self.local_agent = MADDPGAgent.clone_agent(self.target_agent)
        self.__brain_name = self.local_agent.env.brain_names[0]
        self.__brain = self.local_agent.env.brains[self.__brain_name]

        self.filename = filename
        self.filepath = filepath

        if replaybuffer_type.lower() == "base":
            self.memory = ReplayBuffer(agent.action_size, BUFFER_SIZE, BATCH_SIZE, seed=agent.policy.seed)
        elif replaybuffer_type.lower() == "priority":
            self.memory = PrioritizedReplayBuffer(agent.action_size, BUFFER_SIZE, BATCH_SIZE, seed=agent.policy.seed)
        else:
            raise ValueError("Unknown replaybuffer type! Known types are {base, priority}")

        self.actor_optimizer = optim.Adam(self.local_agent.policy.pi.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.local_agent.policy.V.parameters(), lr=learning_rate_critic,
                                           weight_decay=WEIGHT_DECAY)
        self.gamma = gamma
        self.tau = tau
        self.n_agents = agent.n_agents

        self.scores = []
        self.scores_deque = deque(maxlen=100)

    def train(self, n_episodes: int, max_steps_per_episode: int, target: float = 30.) -> MADDPGAgent:

        LOGGER.info('\t' + '-' * 30 + ' Training ' + '-' * 30)
        LOGGER.info(f'\tN Episodes {n_episodes} , max length episode: {max_steps_per_episode}')
        LOGGER.info(f'\tTarget score: {target}')
        LOGGER.info('\t' + '-' * 70)

        trained_episodes = len(self.scores)
        pbar = tqdm(total=n_episodes)

        for i_episode in range(trained_episodes + 1, n_episodes + trained_episodes + 1):
            # Reset the environment
            try:
                score = self.run_training_episode(max_steps_per_episode)
                self.scores_deque.append(np.max(score))
                self.scores.append(np.max(score))
                pbar.set_description(f"Episode {i_episode} Score: {score.mean():.2f}, 100 episode Mean Score: {np.mean(self.scores_deque, axis=0).mean():.2f}")
                if (np.mean(self.scores_deque, axis=0) > target).all() and len(self.scores) > 100:
                    episode100_mean = np.mean(self.scores_deque, axis=0)
                    pbar.clear()
                    pbar.close()
                    LOGGER.info(f"\tTraining Finished at episode {i_episode} - Mean Score: {episode100_mean.mean():.2f}")
                    break
            except KeyboardInterrupt:
                pbar.clear()
                pbar.close()
                LOGGER.info(f"\tTraining Interrupted at episode {len(self.scores)}")
                break
            finally:
                pbar.update(1)
                self.local_agent.policy.to_file(self.filepath + '/' + self.filename)
                # torch.save(self.local_agent.policy.state_dict(), self.filepath + self.filename)

        return self.target_agent

    def run_training_episode(self, max_steps_per_episode: int) -> np.ndarray:
        env_info = self.local_agent.env.reset(train_mode=True)[self.__brain_name] # reset the environment
        state = torch.from_numpy(env_info.vector_observations).float()
        done = False

        rewards = []
        for _ in range(max_steps_per_episode):
            action, next_state, reward, done = self.local_agent.act(state, True)
            self.update(state, action, reward, next_state, done)
            rewards.append(np.array(reward).squeeze())
            state = next_state
            if np.array(done).astype(bool).any():
                break
        return np.array(rewards).sum(axis=0)

    def update(self, state, action, reward, next_state, done) -> None:

        self.memory.add(state.reshape(1, -1), action.reshape(1, -1), reward, next_state.reshape(1, -1), done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.optimize(experiences, 0, 1)

            # experiences = self.memory.sample()
            self.optimize(experiences, 1, 0)

    def optimize(self, experiences: Tuple, own_idx: int, other_idx: int):

        if isinstance(self.memory, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, _ = experiences
        elif isinstance(self.memory, ReplayBuffer):
            states, actions, rewards, next_states, dones = experiences
        else:
            raise ValueError("Unknown Replay Buffer")
        self.local_agent.state_size

        own_states_cols = slice(own_idx * self.local_agent.state_size,
                                (own_idx + 1) * self.local_agent.state_size, 1)
        own_act_cols = slice(own_idx * self.local_agent.action_size,
                                (own_idx + 1) * self.local_agent.action_size, 1)

        other_states_cols = slice(other_idx * self.local_agent.state_size,
                                  (other_idx + 1) * self.local_agent.state_size, 1)
        other_act_cols = slice(other_idx * self.local_agent.action_size,
                                (other_idx + 1) * self.local_agent.action_size, 1)

        own_states = states[:, own_states_cols]
        own_actions = actions[:, own_act_cols]
        own_next_states = next_states[:, own_states_cols]

        other_states = states[:, other_states_cols]
        other_actions = actions[:, other_act_cols]
        other_next_states = next_states[:, other_states_cols]

        # # Concatenate both agent information (own agent first, other agent in second position)
        all_states = torch.cat((own_states, other_states), dim=1).to(device)
        all_actions = torch.cat((own_actions, other_actions), dim=1).to(device)
        all_next_states = torch.cat((own_next_states, other_next_states), dim=1).to(device)


        # Get predicted next-state actions and Q values from target models
        all_next_actions = (
            torch.cat([
                self.target_agent.policy.pi(own_states),
                self.target_agent.policy.pi(other_states)], dim=1)
                    .to(device))
        Q_targets_next = self.target_agent.policy.V(all_next_states, all_next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.local_agent.policy.V(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actions_pred = torch.cat([self.local_agent.policy.pi(own_states), self.local_agent.policy.pi(other_states).detach()],
                                 dim=1).to(device)
        actor_loss = -self.local_agent.policy.V(all_states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # print(f'actor loss: {actor_loss.item():.2f}', f'critic loss: {critic_loss.item():.2f}')
        # ----------------------- update target networks ----------------------- #
        self.target_agent.policy.soft_update(self.local_agent.policy, self.tau)

        self.actor_loss = actor_loss.data
        self.critic_loss = critic_loss.data
