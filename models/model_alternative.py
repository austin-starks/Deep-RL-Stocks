import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utility.utils as utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Original Implementation found on https://github.com/sfujim/TD3/blob/master/TD3.py
class Encoder(nn.Module):
    def __init__(self, inchannel, hidden_size, outchannel, state_length, activation=nn.ReLU):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(inchannel, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            activation(),
            nn.Conv1d(hidden_size, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm1d(outchannel),
        )
        self.relu = activation()
        if inchannel == outchannel:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(inchannel, outchannel, kernel_size=1)
        # self.rnn = nn.GRU(state_length, hidden_size, batch_first=True, dropout=0.25, bidirectional=True, num_layers=2)
        self.output = nn.Linear(outchannel * state_length, outchannel)
    
    def forward(self, X):
        out =  self.layers(X)
        shortcut = self.shortcut(X)
        out = self.relu(out + shortcut)
        # out = self.rnn(out)[0]
        shape = out.shape
        out = self.output(out.reshape((shape[0], shape[1] * shape[2])))
        return out

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, state_length):
        super(Actor, self).__init__()
        self.conv = Encoder(state_dim, 64, 64, state_length)
        self.l1 = nn.Linear(64 , 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim * max_action)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, state):
        state = self.conv(state)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a).view(a.size(0), self.action_dim, self.max_action)
        return a.argmax(-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, state_length):
        super(Critic, self).__init__()

        # Q1 architecture
        self.conv = Encoder(state_dim, 64, 64, state_length)
        self.l1 = nn.Linear(64 + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

        # Q2 architecture
        self.conv2 = Encoder(state_dim, 64, 64, state_length)
        self.l4 = nn.Linear(64 + action_dim, 64)
        self.l5 = nn.Linear(64, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, state, action):
        sa1 = self.conv(state)
        sa1 = torch.cat([sa1, action], 1)
        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        sa2 = self.conv2(state)
        sa2 = torch.cat([sa2, action], 1)
        q2 = F.relu(self.l4(sa2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = self.conv(state)
        sa = torch.cat([sa, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):

        self.actor = Actor(state_dim[0], action_dim, max_action, state_dim[1]).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim[0], action_dim, state_dim[1]).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        if (len(state.shape) == 2):
            state = torch.FloatTensor([state]).to(device)
        else:
            state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy()
        return action

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        if type(state_dim) == int:
            full_state_dim = [max_size] + [state_dim]
        else:
            full_state_dim = [max_size] + [s for s in state_dim]
        self.action = np.zeros((max_size, action_dim))
        self.state = np.zeros(full_state_dim)
        self.next_state = np.zeros(full_state_dim)
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )