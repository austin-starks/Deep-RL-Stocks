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
    def __init__(self, inchannel, state_length, immediate_state_dim, hidden_size, outchannel, activation=nn.ReLU):
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
        self.immediate_state_dim = immediate_state_dim
        self.output = nn.Linear(outchannel * state_length + immediate_state_dim, outchannel)
    
    def forward(self, X, X_immediate):
        out =  self.layers(X)
        shortcut = self.shortcut(X)
        out = self.relu(out + shortcut)
        shape = out.shape
        out = out.reshape((shape[0], shape[1] * shape[2]))
        out = torch.cat([out, X_immediate], 1) 
        out = self.output(out)   
        return out

class Actor(nn.Module):
    def __init__(self, ind_state_dim, ind_state_length, imm_state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv = Encoder(ind_state_dim, ind_state_length, imm_state_dim, 64, 64)
        self.l1 = nn.Linear(64 , 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action


    def forward(self, ind_state, imm_state_dim):
        ind_state = self.conv(ind_state, imm_state_dim)
        a = F.relu(self.l1(ind_state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, indicator_state_dim, state_length, immediate_state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.conv = Encoder(indicator_state_dim, state_length, immediate_state_dim, 64, 64)
        self.l1 = nn.Linear(64 + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

        # Q2 architecture
        self.conv2 = Encoder(indicator_state_dim, state_length, immediate_state_dim, 64, 64)
        self.l4 = nn.Linear(64 + action_dim, 64)
        self.l5 = nn.Linear(64, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, indicator_state_dim, immediate_state_dim, action):
        sa1 = self.conv(indicator_state_dim, immediate_state_dim)
        sa1 = torch.cat([sa1, action], 1)
        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        sa2 = self.conv2(indicator_state_dim, immediate_state_dim)
        sa2 = torch.cat([sa2, action], 1)
        q2 = F.relu(self.l4(sa2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, indicator_state_dim, immediate_state_dim, action):
        sa = self.conv(indicator_state_dim, immediate_state_dim)
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
        indicator_state_dim, immediate_state_dim = state_dim
        self.actor = Actor(indicator_state_dim[0], indicator_state_dim[1], immediate_state_dim[0], action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(indicator_state_dim[0], indicator_state_dim[1], immediate_state_dim[0], action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state_tup):
        ind_state, imm_state = state_tup
        if (len(ind_state.shape) == 2):
            ind_state = torch.FloatTensor([ind_state]).to(device)
            imm_state = torch.FloatTensor([imm_state]).to(device)
        else:
            ind_state = torch.FloatTensor(ind_state).to(device)
            imm_state = torch.FloatTensor(imm_state).to(device)
        action = self.actor(ind_state, imm_state).cpu().data.numpy()
        return action

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        ind_state, imm_state, action, next_ind_state, next_imm_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy 
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_ind_state, next_imm_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_ind_state, next_imm_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(next_ind_state, next_imm_state, action)

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
            actor_loss = -self.critic.Q1(next_ind_state, next_imm_state, 
                                self.actor(next_ind_state, next_imm_state)).mean()

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
        indicator_state, immediate_state = state_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        if type(indicator_state) == int:
            full_indicator_state = [max_size] + [indicator_state]
        else:
            full_indicator_state = [max_size] + [s for s in indicator_state]
        self.action = np.zeros((max_size, action_dim))
        self.indicator_state = np.zeros(full_indicator_state)
        self.immediate_state = np.zeros((max_size, immediate_state[0]))
        self.next_indicator_state = np.zeros(full_indicator_state)
        self.next_immediate_state = np.zeros((max_size, immediate_state[0]))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, 
            state, 
            action, 
            next_state,
            reward, 
            done):
        indicator_state, immediate_state = state 
        next_indicator_state, next_immediate_state = next_state 

        self.immediate_state[self.ptr] = immediate_state
        self.indicator_state[self.ptr] = indicator_state
        self.action[self.ptr] = action
        self.next_immediate_state[self.ptr] = next_immediate_state
        self.next_indicator_state[self.ptr] = next_indicator_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.indicator_state[ind]).to(self.device),
            torch.FloatTensor(self.immediate_state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_indicator_state[ind]).to(self.device),
            torch.FloatTensor(self.next_immediate_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )