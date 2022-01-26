import copy
import collections
from typing import Sequence

import numpy as np
import torch
from torch._C import _last_executed_optimized_graph
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def _mean_or_nan(xs: Sequence[float]) -> float:
	"""Return its mean a non-empty sequence, numpy.nan for a empty one."""
	return np.mean(xs) if xs else np.nan


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class Discriminator(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Discriminator, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, 1)

	def forward(self, state, action):
		return torch.sigmoid(self.get_logits(state, action))

	def get_logits(self, state, action):
		sa = torch.cat([state, action], 1)

		validity = torch.tanh(self.l1(sa))
		validity = torch.tanh(self.l2(validity))
		validity = self.l3(validity)

		return validity


class TD3_GAN(object):
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
			alpha=2.5,
			beta=0.5
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discriminator = Discriminator(state_dim, action_dim).to(device)
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4)
		# self.adversarial_loss = torch.nn.BCELoss()
		self.adversarial_loss = torch.nn.functional.binary_cross_entropy_with_logits

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha
		self.beta = beta

		self.total_it = 0

		# For logging
		log_len = 2500
		self.q_record: collections.deque = collections.deque(maxlen=log_len)
		self.discrimnator_loss_record: collections.deque = collections.deque(maxlen=log_len)
		self.actor_loss_record: collections.deque = collections.deque(maxlen=log_len)
		self.imitation_loss_record: collections.deque = collections.deque(maxlen=log_len)
		self.validity_record: collections.deque = collections.deque(maxlen=log_len)
		self.beta_record: collections.deque = collections.deque(maxlen=log_len)

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer
		state, demo_action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
					torch.randn_like(demo_action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
					self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

			# Select action for Discriminator training
			policy_action = (
				self.actor(state)
			).clamp(-self.max_action, self.max_action)

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, demo_action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# GAN Phase
			valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
			fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

			real_loss = self.adversarial_loss(
				self.discriminator.get_logits(state, demo_action), valid
			)
			fake_loss = self.adversarial_loss(
				self.discriminator.get_logits(state, policy_action), fake
			)
			discriminator_loss = real_loss + fake_loss

			self.discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			self.discriminator_optimizer.step()

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha / Q.abs().mean().detach()

			with torch.no_grad():
				validity = self.discriminator(state, pi)
			imitation_loss = -torch.log(validity.mean())
			q_loss = -lmbda * Q.mean()
			# actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, demo_action)

			# Calculating Uncertainty
			unc = torch.max(target_Q1, target_Q2) - torch.min(target_Q1, target_Q2)
			beta = torch.min(Tensor([1.0]), 1 / unc.max())
			actor_loss = -lmbda * Q.mean() + beta * (-torch.log(validity.mean())) \
			             + (1.0 - beta) * F.mse_loss(pi, demo_action)

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.q_record.extend(Q.mean().detach().cpu().numpy().ravel())
			self.discrimnator_loss_record.extend(discriminator_loss.detach().cpu().numpy().ravel())
			self.actor_loss_record.extend(actor_loss.detach().cpu().numpy().ravel())
			self.imitation_loss_record.extend(imitation_loss.detach().cpu().numpy().ravel())
			self.validity_record.extend(validity.mean().detach().cpu().numpy().ravel())
			self.beta_record.extend(beta.mean().detach().cpu().numpy().ravel())

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

	def get_log(self):

		return [
			("average_q", _mean_or_nan(self.q_record)),
			("average_disc_loss", _mean_or_nan(self.discrimnator_loss_record)),
			("average_actor_loss", _mean_or_nan(self.actor_loss_record)),
			("average_imi_loss", _mean_or_nan(self.imitation_loss_record)),
			("average_valid", _mean_or_nan(self.validity_record)),
			("average_beta", _mean_or_nan(self.beta_record))
		]
