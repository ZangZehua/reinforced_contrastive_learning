import os
import numpy as np
import torch
import torch.optim as optim

from models.actor_critic import ActorCriticMLP


class Buffer:
    def __init__(self):
        self.state = torch.Tensor([]).cuda()
        self.action = torch.Tensor([]).cuda()
        self.action_prob = torch.Tensor([]).cuda()
        self.reward = torch.Tensor([]).cuda()
        self.done = torch.Tensor([]).cuda()

    def clear_buffer(self):
        self.state = torch.Tensor([]).cuda()
        self.action = torch.Tensor([]).cuda()
        self.action_prob = torch.Tensor([]).cuda()
        self.reward = torch.Tensor([]).cuda()
        self.done = torch.Tensor([]).cuda()


class PPO:
    def __init__(self, args, obs_dim, action_dim, action_std_init=None, continuous=False, cnn=False):
        self.args = args
        self.continuous = continuous
        if continuous:
            self.action_std = action_std_init

        self.model = ActorCriticMLP(obs_dim, action_dim, self.args.rl_hidden_dim, action_std_init, continuous)
        self.old_model = ActorCriticMLP(obs_dim, action_dim, self.args.rl_hidden_dim, action_std_init, continuous)
        if self.args.use_cuda:
            self.model.cuda()
            self.old_model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.rl_lr, weight_decay=args.rl_weight_decay)
        self.old_model.load_state_dict(self.model.state_dict())
        self.buffer = Buffer()

    def set_action_std(self, new_action_std):
        if self.continuous:
            self.action_std = new_action_std
            self.model.set_action_std(new_action_std)
            self.old_model.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.continuous:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, train_mode=True):
        with torch.no_grad():
            model_action, action_log_prob = self.old_model.act(state)
        if train_mode:
            self.buffer.state = torch.cat((self.buffer.state, state), dim=0)
            self.buffer.action = torch.cat((self.buffer.action, model_action), dim=0)
            self.buffer.action_prob = torch.cat((self.buffer.action_prob, action_log_prob), dim=0)
        if self.continuous:
            action = torch.clamp(model_action, 0, 1).detach().cpu()
            a0 = torch.round(action[:, 0] * self.args.projection_range).unsqueeze(0)
            a1 = torch.round(action[:, 1] * self.args.projection_range).unsqueeze(0)
            a2 = torch.round(224 - action[:, 2] * self.args.projection_range - action[:, 0] * self.args.projection_range).unsqueeze(0)
            a3 = torch.round(224 - action[:, 3] * self.args.projection_range - action[:, 1] * self.args.projection_range).unsqueeze(0)
            action = torch.cat((a0, a1, a2, a3), dim=0).transpose(0, 1).squeeze(0).int()
        else:
            action = model_action
        return action

    def reward_scaling(self, lambd=1):
        self.buffer.reward_d = (self.buffer.reward_d - self.buffer.reward_d.mean()) / \
                               (self.buffer.reward_d.std() + 1e-7)
        self.buffer.reward_mi = (self.buffer.reward_mi - self.buffer.reward_mi.min()) / \
                                (self.buffer.reward_mi.std() + 1e-7)
        self.buffer.reward = self.buffer.reward_d + lambd * self.buffer.reward_mi

    def learn(self):
        loss_function = torch.nn.MSELoss()
        # Monte Carlo estimate of returns
        self.buffer.reward = self.buffer.reward.squeeze(dim=-1)
        self.buffer.done = self.buffer.done.squeeze(dim=-1)
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.reward), reversed(self.buffer.done)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.args.rl_gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if self.args.use_cuda:
            rewards = rewards.cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = self.buffer.state.detach()
        old_actions = self.buffer.action.detach()
        old_logprobs = self.buffer.action_prob.detach()
        if self.args.use_cuda:
            old_states = old_states.cuda()
            old_actions = old_actions.cuda()
            old_logprobs = old_logprobs.cuda()

        # Optimize policy for K epochs
        for _ in range(self.args.rl_k_epoch):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.model.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.rl_clip_param, 1 + self.args.rl_clip_param) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * loss_function(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.old_model.load_state_dict(self.model.state_dict())

        # clear buffer
        self.buffer.clear_buffer()

    def save_model(self, epoch, save_path_base, agent_type):
        save_path = os.path.join(save_path_base, "epoch_" + str(epoch) + "_" + agent_type + ".pth")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), save_path)
