from __future__ import print_function

import numpy as np
import datetime
import torch
from torchvision import transforms


def search_rc_policy(args, train_loader, model, contrast, criterion_l, criterion_ab, rc_agent):
    ppo_stime = datetime.datetime.now()
    rewards = []
    for idx, (inputs, _, indexes) in enumerate(train_loader):
        if idx >= args.train_episodes:  # train enough episodes
            break
        batch_size = args.batch_size
        inputs = inputs.float()
        if torch.cuda.is_available():
            indexes = indexes.cuda()
            inputs = inputs.cuda()

        episode_reward = torch.zeros((batch_size, 1)).cuda()

        # env.reset()
        with torch.no_grad():
            feat_l, feat_ab = model(inputs)
            states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
            out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
            base_loss = torch.Tensor([]).cuda()
            for i in range(batch_size):
                base_loss = torch.cat((base_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
            base_loss = base_loss.unsqueeze(1)
            # print("check base loss", base_loss)
        for step in range(args.max_step):
            actions = rc_agent.select_action(states)  # tensor [batch_size, action_dim]

            next_inputs = torch.Tensor([]).cuda()
            for i in range(batch_size):
                image = transforms.functional.resized_crop(inputs[i],
                                                           actions[i][0].item(), actions[i][1].item(),
                                                           actions[i][2].item(), actions[i][3].item(),
                                                           [224, 224])
                next_inputs = torch.cat((next_inputs, image.cuda().unsqueeze(0)), dim=0)

            with torch.no_grad():
                feat_l, feat_ab = model(next_inputs)
                next_states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
                out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
                next_loss = torch.Tensor([]).cuda()
                for i in range(batch_size):
                    next_loss = torch.cat((next_loss, criterion_l(out_l[i].unsqueeze(0))+criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
                next_loss = next_loss.unsqueeze(1)
            # print("check loss", next_loss)
            reward = 1/(abs(next_loss - base_loss+args.max_range) + args.epsilon)
            # print("check reward", reward)

            if step == args.max_step-1:
                done = torch.zeros((batch_size, 1)).cuda()
            else:
                done = torch.ones((batch_size, 1)).cuda()
            rc_agent.buffer.reward = torch.cat((rc_agent.buffer.reward, reward), dim=0)
            rc_agent.buffer.done = torch.cat((rc_agent.buffer.done, done), dim=0)
            episode_reward += reward

            states = next_states
            # last_loss = next_loss
        rc_agent.learn()
        rewards.append(episode_reward.cpu().mean(dim=0).numpy())
    ppo_etime = datetime.datetime.now()
    print("rc policy found! time:", ppo_etime-ppo_stime)
    return np.mean(rewards)


def search_hf_policy(args, train_loader, model, contrast, criterion_l, criterion_ab, hf_agent):
    ppo_stime = datetime.datetime.now()
    rewards = []
    for idx, (inputs, _, indexes) in enumerate(train_loader):
        if idx >= args.train_episodes:  # train enough episodes
            break
        batch_size = args.batch_size
        inputs = inputs.float()
        if torch.cuda.is_available():
            indexes = indexes.cuda()
            inputs = inputs.cuda()

        episode_reward = torch.zeros((batch_size, 1)).cuda()

        # env.reset()
        with torch.no_grad():
            feat_l, feat_ab = model(inputs)
            states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
            out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
            base_loss = torch.Tensor([]).cuda()
            for i in range(batch_size):
                base_loss = torch.cat((base_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
            base_loss = base_loss.unsqueeze(1)
        for step in range(args.max_step):
            actions = hf_agent.select_action(states)  # tensor [batch_size, action_dim]

            next_inputs = torch.Tensor([]).cuda()
            for i in range(batch_size):
                image = transforms.transforms.RandomHorizontalFlip(actions[i].item())(inputs[i])
                next_inputs = torch.cat((next_inputs, image.cuda().unsqueeze(0)), dim=0)

            with torch.no_grad():
                feat_l, feat_ab = model(next_inputs)
                next_states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
                out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
                next_loss = torch.Tensor([]).cuda()
                for i in range(batch_size):
                    next_loss = torch.cat((next_loss, criterion_l(out_l[i].unsqueeze(0))+criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
                next_loss = next_loss.unsqueeze(1)
            # print(next_loss)
            reward = 1/(abs(next_loss - base_loss+args.max_range) + args.epsilon)
            # print("check reward", reward)

            if step == args.max_step-1:
                done = torch.zeros((batch_size, 1)).cuda()
            else:
                done = torch.ones((batch_size, 1)).cuda()
            hf_agent.buffer.reward = torch.cat((hf_agent.buffer.reward, reward), dim=0)
            hf_agent.buffer.done = torch.cat((hf_agent.buffer.done, done), dim=0)
            episode_reward += reward

            states = next_states
            # last_loss = next_loss
        hf_agent.learn()
        rewards.append(episode_reward.cpu().mean(dim=0).numpy())
    ppo_etime = datetime.datetime.now()
    print("hf policy found! time:", ppo_etime - ppo_stime)
    return 0


def search_main_policy(args, train_loader, model, contrast, criterion_l, criterion_ab, policy_agent, rc_agent, hf_agent):
    ppo_stime = datetime.datetime.now()
    rewards = []
    for idx, (inputs, _, indexes) in enumerate(train_loader):
        if idx >= args.train_episodes:  # train enough episodes
            break
        batch_size = args.batch_size
        inputs = inputs.float()
        if torch.cuda.is_available():
            indexes = indexes.cuda()
            inputs = inputs.cuda()

        episode_reward = torch.zeros((batch_size, 1)).cuda()

        # env.reset()
        with torch.no_grad():
            feat_l, feat_ab = model(inputs)
            states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
            out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
            base_loss = torch.Tensor([]).cuda()
            for i in range(batch_size):
                base_loss = torch.cat(
                    (base_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
            base_loss = base_loss.unsqueeze(1)
            # print("check base loss", base_loss)
        for step in range(args.max_step):
            actions = policy_agent.select_action(states)  # tensor [batch_size, action_dim]

            next_inputs = torch.Tensor([]).cuda()
            for i in range(batch_size):
                if actions[i].item() == 0:
                    sub_policy = rc_agent.select_action(states[i].unsqueeze(0))  # tensor [batch_size, action_dim]
                    image = transforms.functional.resized_crop(inputs[i],
                                                               sub_policy[0].item(), sub_policy[1].item(),
                                                               sub_policy[2].item(), sub_policy[3].item(),
                                                               [224, 224])
                elif actions[i].item == 1:
                    sub_policy = hf_agent.select_action(states[i].unsqueeze(0))  # tensor [batch_size, n_actions]
                    image = transforms.RandomHorizontalFlip(sub_policy.item())(inputs[i])
                else:
                    image = None

                next_inputs = torch.cat((next_inputs, image.cuda().unsqueeze(0)), dim=0)

            with torch.no_grad():
                feat_l, feat_ab = model(next_inputs)
                next_states = torch.cat((feat_l, feat_ab), dim=1)  # tensor [batch_size, feature_dim]
                out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, indexes)
                next_loss = torch.Tensor([]).cuda()
                for i in range(batch_size):
                    next_loss = torch.cat(
                        (next_loss, criterion_l(out_l[i].unsqueeze(0)) + criterion_ab(out_ab[i].unsqueeze(0))), dim=0)
                next_loss = next_loss.unsqueeze(1)
            reward = 1 / (abs(next_loss - base_loss + args.max_range) + args.epsilon)
            # print("check reward", reward)

            if step == args.max_step - 1:
                done = torch.zeros((batch_size, 1)).cuda()
            else:
                done = torch.ones((batch_size, 1)).cuda()
            policy_agent.buffer.reward = torch.cat((policy_agent.buffer.reward, reward), dim=0)
            policy_agent.buffer.done = torch.cat((policy_agent.buffer.done, done), dim=0)
            episode_reward += reward

            states = next_states
            # last_loss = next_loss
        policy_agent.learn()
        rewards.append(episode_reward.cpu().mean(dim=0).numpy())
    ppo_etime = datetime.datetime.now()
    print("main policy found! time:", ppo_etime - ppo_stime)
    return 0


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    meter = AverageMeter()
