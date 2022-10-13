"""
Train CMC with AlexNet
"""
from __future__ import print_function

import os
import sys
import csv
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import datetime

import tensorboard_logger as tb_logger

from torchvision import transforms
from dataset import RGB2Lab, RGB2YCbCr, ImageFolderInstance
from util import adjust_learning_rate, AverageMeter
from util import search_main_policy, search_hf_policy, search_rc_policy

from models.alexnet import MyAlexNetCMC
from models.ppo_agent import PPO
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss


def parse_args():
    parser = argparse.ArgumentParser('argument for pre-training')
    pretrain_time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d%H%M%S"))

    # base
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')

    # argsimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--random_epoch', type=int, default=None, help='no help')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='stl10', choices=['imagenet100', 'imagenet', "tiny",
                                                                         "stl10", "cifar10", "cifar100"])

    # specify folder
    parser.add_argument('--data_folder', type=str, default="/home/hujie/zdata/data/", help='path to data')
    parser.add_argument('--model_path', type=str, default="models_pt", help='path to save model')
    parser.add_argument('--tb_path', type=str, default="runs_pt", help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr'])

    # mixed precision setting
    parser.add_argument('--args_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')
    parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')

    # ppo config
    parser.add_argument('--rl_lr', type=float, default=0.0003, help='ppo lr')
    parser.add_argument('--rl_weight_decay', type=float, default=0.001, help='ppo weight decay')
    parser.add_argument('--rl_hidden_dim', type=int, default=256, help='ppo hidden dim')
    parser.add_argument('--rl_k_epoch', type=int, default=10)
    parser.add_argument('--rl_batch_size', type=int, default=32)
    parser.add_argument('--rl_gamma', type=float, default=0.99)
    parser.add_argument('--rl_lamda', type=float, default=0.98)
    parser.add_argument('--rl_clip_param', type=float, default=0.2)
    parser.add_argument('--rl_c1', type=float, default=0.5)
    parser.add_argument('--rl_c2', type=float, default=0.01)
    parser.add_argument('--rl_update_interval', type=int, default=2000)
    parser.add_argument('--rl_update_episode', type=int, default=50)
    parser.add_argument('--rl_action_std', type=float, default=0.6)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--reward_type', type=str, default="infoNCE", choices=["cosine_dist", "infoNCE"])
    parser.add_argument('--max_step', type=int, default=5)
    parser.add_argument('--projection_range', type=int, default=50)
    parser.add_argument('--dont_search', action='store_true')
    parser.add_argument('--max_range', type=float, default=1.5)
    parser.add_argument('--epsilon', type=float, default=0.1)

    args = parser.parse_args()

    args.data_base_path = args.data_folder
    if args.dataset == "stl10":
        save_path_base = "saved/STL-10_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "STL-10")
    elif args.dataset == "cifar10":
        save_path_base = "saved/CIFAR-10_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "CIFAR-10")
    elif args.dataset == "cifar100":
        save_path_base = "saved/CIFAR-100_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "CIFAR-100")
    elif args.dataset == "tiny":
        save_path_base = "saved/Tiny_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "tiny-imagenet-200")
    elif args.dataset == "imagenet":
        save_path_base = "saved/IMAGENET_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "imagenet")
    else:
        raise FileNotFoundError

    args.model_path = os.path.join(save_path_base, args.model_path)
    args.tb_path = os.path.join(save_path_base, args.tb_path)

    if args.random_epoch is not None:
        args.resume = os.path.join(args.data_base_path, os.path.join("cmc_models_" + args.dataset, "ckpt_epoch_" + str(args.random_epoch) + ".pth" ))

    # create save folder
    if not os.path.isdir(save_path_base):
        os.makedirs(save_path_base)
        args.save_path_base = save_path_base

    if args.dataset == 'imagenet':
        if 'alexnet' not in args.model:
            args.crop_low = 0.08

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)

    if not os.path.isdir(args.tb_path):
        os.makedirs(args.tb_path)

    if not os.path.isdir(args.data_folder):
        raise ValueError('data path not exist: {}'.format(args.data_folder))

    return args


def get_train_loader(args):
    """get the train loader"""
    data_folder = os.path.join(args.data_folder, 'unlabeled')

    if args.view == 'Lab':
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2Lab()
    elif args.view == 'YCbCr':
        mean = [116.151, 121.080, 132.342]
        std = [109.500, 111.855, 111.964]
        color_transfer = RGB2YCbCr()
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        # transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def search_policy(args, train_loader, model, contrast, criterion_l, criterion_ab, policy_agent, rc_agent, hf_agent):
    model.eval()
    contrast.eval()
    policy_agent.model.train()
    rc_agent.model.train()
    hf_agent.model.train()
    rc_reward = search_rc_policy(args, train_loader, model, contrast, criterion_l, criterion_ab, rc_agent)
    hf_reward = search_hf_policy(args, train_loader, model, contrast, criterion_l, criterion_ab, hf_agent)
    main_reward = search_main_policy(args, train_loader, model, contrast, criterion_l, criterion_ab, policy_agent,
                                     rc_agent, hf_agent)
    policy_agent.model.eval()
    rc_agent.model.eval()
    hf_agent.model.eval()
    model.train()
    contrast.train()
    print("===>Policy Found!")
    return rc_reward, hf_reward, main_reward


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = MyAlexNetCMC(args.feat_dim)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    policy_agent = PPO(args, args.feat_dim * 2, 2)
    rc_agent = PPO(args, args.feat_dim * 2, 4, args.rl_action_std, continuous=True)
    hf_agent = PPO(args, args.feat_dim * 2, 2)

    return model, contrast, criterion_ab, criterion_l, policy_agent, rc_agent, hf_agent


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, args):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            index = index.cuda()
            inputs = inputs.cuda()

        # ===================forward=====================
        feat_l, feat_ab = model(inputs)
        out_l, out_ab = contrast(feat_l, feat_ab, index)

        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()

        loss = l_loss + ab_loss
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                  'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lprobs=l_prob_meter,
                abprobs=ab_prob_meter))
            sys.stdout.flush()

    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg


def main():
    # parse the args
    args = parse_args()
    config = None

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab, criterion_l, policy_agent, rc_agent, hf_agent = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_path, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        if not args.dont_search:
            rc_reward, hf_reward, main_reward = search_policy(args, train_loader, model, contrast, criterion_l, criterion_ab,
                                                              policy_agent, rc_agent, hf_agent)
        time2 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                 optimizer, args)
        time3 = time.time()
        print('epoch {}, total time {:.2f}, {:.2f}'.format(epoch, time2 - time1, time3 - time2))

        # tensorboard logger
        logger.log_value('l_loss', l_loss, epoch)
        logger.log_value('l_prob', l_prob, epoch)
        logger.log_value('ab_loss', ab_loss, epoch)
        logger.log_value('ab_prob', ab_prob, epoch)
        logger.log_value('main_reward', main_reward, epoch)
        logger.log_value('rc_reward', rc_reward, epoch)
        logger.log_value('hf_reward', hf_reward, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                # 'args': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
