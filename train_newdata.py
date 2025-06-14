import argparse
import logging
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.xianyan import generate_shifted_images_batch
from utils.plot import plot_loss, plot_loss_cice, plot_loss_consist, display_images
from utils.val_newdata import test_single_volume
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/tooth', help='Name of Experiment')
parser.add_argument('--data_name', type=str, default='tooth', help='Name of data')
parser.add_argument('--exp', type=str, default='DTS_OAP_v4_attention', help='experiment_name')
parser.add_argument('--model', type=str, default='DTS_OAP_v4_attention', help='model_name')
parser.add_argument('--max_iterations', type=int, default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
# masks and unlabel
parser.add_argument('--labeled_bs', type=int, default=3, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=14, help='labeled data')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')

args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "tooth":
        ref_dict = {"3": 18, "7": 35,
                    "14": 70, "35": 175, "70": 350}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    # 传回来三个输出
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,worker_init_fn=worker_init_fn)
    model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    consistency_criterion = losses.mse_loss
    dice_loss = losses.DiceLoss(n_classes=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    # loss1 = []
    # loss2 = []
    # loss3 = []
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['images'], sampled_batch['masks']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs = model(volume_batch)
            num_outputs = len(outputs)
            y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)
            loss_seg = 0
            loss_seg_dice = 0
            loss_xianyan = 0
            for idx in range(num_outputs):
                y = outputs[idx][:labeled_bs, ...]
                y_prob = F.softmax(y, dim=1)
                loss_seg += ce_loss(y, label_batch[:labeled_bs][:].long())
                loss_seg_dice += dice_loss(y_prob, label_batch[:labeled_bs].unsqueeze(
                    1))
                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all
                y_pseudo_label[idx] = sharpening(y_prob_all)


                y_xianyan = torch.argmax(y_prob, dim=1)
                label_xianyan_x, label_xianyan_y = generate_shifted_images_batch(label_batch, labeled_bs)
                # y_prob_xianyan_x, y_prob_xianyan_y = generate_shifted_images_batch(y_prob[:labeled_bs], labeled_bs)
                y_prob_xianyan_x, y_prob_xianyan_y = generate_shifted_images_batch(y_xianyan, labeled_bs)
                loss_xianyan += consistency_criterion(torch.tensor(y_prob_xianyan_x).float(),
                                                      torch.tensor(label_xianyan_x).float())+ \
                                consistency_criterion( torch.tensor(y_prob_xianyan_y).float(),
                                                         torch.tensor(label_xianyan_y).float())

            loss_consist = 0

            # for i in range(num_outputs):
            #     for j in range(num_outputs):
            #         if i != j:
            #             loss_consist += consistency_criterion(y_ori[i],
            #                                                   y_pseudo_label[j])  # 不同上采样方式得到的out（预测）与伪标签进行均方误差


            # loss_consist = consistency_criterion(y_ori[0], y_ori[1]) + consistency_criterion(y_ori[0], y_ori[2]) \
            #                 + consistency_criterion(y_ori[1], y_ori[0]) + consistency_criterion(y_ori[1], y_ori[3]) \
            #                 + consistency_criterion(y_ori[2], y_ori[0]) + consistency_criterion(y_ori[2], y_ori[3]) \
            #                 + consistency_criterion(y_ori[3], y_ori[2]) + consistency_criterion(y_ori[3], y_ori[1])


            # loss_consist = consistency_criterion(y_pseudo_label[0], y_pseudo_label[1]) + consistency_criterion(y_pseudo_label[0], y_pseudo_label[2]) \
            #                + consistency_criterion(y_pseudo_label[1], y_pseudo_label[0]) + consistency_criterion(y_pseudo_label[1], y_pseudo_label[3]) \
            #                + consistency_criterion(y_pseudo_label[2], y_pseudo_label[0]) + consistency_criterion(y_pseudo_label[2], y_pseudo_label[3]) \
            #                + consistency_criterion(y_pseudo_label[3], y_pseudo_label[2]) + consistency_criterion(y_pseudo_label[3], y_pseudo_label[1])

            #
            # loss_consist = consistency_criterion(y_ori[0], y_pseudo_label[1]) + consistency_criterion(y_ori[0], y_pseudo_label[2]) \
            #                 + consistency_criterion(y_ori[1], y_pseudo_label[0]) + consistency_criterion(y_ori[1], y_pseudo_label[3]) \
            #                 + consistency_criterion(y_ori[2], y_pseudo_label[0]) + consistency_criterion(y_ori[2], y_pseudo_label[3]) \
            #                 + consistency_criterion(y_ori[3], y_pseudo_label[2]) + consistency_criterion(y_ori[3], y_pseudo_label[1])
            # main and aux
            loss_consist = consistency_criterion(y_ori[0], y_pseudo_label[1]) + consistency_criterion(y_ori[0],y_pseudo_label[2]) \
                            + consistency_criterion(y_ori[3], y_pseudo_label[1]) + consistency_criterion(y_ori[3], y_pseudo_label[2]) \
                            + consistency_criterion(y_ori[0], y_pseudo_label[3]) + consistency_criterion(y_ori[0],y_pseudo_label[3])

            # loss_consist = consistency_criterion(y_ori[0], y_pseudo_label[1]) + consistency_criterion(y_ori[0],y_pseudo_label[1])

            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num // 150)  # 更新权值
            loss = args.lamda * loss_seg_dice \
                    + consistency_weight * loss_consist \
                    + loss_xianyan * 0.1


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f ,loss_xianyan: %03f' % (
                iter_num, loss, loss_seg_dice, loss_consist, loss_xianyan))

            # loss1.append(loss_seg_dice)
            # loss2.append(loss_consist)
            # loss3.append(loss)
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                  classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    # return loss1, loss2, loss3


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    snapshot_path = "../model/{}_{}_{}_labeled/{}".format(args.data_name, args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # loss_seg_dice, loss_consist, loss = train(args, snapshot_path)
    train(args, snapshot_path)
    # plot_loss_path = "../model/{}_{}_{}_labeled/{}/plot_losses/".format(args.data_name, args.exp, args.labelnum, args.model)
    # if os.path.exists(plot_loss_path):
    #     shutil.rmtree(plot_loss_path)
    # os.makedirs(plot_loss_path)

    # plot_loss_cice(loss_seg_dice, plot_loss_path)
    # plot_loss_consist(loss_consist, plot_loss_path)
    # plot_loss(loss, plot_loss_path)
