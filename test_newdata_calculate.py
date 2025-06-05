import argparse
import os
import shutil

import h5py

import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from utils.plot import display_images
from utils.niitopng import nii_to_image
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/tooth', help='Name of Experiment')
parser.add_argument('--data_name', type=str, default='tooth', help='Name of unlabel')
parser.add_argument('--exp', type=str, default='DTS_OAP_v4_attention', help='experiment_name')
parser.add_argument('--model', type=str, default='DTS_OAP_v4_attention', help='model_name')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=14, help='labeled unlabel')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, pre_out_nii, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (256 / x, 256 / y), order=0)
    label = zoom(label, (256 / x, 256 / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        if len(out_main) > 1:
            # out_main = (out_main[0] + out_main[3])/2
            out_main = out_main[0]
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        # display_images(slice, label, out)
        pred = zoom(out, (x / 256, y / 256), order=0)
        lableout = zoom(label, (x / 256, y / 256), order=0)
        prediction = out
    if np.sum(prediction == 1) == 0:
        first_metric = 0, 0, 0, 0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2) == 0:
        second_metric = 0, 0, 0, 0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3) == 0:
        third_metric = 0, 0, 0, 0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(pred.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(lableout.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    #
    # sitk.WriteImage(prd_itk, pre_out_nii + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, pre_out_nii + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, pre_out_nii + case + "_gt.nii.gz")

    sitk.WriteImage(prd_itk, pre_out_nii + "pred_{}.nii.gz".format(case))
    # sitk.WriteImage(img_itk, pre_out_nii + "img_{}.nii.gz".format(case))
    sitk.WriteImage(lab_itk, pre_out_nii + "gt_{}.nii.gz".format(case))

    # nii_to_image(pre_out_nii, test_save_path)
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "../model/{}_{}_{}_labeled/{}".format(FLAGS.data_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    test_save_path = "../model/{}_{}_{}_labeled/{}/predictions/".format(FLAGS.data_name, FLAGS.exp, FLAGS.labelnum,
                                                                        FLAGS.model)
    pre_out_nii = "../model/{}_{}_{}_labeled/{}/pre_nii/".format(FLAGS.data_name, FLAGS.exp, FLAGS.labelnum,
                                                                 FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    if os.path.exists(pre_out_nii):
        shutil.rmtree(pre_out_nii)
    os.makedirs(pre_out_nii)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path), strict=False)
    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, pre_out_nii, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print((metric[0] + metric[1] + metric[2]) / 3)
    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0] + metric[1] + metric[2]) / 3))
