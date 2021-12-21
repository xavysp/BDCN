import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import time
import re
import os
import sys
import cv2
import ablation
from datasets.dataset import *
import argparse
import cfg


def test(model, args, device=None):
    test_root = cfg.config_test[args.test_data]['data_root']
    test_lst = cfg.config_test[args.test_data]['data_lst']
    test_name_lst = os.path.join(test_root, test_lst)

    if 'Multicue' in args.test_data:
        test_lst = test_lst % args.k

    mean_bgr = np.array(cfg.config_test[args.test_data]['mean_bgr'])

    test_img = Data_test(test_root, test_lst, 0.5,
                         mean_bgr=mean_bgr,dataset_name=args.test_data)
    testloader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=8)
    # lst = np.loadtxt(test_name_lst, dtype=str)[:, 0]
    # nm = [os.path.splitext(os.path.split(x)[-1])[0] for x in lst]
    save_dir = os.path.join(
        'results', 'edges',args.train_data+'-B'+str(args.block)+str(2)+args.test_data)
    os.makedirs(save_dir,exist_ok=True)

    model.eval()
    data_iter = iter(testloader)
    iter_per_epoch = len(testloader)
    start_time = time.time()
    all_t = 0
    for i, (data, _) in enumerate(testloader):
        # if args.cuda:
        #     data = data.cuda()
        # data = Variable(data, volatile=True)
        data =data.to(device)
        t1 = time.time()
        out = model(data)
        name = testloader.dataset.images_name[i]
        t = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
        print(i, 'predicted data saving ',os.path.join(save_dir, 'fuse', '%s.png'%name))
        if not os.path.exists(os.path.join(save_dir, 'fuse')):
            os.mkdir(os.path.join(save_dir, 'fuse'))
        cv2.imwrite(os.path.join(save_dir, 'fuse', '%s.png'%name), 255-t*255)
        all_t += time.time() - t1

    print("Total Time", all_t)
    print('Overall Time use: ', time.time() - start_time)

def main():
    import time
    print(time.localtime())
    args = parse_args()
    args.bdcn = not args.no_bdcn
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    model = ablation.BDCN(ms=args.ms, block=args.block, bdcn=not args.no_bdcn,
        direction=args.dir, k=args.num_conv, rate=args.rate).to(device)
    chckpnt_dir = os.path.join('params',args.train_data+'-B'+str(args.block), args.model)
    model.load_state_dict(torch.load(chckpnt_dir,map_location=device))
    print('Successfuly checkpoint loaded ', chckpnt_dir)
    test(model, args, device=device)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('--test_data', type=str,
        default='BRIND', help='The dataset to test')
    parser.add_argument('--train_data', type=str,
        default='BIPED', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='bdcn_20000.pth',
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='result',
        help='the dir to store result')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    parser.add_argument('--ms', action='store_true', default=False,
        help='whether employ the ms blocks, default False')
    parser.add_argument('--block', type=int, default=3,
        help='how many blocks of the model, default 5')
    parser.add_argument('--no-bdcn', action='store_true', default=False,
        help='whether to employ our policy to train the model, default False')
    parser.add_argument('--dir', type=str, choices=['both', 's2d', 'd2s'], default='both',
        help='the direction of cascade, default both')
    parser.add_argument('--num-conv', type=int, choices=[0,1,2,3,4], default=3,
        help='the number of convolution of SEB, default 3')
    parser.add_argument('--rate', type=int, default=4,
        help='the dilation rate of scale enhancement block, default 4')
    return parser.parse_args()

if __name__ == '__main__':
    main()
