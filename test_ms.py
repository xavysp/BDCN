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
import bdcn
from datasets.dataset import Data
import argparse
import cfg


def test(model, args,running_on='cpu'):
    test_root = cfg.config_test[args.test_data]['data_root']
    test_lst = cfg.config_test[args.test_data]['data_lst']
    test_name_lst = os.path.join(test_root, 'voc_valtest.txt')
    # if 'Multicue' in args.dataset:
    #     test_lst = test_lst % args.k
    #     test_name_lst = os.path.join(test_root, 'test%d_id.txt'%args.k)
    mean_bgr = np.array(cfg.config_test[args.test_data]['mean_bgr'])
    test_img = Data(test_root, test_lst, mean_bgr=mean_bgr, is_train=False, dataset_name=args.test_data,
                    scale=[0.5, 1, 1.5])
    testloader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=8)
    assert len(test_img ) >0

    base_dir = args.res_dir
    dataset_save_dir = os.path.join('edges', args.model_name + '_' + args.train_data + str(2) + args.test_data)

    save_dir = os.path.join(base_dir, dataset_save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.cuda:
        model.cuda()
    model.eval()
    data_iter = iter(testloader)
    iter_per_epoch = len(testloader)
    start_time = time.time()
    all_t = []

    with torch.no_grad():

        for i, (ms_data, label) in enumerate(testloader):
            ms_fuse = np.zeros((label.size()[2], label.size()[3]))
            tm = time.time()
            for data in ms_data:
                if args.cuda:
                    data = data.cuda()
                data = Variable(data, volatile=True)
                out = model(data)
                fuse = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
                fuse = cv2.resize(fuse, (label.size()[3], label.size()[2]), interpolation=cv2.INTER_LINEAR)
                ms_fuse += fuse
            ms_fuse /= len(ms_data)
            # tm = time.time()
            if not os.path.exists(os.path.join(save_dir, 'ms_pred')):
                os.mkdir(os.path.join(save_dir, 'ms_pred'))
            name = testloader.dataset.images_name[i]
            cv2.imwrite(os.path.join(save_dir, 'ms_pred', '%s.png'%name), np.uint8(255-ms_fuse*255))
            all_t.append(time.time() - tm)
            print('Done: ',name,'in ', i+1)
    all_t = np.array(all_t)
    print('Average time per image: ', all_t.mean())
    print ('Overall Time use: ', time.time() - start_time)

def main():
    import time
    print (time.localtime())
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

    model = bdcn.BDCN().to(device)
    model.load_state_dict(torch.load('%s' % (args.model), map_location=device))
    test(model, args, running_on=device)
def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--model_name', type=str,
                        default='BDCN', help='model name')
    parser.add_argument('--train_data', type=str,
                        default='BIPED', help='Dataset used 4 training')
    parser.add_argument('--test_data', type=str,
                        default='CID', help='The dataset 4 testing')  # choices=cfg.config_test.keys(),
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='params/bdcn_30000.pth',
                        help='the model to test')  # 'params/bdcn_3000.pth' 'params/bdcn_6000.pth' 'params/bdcn_pretrained_on_bsds500.pth'
    parser.add_argument('--res_dir', type=str, default='results',
                        help='the dir to store result')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    return parser.parse_args()

if __name__ == '__main__':
    main()
