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
from datasets.dataset import *
import argparse
import cfg

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))


def test(model, args, running_on='cpu'):

    test_root = cfg.config_test[args.test_data]['data_root']
    test_lst = cfg.config_test[args.test_data]['data_lst']
    test_name_lst = os.path.join(test_root, 'voc_valtest.txt')
    # test_name_lst = os.path.join(test_root, 'test_id.txt')
    # if 'Multicue' in args.test_data:
    #     test_lst = test_lst % args.k
    #     test_name_lst = os.path.join(test_root, 'test%d_id.txt'%args.k)
    mean_bgr = np.array(cfg.config_test[args.test_data]['mean_bgr'])
    test_img = Data_test(test_root, test_lst, mean_bgr=mean_bgr,is_train=False,dataset_name=args.test_data)
    testloader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=8)
    # nm = np.loadtxt(test_name_lst, dtype=str)
    # print(len(testloader), len(nm))
    assert len(test_img ) >0
    save_res = True
    base_dir = args.res_dir
    dataset_save_dir = os.path.join('edges',args.model_name+'_'+args.train_data+str(2)+args.test_data)

    save_dir = os.path.join(base_dir,dataset_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("Saving in: ", save_dir)

    if running_on.__str__()=='gpu':
        model.cuda()
    model.eval()
    data_iter = iter(testloader)
    iter_per_epoch = len(testloader)
    start_time = time.time()
    all_t = []
    with torch.no_grad():
        for i, (data,_) in enumerate(testloader):
            # data,_= test_data if len(test_data)>1 else [test_data,None]
            if running_on.__str__()=='gpu' or running_on.__str__()=='cuda':
                data = data.cuda()
            # data = Variable(data)
            tm = time.time()
            out = model(data)
            fuse = torch.sigmoid(out[-1]).detach().cpu().data.numpy()[0, 0, :, :]
            if not os.path.exists(os.path.join(save_dir, 'pred')):
                os.mkdir(os.path.join(save_dir, 'pred'))
            name = testloader.dataset.images_name[i]
            tmp_shape = testloader.dataset.img_shape[i]
            img_shape = tmp_shape if tmp_shape is not None else None
            if img_shape is not None:
                fuse = cv2.resize(fuse, dsize=(img_shape[1], img_shape[0]))
            cv2.imwrite(os.path.join(save_dir, 'pred', '%s.png'%name), np.uint8(255-fuse*255))
            all_t.append(time.time() - tm)
            print('Done: ',name,'in ', i+1, 'shape:', fuse.shape)
            torch.cuda.empty_cache()

    all_t = np.array(all_t)
    print (all_t.sum())
    print ('Overall Time use: ', time.time() - start_time)
    print ('Average time per image: ', all_t.mean())
    print("+++ Testing on {} data done, saved in: {}".format(args.test_data, save_dir), " +++")

def main():
    import time
    print (time.localtime())
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    # model = bdcn.BDCN()
    model = bdcn.BDCN().to(device)
    model_dir = os.path.join('params',args.train_data,args.model)
    model.load_state_dict(torch.load('%s' % (model_dir),map_location=device))
    print("====== Checkpoint> ", model_dir,"==========")
    test(model, args, running_on=device)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--model_name', type=str,
                        default='BDCN', help='model name')
    parser.add_argument('--train_data', type=str,
        default='BIPED', help='Dataset used 4 training')

    parser.add_argument('--test_data', type=str,
                        default='BIPED', help='The dataset 4 testing') #choices=cfg.config_test.keys(),
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether use gpu to train network')
    # parser.add_argument('-c', '--cuda', action='store_true',
    #     help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='bdcn_20000.pth',
        help='the model to test') # 'params/bdcn_3000.pth' 'params/bdcn_6000.pth' 'params/bdcn_pretrained_on_bsds500.pth'
    # 'params/bdcn_8000.pth'
    parser.add_argument('--res_dir', type=str, default='results',
        help='the dir to store result')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    return parser.parse_args()

if __name__ == '__main__':
    main()
