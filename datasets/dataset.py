import os
import cv2
import numpy as np
from PIL import Image
import cv2 as cv
import scipy.io as io
import torch
from torch.utils import data
import random
from datasets.utls import cv_imshow
#from cStringIO import StringIO

# def load_image_with_cache(path, cache=None, lock=None):
# 	if cache is not None:
# 		if not cache.has_key(path):
# 			with open(path, 'rb') as f:
# 				cache[path] = f.read()
# 		return Image.open(io.StringIO(cache[path]))
# 	return cv.imread(path)


class Data(data.Dataset):
	def __init__(self, root, lst, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892],),
		crop_size=None, rgb=False, scale=None,is_train=True,dataset_name='BIPED'):
		self.mean_bgr = mean_bgr
		self.is_train = is_train
		self.dataset_name = dataset_name
		self.root = root
		self.lst = lst
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}
		self.images_name = []
		lst_dir = os.path.join(self.root, self.lst)
		# self.files = np.loadtxt(lst_dir, dtype=str)
		with open(lst_dir, 'r') as f:
			self.files = f.readlines()
			self.files = [line.strip().split(' ') for line in self.files]
		if not self.is_train:

			for i in range(len(self.files)):
				folder, filename = os.path.split(self.files[i][0])
				name, ext = os.path.splitext(filename)
				self.images_name.append(name)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		if self.dataset_name.lower() =='biped':
			base_im_dir = self.root+'imgs/train/' if self.is_train else self.root+'imgs/test/'
			img_file = base_im_dir  + data_file[0]
		else:
			img_file = self.root + data_file[0]
		# print(img_file)
		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')
		# img = Image.open(img_file)
		# img = load_image_with_cache(img_file, self.cache)
		img = cv.imread(img_file)
		# load gt image
		if self.dataset_name.lower() =='biped':
			base_gt_dir = self.root+'edge_maps/train/' if self.is_train else self.root+'edge_maps/test/'
			gt_file  = base_gt_dir  + data_file[1]
		else:
			gt_file = self.root + data_file[1]
		# gt = Image.open(gt_file)
		gt = cv.imread(gt_file,cv.IMREAD_GRAYSCALE)
		if self.is_train:
			img = cv.resize(img,dsize=(400,400))
			gt = cv.resize(gt,dsize=(400,400))
		return self.transform(img, gt)

	def transform(self, img, gt):
		gt = np.array(gt, dtype=np.float32)
		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		gt /= 255.
		if self.yita is not None:
			gt[gt >= self.yita] = 1
		gt = torch.from_numpy(np.array([gt])).float()
		img = np.array(img, dtype=np.float32)
		if self.rgb:
			img = img[:, :, ::-1] # RGB->BGR
		img -= self.mean_bgr
		data = []
		if self.scale is not None and self.is_train:
			for scl in self.scale:
				img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
				data.append(torch.from_numpy(img_scale.transpose((2,0,1))).float())
			return data, gt
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img.copy()).float()
		if self.crop_size:
			_, h, w = gt.size()
			assert(self.crop_size < h and self.crop_size < w)
			i = random.randint(0, h - self.crop_size)
			j = random.randint(0, w - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
			gt = gt[:, i:i+self.crop_size, j:j+self.crop_size]
		return img, gt
