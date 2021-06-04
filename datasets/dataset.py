import os
import cv2
import numpy as np
# from PIL import Image
import cv2 as cv
import json
# import scipy.io as io
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
		self.img_shape = []
		# self.files = np.loadtxt(lst_dir, dtype=str)
		if self.lst is not None:
			lst_dir = os.path.join(self.root, self.lst)
			if self.dataset_name.lower()=='biped':

				with open(lst_dir, 'r') as f:
					self.files = f.readlines()
					self.files = [line.strip().split(' ') for line in self.files]
				if not self.is_train:

					for i in range(len(self.files)):
						folder, filename = os.path.split(self.files[i][0])
						name, ext = os.path.splitext(filename)
						self.images_name.append(name)
						self.img_shape.append(None)
			else:
				with open(lst_dir) as f:
					files = json.load(f)
				self.files =[]
				for pair in files:
					tmp_img = pair[0]
					tmp_gt = pair[1]
					self.files.append(
						(os.path.join(self.root, tmp_img),
						 os.path.join(self.root, tmp_gt),))

		else:
			images_path = os.listdir(self.root)
			labels_path = [None for i in images_path]
			self.files = [images_path, labels_path]
			for i in range(len(self.files[0])):
				folder, filename = os.path.split(self.files[0][i])
				name, ext = os.path.splitext(filename)
				tmp_img = cv.imread(os.path.join(self.root,self.files[0][i]))
				tmp_shape = tmp_img.shape
				self.images_name.append(name)
				self.img_shape.append(tmp_shape)


	def __len__(self):
		lenght_data = len(self.files) if self.dataset_name !='CLASSIC' else len(self.files[0])
		return lenght_data

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		if self.dataset_name.lower() =='biped':
			base_im_dir = self.root+'imgs/train/' if self.is_train else self.root+'imgs/test/'
			img_file = base_im_dir + data_file[0]
		else:
			img_file = os.path.join(self.root,data_file[0])
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
			gt_file = data_file[1] if len(data_file)>1 else None
		# gt = Image.open(gt_file)
		if self.is_train:
			gt = cv.imread(gt_file, cv.IMREAD_GRAYSCALE)
			img = cv.resize(img,dsize=(500,500))  #500 for MDBD
			gt = cv.resize(gt,dsize=(500,500))
		else:
			gt=None
		return self.transform(img, gt)

	def transform(self, img, gt):
		if gt is not None:

			gt = np.array(gt, dtype=np.float32)
			if len(gt.shape) == 3:
				gt = gt[:, :, 0]
			gt /= 255.
			if self.yita is not None:
				gt[gt >= self.yita] = 1
				gt = np.clip(gt,0.,1.)
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
		if not self.is_train and gt is None:
			# img = cv.resize(img,dsize=(2400,2400)) # just for Robert dataset
			gt = np.zeros((img.shape[:2]))
			gt = torch.from_numpy(np.array([gt])).float()
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

class Data_test(data.Dataset):
	def __init__(self, root, lst, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892],),
		crop_size=None, rgb=False, scale=None,is_train=True,dataset_name='BIPED'):
		self.mean_bgr = mean_bgr
		self.dataset_name = dataset_name
		self.root = root
		self.lst = lst
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}
		self.images_name = []
		self.img_shape = []
		# self.files = np.loadtxt(lst_dir, dtype=str)
		if self.lst is not None:
			lst_dir = os.path.join(self.root, self.lst)
			with open(lst_dir, 'r') as f:
				self.files = f.readlines()
				self.files = [line.strip().split(' ') for line in self.files]

			for i in range(len(self.files)):
				folder, filename = os.path.split(self.files[i][0])
				name, ext = os.path.splitext(filename)
				self.images_name.append(name)
				self.img_shape.append(None)
		else:
			images_path = os.listdir(self.root)
			labels_path = [None for i in images_path]
			self.files = [images_path, labels_path]
			for i in range(len(self.files[0])):
				folder, filename = os.path.split(self.files[0][i])
				name, ext = os.path.splitext(filename)
				tmp_img = cv.imread(os.path.join(self.root,self.files[0][i]))
				tmp_shape = tmp_img.shape
				self.images_name.append(name)
				self.img_shape.append(tmp_shape)


	def __len__(self):
		lenght_data = len(self.files) if self.dataset_name !='CLASSIC' else len(self.files[0])
		return lenght_data

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		if self.dataset_name.lower() =='biped':
			base_im_dir = self.root+'imgs/test/'
			img_file = base_im_dir  + data_file[0]
		else:
			img_file = os.path.join(self.root,data_file[0])
		# print(img_file)
		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')
		# img = Image.open(img_file)
		# img = load_image_with_cache(img_file, self.cache)
		img = cv.imread(img_file)
		# load gt image
		if self.dataset_name.lower() =='biped':
			base_gt_dir = self.root+'edge_maps/test/'

		gt=None
		return self.transform(img, gt)

	def transform(self, img, gt):
		img = np.array(img, dtype=np.float32)
		if self.rgb:
			img = img[:, :, ::-1] # RGB->BGR
		img -= self.mean_bgr

		if gt is None:
			# img = cv.resize(img,dsize=(2400,2400)) # just for Robert dataset
			gt = np.zeros((img.shape[:2]))
			gt = torch.from_numpy(np.array([gt])).float()
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img.copy()).float()
		return img, gt
