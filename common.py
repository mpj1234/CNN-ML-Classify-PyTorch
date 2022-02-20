# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/2/17 15:46
  @version V1.0
"""
import itertools
import os
import random
import re
import time
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torchvision.models import inception_v3, resnet18, resnet34, resnet50, resnext50_32x4d, densenet121, mnasnet0_5, \
	mnasnet1_0
from torchvision.transforms import transforms

from log import log


class DoNotHandle(nn.Module):
	"""
	写一个不做任何处理的层，用来替换inception v3最后的fc层的
	"""

	def __init__(self):
		super(DoNotHandle, self).__init__()

	def forward(self, x):
		return x


def create_model(path='./imagenet', model_name='inception_v3'):
	"""
	创建模型
	:param model_name: 模型名称
	:return: 创造好的模型
	:param path 文件存放路径
	"""
	os.environ['TORCH_HOME'] = path

	if model_name == 'inception_v3':
		model = inception_v3(pretrained=True)
		# 将最后的全连接层替换为不做任何处理的层
		model.fc = DoNotHandle()
	elif model_name == 'resnet18':
		model = resnet18(pretrained=True)
		model.fc = DoNotHandle()
	elif model_name == 'resnet34':
		model = resnet34(pretrained=True)
		model.fc = DoNotHandle()
	elif model_name == 'resnet50':
		model = resnet50(pretrained=True)
		model.fc = DoNotHandle()
	elif model_name == 'resnext50_32x4d':
		model = resnext50_32x4d(pretrained=True)
		model.fc = DoNotHandle()
	elif model_name == 'densenet121':
		model = densenet121(pretrained=True)
		model.classifier = DoNotHandle()
	elif model_name == 'mnasnet0_5':
		model = mnasnet0_5(pretrained=True)
		model.classifier = DoNotHandle()
	elif model_name == 'mnasnet1_0':
		model = mnasnet1_0(pretrained=True)
		model.classifier = DoNotHandle()
	else:
		raise ValueError('Model not supported')

	for param in model.parameters():
		param.requires_grad = False
	model.eval()
	# 设置为不训练
	model.eval()
	model.cuda()
	return model


def get_images_list(images_dir: str) -> List[str]:
	dir_list = [x[0] for x in os.walk(images_dir)]
	dir_list = dir_list[1:]
	list_images = []
	for image_sub_dir in dir_list:
		sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('jpg|JPG', f)]
		list_images.extend(sub_dir_images)
	# 讲图片目录打乱
	random.shuffle(list_images)
	return list_images


def predict_feature(model, features, labels, list_images):
	"""
	预测特征
	:param model: 模型
	:param features: 用来存放特征的列表
	:param labels: 用来存放标签的列表
	:param list_images: 保存图片的列表
	"""
	tf = transforms.Compose([
		lambda x: Image.open(x).convert('RGB'),
		transforms.ToTensor(),
	])

	for ind, image in enumerate(list_images):
		# 根据路径名获得图片的label
		im_label = image.split('/')[-2]

		# 打印提示信息
		if ind % 100 == 0:
			print('Processing', image, im_label)
		if not os.path.exists(image):
			log.warning('File not found: %s', image)
			continue

		# 读取图片，并转换成tensor，增加batch维度，并转换成cuda
		image_data = tf(image).unsqueeze(0).cuda()
		output = model(image_data)
		features[ind, :] = output.cpu().detach().numpy().squeeze()
		labels.append(im_label)


def plot_features(feature_labels, t_sne_features, name, isTrain):
	"""feature plot"""
	plt.figure(figsize=(9, 9), dpi=100)

	colors = itertools.cycle(["r", "b", "g", "c", "m", "y",
	                          "slategray", "plum", "cornflowerblue",
	                          "hotpink", "darkorange", "forestgreen",
	                          "tan", "firebrick", "sandybrown"])

	label_feature_map = {}
	for label, feature in zip(feature_labels, t_sne_features):
		if label not in label_feature_map:
			label_feature_map[label] = [feature]
		else:
			label_feature_map[label].append(feature)

	for label, features in label_feature_map.items():
		features = np.array(features)
		plt.scatter(features[:, 0], features[:, 1], c=next(colors), s=10, edgecolors='none')
		plt.annotate(label, xy=(np.mean(features[:, 0]), np.mean(features[:, 1])))

	save_path = './result/{}/{}.png'.format('train' if isTrain else 'test', name)
	plt.savefig(save_path)
	plt.show()


# Graphics
def plot_confusion_matrix_detail(y_true, y_pred, matrix_title, normalize=False, cmap=plt.cm.Blues, isTrain=True):
	"""confusion matrix computation and display"""
	plt.figure(figsize=(9, 9), dpi=100)

	# use sklearn confusion matrix
	cm_array = confusion_matrix(y_true, y_pred)

	if normalize:
		cm_array = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
		np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

	plt.imshow(cm_array, interpolation='nearest', cmap=cmap)
	plt.title(matrix_title, fontsize=16)

	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)

	true_labels = np.unique(y_true)
	pred_labels = np.unique(y_pred)
	xtick_marks = np.arange(len(true_labels))
	ytick_marks = np.arange(len(pred_labels))

	plt.xticks(xtick_marks, true_labels, rotation=90)
	plt.yticks(ytick_marks, pred_labels)
	fmt = '.2f' if normalize else 'd'
	thresh = cm_array.max() / 2.
	for i, j in itertools.product(range(cm_array.shape[0]), range(cm_array.shape[1])):
		plt.text(j, i, format(cm_array[i, j], fmt),
		         horizontalalignment="center",
		         color="white" if cm_array[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label', fontsize=14)
	plt.xlabel('Predicted label', fontsize=14)
	plt.tight_layout()

	save_path = './result/{}/{}.png'.format('train' if isTrain else 'test', matrix_title)
	plt.savefig(save_path)
	plt.show()


def plot_confusion_matrix(y_true, y_pred, matrix_title, cmap=plt.cm.Blues, isTrain=True):
	"""confusion matrix computation and display"""
	plt.figure(figsize=(9, 9), dpi=100)

	# use sklearn confusion matrix
	cm_array = confusion_matrix(y_true, y_pred)

	plt.imshow(cm_array, interpolation='nearest', cmap=cmap)
	plt.title(matrix_title, fontsize=16)

	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)

	true_labels = np.unique(y_true)
	pred_labels = np.unique(y_pred)
	xtick_marks = np.arange(len(true_labels))
	ytick_marks = np.arange(len(pred_labels))

	plt.xticks(xtick_marks, true_labels, rotation=90)
	plt.yticks(ytick_marks, pred_labels)
	plt.tight_layout()
	plt.ylabel('True label', fontsize=14)
	plt.xlabel('Predicted label', fontsize=14)
	plt.tight_layout()

	save_path = './result/{}/{}.png'.format('train' if isTrain else 'test', matrix_title)
	plt.savefig(save_path)
	plt.show()


# 用来详细显示混淆矩阵的结果
def run_classifier(clf, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str,
                   isTrain=True):
	"""run chosen classifier and display results"""
	start_time = time.time()
	if isTrain:
		clf.fit(x_train_data, y_train_data)
	y_pred = clf.predict(x_test_data)
	timeInterval = time.time() - start_time
	# print("%f seconds" % timeInterval)
	log.debug("%f seconds" % timeInterval)

	# confusion matrix computation and display
	# 混淆矩阵计算与显示
	score = accuracy_score(y_test_data, y_pred) * 100
	# print(acc_str.format(score))
	log.debug(acc_str.format(score))
	plot_confusion_matrix(y_test_data, y_pred, matrix_header_str, isTrain=isTrain)
