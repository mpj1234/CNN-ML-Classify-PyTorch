# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/2/18 16:24
  @version V1.0
"""
import os.path
import pickle
import random
import time

import numpy as np
import torch
from sklearn.manifold import TSNE

import common as common
from log import log, destroyLog

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

model_path = './imagenet'

images_dir = './caltech_101_images/test/'

# 是否导入特征处理好的文件直接用来分类
isLoadModel = False

log.debug("***pytorch test start***")
global_start_time = time.time()


# Classifier performance
def run_classifier(clf, x_test_data, y_test_data, acc_str, matrix_header_str):
	common.run_classifier(clf, None, None, x_test_data, y_test_data, acc_str, matrix_header_str, isTrain=False)


def extract_features(model, list_images, nb_features=2048):
	"""
	提取用来分类的特征
	:param nb_features: 提取出来的特征数量
	:param model: 模型
	:param list_images 图片列表
	:return:
	"""
	nb_features = nb_features
	test_features = np.empty((len(list_images), nb_features))
	test_labels = []

	common.predict_feature(model, test_features, test_labels, list_images)

	return test_features, test_labels


if __name__ == '__main__':
	model_name = 'mnasnet0_5'
	model = common.create_model(model_path, model_name=model_name)
	number_features = 1280
	log.debug(f"test model name: {model_name}")
	model_load_after_start_time = time.time()

	# 加载数据
	if os.path.exists("./model/train_features") and isLoadModel:
		log.debug("Pre-extracted train_features and labels exist, loading them...")
		test_features = pickle.load(open("./model/test_features", "rb"))
		test_labels = pickle.load(open("./model/test_labels", "rb"))
	else:
		log.debug("Pre-extracted train_features and labels not exist, extracting them...")
		list_images = common.get_images_list(images_dir)

		test_features, test_labels = extract_features(model, list_images, number_features)

	pickle.dump(test_features, open('./model/test_features', 'wb'))
	pickle.dump(test_labels, open('./model/test_labels', 'wb'))
	log.debug('CNN features obtained and saved.')

	# 分类
	if os.path.exists("./model/tsne_test_features.npz") and isLoadModel:
		log.debug("Pre-extracted tsne_test_features exist, loading them...")
		tsne_test_features = np.load("./model/tsne_test_features.npz")['tsne_features']
	else:
		log.debug("Pre-extracted tsne_test_features not exist, extracting them...")
		tsne_test_features = TSNE().fit_transform(test_features)
		np.savez("./model/tsne_test_features.npz", tsne_features=tsne_test_features)

	common.plot_features(test_labels, tsne_test_features, "tsne_test_features", isTrain=False)

	X_test, y_test = test_features, test_labels
	log.debug('test datasets prepared.')
	log.debug('Test dataset size: %d' % len(X_test))

	# classify the images with a Linear Support Vector Machine (SVM)
	log.debug('Support Vector Machine LinearSVC starting ...')
	clf = pickle.load(open('./model/LinearSVC.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-LinearSVC Accuracy: {0:0.1f}%", "LinearSVC Confusion matrix")

	log.debug('Support Vector Machine SVC finished.')
	clf = pickle.load(open('./model/SVC.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-SVC Accuracy: {0:0.1f}%", "SVC Confusion matrix")

	# classify the images with an Extra Trees Classifier
	log.debug('Extra Trees Classifier starting ...')
	clf = pickle.load(open('./model/ExtraTreesClassifier.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-ET Accuracy: {0:0.1f}%", "Extra Trees Confusion matrix")

	# classify the images with a Random Forest Classifier
	log.debug('Random Forest Classifier starting ...')
	clf = pickle.load(open('./model/RandomForestClassifier.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-RF Accuracy: {0:0.1f}%", "Random Forest Confusion matrix")

	# classify the images with a k-Nearest Neighbors Classifier
	log.debug('K-Nearest Neighbours Classifier starting ...')
	clf = pickle.load(open('./model/KNeighborsClassifier.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-KNN Accuracy: {0:0.1f}%", "K-Nearest Neighbor Confusion matrix")

	# classify the image with a Multi-layer Perceptron Classifier
	log.debug('Multi-layer Perceptron Classifier starting ...')
	clf = pickle.load(open('./model/MLPClassifier.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-MLP Accuracy: {0:0.1f}%", "Multi-layer Perceptron Confusion matrix")

	# classify the images with a Gaussian Naive Bayes Classifier
	log.debug('Gaussian Naive Bayes Classifier starting ...')
	clf = pickle.load(open('./model/GaussianNB.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-GNB Accuracy: {0:0.1f}%", "Gaussian Naive Bayes Confusion matrix")

	# classify the images with a Linear Discriminant Analysis Classifier
	log.debug('Linear Discriminant Analysis Classifier starting ...')
	clf = pickle.load(open('./model/LinearDiscriminantAnalysis.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-LDA Accuracy: {0:0.1f}%", "Linear Discriminant Analysis Confusion matrix")

	# classify the images with a Quadratic Discriminant Analysis Classifier
	log.debug('Quadratic Discriminant Analysis Classifier starting ...')
	clf = pickle.load(open('./model/QuadraticDiscriminantAnalysis.pkl', 'rb'))
	run_classifier(clf, X_test, y_test, "CNN-QDA Accuracy: {0:0.1f}%",
	               "Quadratic Discriminant Analysis Confusion matrix")

	log.debug(f'model_load_after_time: {time.time() - model_load_after_start_time}')
	log.debug(f'test classification finished total time: {time.time() - global_start_time}')

	destroyLog()
