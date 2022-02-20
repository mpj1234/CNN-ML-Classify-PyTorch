# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/2/15 20:34
  @version V1.0
"""
import os.path
import pickle
import random
import time

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC

import common as common
from log import log, destroyLog
from common import run_classifier

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

model_path = './imagenet'

images_dirs = ['./caltech_101_images/train/', './caltech_101_images/test/']

log.debug("***pytorch train start***")
global_start_time = time.time()


def extract_features(model, image_path_map, nb_features=2048):
	"""
	提取用来分类的特征
	:param nb_features: 提取出来的特征数量
	:param model: 模型
	:param image_path_map:
	:return:
	"""

	train_features = np.empty((len(image_path_map["train"]), nb_features))
	train_labels = []
	test_features = np.empty((len(image_path_map["test"]), nb_features))
	test_labels = []

	features = None
	labels = None

	for key, list_images in image_path_map.items():
		if key == "train":
			features = train_features
			labels = train_labels
		elif key == "test":
			features = test_features
			labels = test_labels

		common.predict_feature(model, features, labels, list_images)

	return train_features, train_labels, test_features, test_labels


if __name__ == '__main__':
	model_name = 'mnasnet1_0'
	model = common.create_model(model_path, model_name=model_name)
	number_features = 1280
	# model = common.create_model(model_path, model_name='inception_v3')
	# number_features = 2048
	# model = common.create_model(model_path, model_name='resnet18')
	# number_features = 512
	# model = common.create_model(model_path, model_name='resnet34')
	# number_features = 512
	# model = common.create_model(model_path, model_name='resnet50')
	# number_features = 2048
	# model = common.create_model(model_path, model_name='resnext50_32x4d')
	# number_features = 2048
	# model = common.create_model(model_path, model_name='densenet121')
	# number_features = 1024
	# model = common.create_model(model_path, model_name='mnasnet0_5')
	# number_features = 1280
	# model = common.create_model(model_path, model_name='mnasnet1_0')
	# number_features = 1280
	log.debug(f"train model name: {model_name}")
	model_load_after_start_time = time.time()

	# 加载数据
	if os.path.exists("./model/train_features"):
		log.debug("Pre-extracted train_features and labels exist, loading them...")
		train_features = pickle.load(open("./model/train_features", "rb"))
		train_labels = pickle.load(open("./model/train_labels", "rb"))
		test_features = pickle.load(open("./model/test_features", "rb"))
		test_labels = pickle.load(open("./model/test_labels", "rb"))
	else:
		log.debug("Pre-extracted train_features and labels not exist, extracting them...")
		image_path_map = {}
		for images_dir in images_dirs:
			list_images = common.get_images_list(images_dir)
			if images_dir.find("train") != -1:
				image_path_map["train"] = list_images
			elif images_dir.find("test") != -1:
				image_path_map["test"] = list_images

		train_features, train_labels, test_features, test_labels = extract_features(model, image_path_map,
		                                                                            number_features)

	pickle.dump(train_features, open('./model/train_features', 'wb'))
	pickle.dump(train_labels, open('./model/train_labels', 'wb'))
	pickle.dump(test_features, open('./model/test_features', 'wb'))
	pickle.dump(test_labels, open('./model/test_labels', 'wb'))
	log.debug('CNN features obtained and saved.')

	# 分类
	if os.path.exists("./model/tsne_train_features.npz"):
		log.debug("Pre-extracted tsne_train_features exist, loading them...")
		tsne_train_features = np.load("./model/tsne_train_features.npz")['tsne_features']
		tsne_test_features = np.load("./model/tsne_test_features.npz")['tsne_features']
	else:
		log.debug("Pre-extracted tsne_train_features not exist, extracting them...")
		tsne_train_features = TSNE().fit_transform(train_features)
		tsne_test_features = TSNE().fit_transform(test_features)
		np.savez("./model/tsne_train_features.npz", tsne_features=tsne_train_features)
		np.savez("./model/tsne_test_features.npz", tsne_features=tsne_test_features)
	common.plot_features(train_labels, tsne_train_features, "tsne_train_features", isTrain=True)
	common.plot_features(test_labels, tsne_test_features, "tsne_test_features", isTrain=True)

	X_train, X_test, y_train, y_test = train_features, test_features, train_labels, test_labels
	log.debug('Training and test datasets prepared.')
	log.debug('Training dataset size: %d' % len(X_train))
	log.debug('Test dataset size: %d' % len(X_test))

	# classify the images with a Linear Support Vector Machine (SVM)
	log.debug('Support Vector Machine LinearSVC starting ...')
	clf = LinearSVC()
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-LinearSVC Accuracy: {0:0.1f}%",
	               "LinearSVC Confusion matrix")
	pickle.dump(clf, open('./model/LinearSVC.pkl', 'wb'))

	# Best parameters set:
	# {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
	# 153.569978 seconds
	# CNN-C-SVC Accuracy: 94.5%
	log.debug('Support Vector Machine SVC finished.')
	clf = SVC(C=10, gamma=0.001, kernel='rbf')
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-SVC Accuracy: {0:0.1f}%", "SVC Confusion matrix")
	pickle.dump(clf, open('./model/SVC.pkl', 'wb'))

	# RandomForestClassifier/ExtraTreesClassifier defaults:
	# (n_estimators=10, criterion='gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
	# min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
	# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,
	# class_weight=None)

	# classify the images with an Extra Trees Classifier
	log.debug('Extra Trees Classifier starting ...')
	clf = ExtraTreesClassifier(n_jobs=4, n_estimators=100, criterion='gini', min_samples_split=10,
	                           max_features=50, max_depth=40, min_samples_leaf=4)
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-ET Accuracy: {0:0.1f}%", "Extra Trees Confusion matrix")
	pickle.dump(clf, open('./model/ExtraTreesClassifier.pkl', 'wb'))

	# classify the images with a Random Forest Classifier
	log.debug('Random Forest Classifier starting ...')
	clf = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-RF Accuracy: {0:0.1f}%",
	               "Random Forest Confusion matrix")
	pickle.dump(clf, open('./model/RandomForestClassifier.pkl', 'wb'))

	# KNeighborsClassifier defaults:
	# n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None,
	# n_jobs=1, **kwargs

	# classify the images with a k-Nearest Neighbors Classifier
	log.debug('K-Nearest Neighbours Classifier starting ...')
	clf = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-KNN Accuracy: {0:0.1f}%",
	               "K-Nearest Neighbor Confusion matrix")
	pickle.dump(clf, open('./model/KNeighborsClassifier.pkl', 'wb'))

	# MPLClassifier defaults:
	# hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’,
	# learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None,
	# tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
	# validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08

	# classify the image with a Multi-layer Perceptron Classifier
	log.debug('Multi-layer Perceptron Classifier starting ...')
	clf = MLPClassifier()
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-MLP Accuracy: {0:0.1f}%",
	               "Multi-layer Perceptron Confusion matrix")
	pickle.dump(clf, open('./model/MLPClassifier.pkl', 'wb'))

	# GaussianNB defaults:
	# priors=None

	# classify the images with a Gaussian Naive Bayes Classifier
	log.debug('Gaussian Naive Bayes Classifier starting ...')
	clf = GaussianNB()
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-GNB Accuracy: {0:0.1f}%",
	               "Gaussian Naive Bayes Confusion matrix")
	pickle.dump(clf, open('./model/GaussianNB.pkl', 'wb'))

	# LinearDiscriminantAnalysis defaults:
	# solver=’svd’, shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001

	# classify the images with a Linear Discriminant Analysis Classifier
	log.debug('Linear Discriminant Analysis Classifier starting ...')
	clf = LinearDiscriminantAnalysis()
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-LDA Accuracy: {0:0.1f}%",
	               "Linear Discriminant Analysis Confusion matrix")
	pickle.dump(clf, open('./model/LinearDiscriminantAnalysis.pkl', 'wb'))

	# QuadraticDiscriminantAnalysis defaults:
	# priors=None, reg_param=0.0, store_covariance=False, tol=0.0001, store_covariances=None

	# classify the images with a Quadratic Discriminant Analysis Classifier
	log.debug('Quadratic Discriminant Analysis Classifier starting ...')
	clf = QuadraticDiscriminantAnalysis()
	run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-QDA Accuracy: {0:0.1f}%",
	               "Quadratic Discriminant Analysis Confusion matrix")
	pickle.dump(clf, open('./model/QuadraticDiscriminantAnalysis.pkl', 'wb'))

	log.debug(f'model_load_after_time: {time.time() - model_load_after_start_time}')
	log.debug(f'pytorch train classification finished total time: {time.time() - global_start_time}')
destroyLog()
