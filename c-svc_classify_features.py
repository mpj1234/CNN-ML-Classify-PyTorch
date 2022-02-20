import pickle
import time

import joblib
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import common


def train_svm_classifier(features, labels):
	# prepare training and test datasets 准备训练和测试数据集
	X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels,
	                                                                    test_size=0.3, random_state=42)

	# train and then classify the images with C-Support Vector Classification 训练然后使用 C-支持向量分类对图像进行分类
	param = [
		{
			"kernel": ["linear"],
			"C": [1, 10, 100, 1000]
		},
		{
			"kernel": ["rbf"],
			"C": [1, 10, 100, 1000],
			"gamma": [1e-2, 1e-3, 1e-4, 1e-5]
		}
	]

	print('C-Support Vector Classification starting ...')
	start_time = time.time()

	# request probability estimation 请求概率估计
	svm_c = SVC(probability=True)

	# 10-fold cross validation, use 4 thread as each fold and each parameter set can be trained in parallel
	# 10折交叉验证，每折使用4个线程，每个参数集可以并行训练
	clf = model_selection.GridSearchCV(svm_c, param, cv=2, n_jobs=4, verbose=3)

	clf.fit(X_train, y_train)

	# let us know the training outcome - so we don't have to do it again!
	# 让我们知道培训结果 - 这样我们就不必再做一次了！
	print("\nBest parameters set:")
	print(clf.best_params_)

	y_pred = clf.predict(X_test)
	print("%f seconds" % (time.time() - start_time))

	# save the C-SVC training results for future use
	# 保存 C-SVC 训练结果以备将来使用
	joblib.dump(clf.best_estimator_, './model/svc_estimator.pkl')
	joblib.dump(clf, './model/svc_clf.pkl')

	# confusion matrix computation and display
	# 混淆矩阵计算与显示
	print("CNN-C-SVC Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
	common.plot_confusion_matrix(y_test, y_pred, "C-SVC Confusion matrix", isTrain=True)


def c_svc_classify():
	# Classification

	# features and labels from feature extraction 特征提取的特征和标签
	loaded_features = pickle.load(open('model/train_features', 'rb'))
	loaded_labels = pickle.load(open('model/train_labels', 'rb'))

	train_svm_classifier(loaded_features, loaded_labels)


if __name__ == "__main__":
	c_svc_classify()
