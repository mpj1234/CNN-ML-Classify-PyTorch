# -*- coding: UTF-8 -*-
"""
  用来进行数据集分割，我一开始把数据集都放到了chicken/train目录下，
  然后分成了两个部分，一个是训练集，一个是测试集，将训练集放到了chicken/test目录下，
  训练集大小为0.7，测试集大小为0.3
  @Author: mpj
  @Date  : 2022/2/16 15:11
  @version V1.0
"""
import os
import shutil
import random

trainPath = './caltech_101_images/train'
testPath = './caltech_101_images/test'

testRage = 0.3

dirList = os.listdir(trainPath)
for dirName in dirList:
	if os.path.isdir(trainPath + "/" + dirName):
		os.mkdir(testPath + "/" + dirName)
		fileList = os.listdir(trainPath + '/' + dirName)
		fileNum = len(fileList)
		testNum = int(fileNum * testRage)
		for i in range(testNum):
			randomIndex = random.randint(0, fileNum - 1)
			fileName = fileList[randomIndex]
			# 已经准备移动这个文件，就从fileList中删除这个文件，文件数量减一，防止重复移动
			del fileList[randomIndex]
			fileNum -= 1
			shutil.move(trainPath + '/' + dirName + '/' + fileName, testPath + '/' + dirName + '/' + fileName)
		print(dirName + ": " + str(fileNum) + " -> " + str(testNum))
