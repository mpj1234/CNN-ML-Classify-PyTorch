# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/2/15 18:34
  @version V1.0
"""
import logging
import os
from datetime import datetime


class ContextHandle(logging.Handler):

	def __init__(self, path, console=False):
		super(ContextHandle, self).__init__()
		self.filenames = []
		self.writer = None
		self.filename()
		self.path = path
		self.console = console
		if path is not None:
			self.writer = open(path, "a")
			self.writer.write("*" * 50 + "logs start" + "*" * 50 + "\n")

	def filename(self):
		for root, dirs, files in os.walk("./"):
			for file in files:
				if os.path.splitext(file)[1] == '.py':
					self.filenames.append(file)

	def emit(self, record):
		if record.filename in self.filenames:
			msg = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ' \
			      f'{record.filename} {record.funcName} {record.levelname}] {record.msg}\n'
			if self.path is not None:
				self.writer.write(msg)
			if self.console:
				print(msg, end="")


__handle = ContextHandle("logs/CNN_ML.log", True)
logging.basicConfig(
	level=logging.DEBUG,
	handlers=[__handle]
)
log = logging.getLogger("logs")


def destroyLog():
	if __handle.writer is not None:
		__handle.writer.write("*" * 50 + "logs end" + "*" * 52 + "\n")
		__handle.writer.flush()
		__handle.writer.close()
