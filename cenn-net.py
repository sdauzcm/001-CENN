import os 
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *

class myNet(object):

	def __init__(self, img_rows = 173, img_cols = 259):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, label_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, label_train, imgs_test

	def train(self):

		imgs_train, label_train, imgs_test = self.load_data()   

		inputs = Input((self.img_rows, self.img_cols,1))
		conv1 = Conv2D(1, 1, padding = 'same', kernel_initializer = 'he_normal')(inputs)
		model = Model(input = inputs, output = conv1)

		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
		
		model.fit(imgs_train, label_train, batch_size=4, nb_epoch=10)
		
		label_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('/home/wl/keras/examples/RSSNET/results/label_test.npy', label_test)

	def save_img(self):

		i=0
		print("array to image")
		imgs = np.load('/home/wl/keras/examples/RSSNET/results/label_test.npy')
		print imgs
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("/home/wl/keras/examples/RSSNET/results/label/%d.jpg"%(i))




if __name__ == '__main__':
	mynet = myNet()
	mynet.train()
	mynet.save_img()

