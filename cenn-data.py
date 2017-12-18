from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob

class dataProcess(object):

	def __init__(self, out_rows, out_cols, 
		data_path = "/home/wl/keras/examples/RSSNET/data/train/image", 
		label_path = "/home/wl/keras/examples/RSSNET/data/train/label", 
		test_path = "/home/wl/keras/examples/RSSNET/data/test", 
		npy_path = "/home/wl/keras/examples/RSSNET/npydata"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.test_path = test_path
		self.npy_path = npy_path

	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*.tif")
		labels = glob.glob(self.label_path+"/*.png")
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(labels),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.data_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			imgdatas[i] = img
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		i=0
		for imgname in labels:
			midname = imgname[imgname.rindex("/")+1:]
			label = load_img(self.label_path + "/" + midname,grayscale = True)
			label = img_to_array(label)
			imglabels[i] = label
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/label_train.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*.tif")
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.test_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			imgdatas[i] = img
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		label_train = np.load(self.npy_path+"/label_train.npy")
		imgs_train = imgs_train.astype('float32')
		label_train = label_train.astype('float32')
		imgs_train /= 255	
		label_train /= 255
		label_train[label_train > 0.5] = 1
		label_train[label_train <= 0.5] = 0
		return imgs_train,label_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255	
		return imgs_test

if __name__ == "__main__":
	mydata = dataProcess(173,259)
	mydata.create_train_data()
	mydata.create_test_data()
