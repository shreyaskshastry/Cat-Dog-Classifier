import numpy as np
from os import listdir
import keras
from sklearn.model_selection import train_test_split
from skimage import io
from scipy.misc import imresize

def get_img(data_path):
    # Getting image array from path:
    img_size = 64
    img = io.imread(data_path)
    img = imresize(img, (img_size, img_size, 3))
    return img

def load_datasets(path = 'E:\\python\\DEEPLearning\\Data\\Train_Data'):
    try:
        X = np.load('E:\\python\\DEEPLearning\\Data\\npy_train_data\\X.npy')
        Y = np.load('E:\\python\\DEEPLearning\\Data\\npy_train_data\\Y.npy')
    except:
        labels = listdir(path)
        print('Categories:\n', labels)
        lendata = 0
        for label in (labels):
            lendata += len(listdir(path + '\\' + label))
        X = np.zeros((lendata, 64, 64, 3), dtype='float64')
        Y = np.zeros(lendata)
        count_data = 0
        count_categori = [-1, '']  # For encode labels
        for label in labels:
            dpath = path + '\\' + label
            for data in listdir(dpath):
                img = get_img(dpath + '\\' + data)
                X[count_data] = img
                # For encode labels:
                if label != count_categori[1]:
                    count_categori[0] += 1
                    count_categori[1] = label
                Y[count_data] = count_categori[0]
                count_data += 1
        #creating the numpy dataset
        Y = keras.utils.to_categorical(Y)
        import os
        if not os.path.exists('E:\\python\\DEEPLearning\\Data\\npy_train_data'):
            os.makedirs('E:\\python\\DEEPLearning\\Data\\npy_train_data')
        np.save('E:\\python\\DEEPLearning\\Data\\npy_train_data\\X.npy', X)
        np.save('E:\\python\\DEEPLearning\\Data\\npy_train_data\\Y.npy', Y)
    X = X/255
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test
