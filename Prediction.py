from keras.models import model_from_json
from Essentials import get_img
import numpy as np

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y

if __name__ == '__main__':
    img_path = 'C:\\Users\\shrey\\Desktop\\dog4.jpg'
    img = get_img(img_path)
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, X)
    print('It is a ' + Y + ' !')
