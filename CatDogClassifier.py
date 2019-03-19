from Essentials import load_datasets
from Modeldata import get_model,save_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator


def train_model(model, X, X_test, Y, Y_test):
    checkpoints = []
    checkpoints.append(ModelCheckpoint('E:\\python\\DEEPLearning\\Data\\Checkpoints\\best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='E:\\python\\DEEPLearning\\Data\\Checkpoints\.\logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))
    generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
    generated_data.fit(X)
    model.fit_generator(generated_data.flow(X, Y, batch_size=8), steps_per_epoch=X.shape[0]//8, epochs=25, validation_data=(X_test, Y_test), callbacks=checkpoints)

    return model

X,X_test,Y,Y_test = load_datasets()
model = get_model(len(Y[0]))
model = train_model(model, X, X_test, Y, Y_test)
save_model(model)


