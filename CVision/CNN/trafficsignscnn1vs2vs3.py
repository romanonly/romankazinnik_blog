from numpy import random
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from matplotlib import pyplot as plt
from skimage import io, color, exposure, transform

from trafficsignslib import *

# Main parameters
NUM_CLASSES = 43
IMG_SIZE = 48

root_dir = '/home/roman/PycharmProjects/NN/GTSRB/Final_Training/Images/'
root_GTSRB = '/home/roman/PycharmProjects/NN/GTSRB'

lr = 0.01
batch_size = 32
nb_epoch = 25
nTrainImg = 5000

if __name__ == "__main__":

    print(" Load Train data ")
    X,Y = init(root_dir, NUM_CLASSES, IMG_SIZE)

    items = range(1, X.shape[0])
    xlen = min(nTrainImg,X.shape[0])

    new_items = random.choice(items, xlen) #.choices(items, k = 3)
    XX = X[new_items, :,:]
    YY = Y[new_items]
    print(XX.shape)

    print(" Load Test data ")
    X_test, y_test = read_test(root_GTSRB, IMG_SIZE)

    print("Training")

    # train the model using SGD + momentum (how original).
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    modelTT = concat_cnn_model(IMG_SIZE, NUM_CLASSES, sgd, isAddLayer1 = True, isTwoOutputs=True)
    modelT = concat_cnn_model(IMG_SIZE, NUM_CLASSES, sgd, isAddLayer1=True, isTwoOutputs= False)
    modelF = concat_cnn_model(IMG_SIZE, NUM_CLASSES, sgd, isAddLayer1=False)
    model = cnn_model(IMG_SIZE, NUM_CLASSES, sgd)

    print("*********** modelTT ****************")

    hstryTT = run_train_test(XX, [YY,YY], modelTT, X_test, y_test, batch_size, nb_epoch, h5name='modelTT.h5')#,loss_weights=[0.1, 1.0])
    #m, a = read_model(X_test, 1000, modelName_h5="modelTT.h5", modelName_json="modelTT.h5.json")
    print("*********** modelT ****************")
    hstryT = run_train_test(XX, YY, modelT, X_test, y_test, batch_size, nb_epoch, h5name='modelT.h5')
    print("************modelF ***************")
    hstryF = run_train_test(XX, YY, modelF, X_test, y_test, batch_size, nb_epoch, h5name='modelF.h5')
    print("************model ***************")
    hstry = run_train_test(XX, YY, model, X_test, y_test, batch_size, nb_epoch, h5name='model.h5')

    print(" Plot results")

    print(" hstryTT.history['val_out_layer2_acc'] = ", hstryTT.history['val_out_layer2_acc'])
    print(" hstryTT.history['out_layer2_acc'] = ", hstryTT.history['out_layer2_acc'])

    print(" hstryT.history['val_acc'] = ", hstryT.history['val_acc'])
    print(" hstryT.history['acc'] = ", hstryT.history['acc'])

    print(" hstryT.history['val_acc'] = ", hstryF.history['val_acc'])
    print(" hstryT.history['acc'] = ", hstryF.history['acc'])

    print(" hstryT.history['val_acc'] = ", hstry.history['val_acc'])
    print(" hstryT.history['acc'] = ", hstry.history['acc'])

    plt.figure()
    plt.plot(hstryTT.history['val_out_layer2_acc'], color="black")
    plt.plot(hstryT.history['val_acc'], color="r")
    plt.plot(hstryF.history['val_acc'], color="g")
    plt.plot(hstry.history['val_acc'], color="b")
    plt.title("val_acc")
    plt.show()

    plt.figure()
    plt.plot(hstryTT.history['out_layer2_acc'], color="black")
    plt.plot(hstryT.history['acc'], color="r")
    plt.plot(hstryF.history['acc'], color="g")
    plt.plot(hstry.history['acc'], color="b")
    plt.title("acc")
    plt.show()
