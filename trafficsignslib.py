import pandas as pd
import os
import glob
import h5py
import numpy as np
from skimage import io, color, exposure, transform

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.layers import Input
from keras.layers import Dense, Merge, Concatenate, concatenate


def preprocess_img(img, IMG_SIZE):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


def init2(root_dir, NUM_CLASSES, IMG_SIZE):
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        img = preprocess_img(io.imread(img_path), IMG_SIZE)
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File('X.h5', 'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

    try:
        with  h5py.File('X.h5') as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5")
    except (IOError, OSError, KeyError):
        print("Error in reading X.h5. Processing all images...")

def init(root_dir, NUM_CLASSES, IMG_SIZE):
    try:
        with  h5py.File('X.h5') as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5")

    except (IOError, OSError, KeyError):
        print("Error in reading X.h5. Processing all images...")
        #root_dir = 'GTSRB/Final_Training/Images/'
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path), IMG_SIZE)
                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)

                if len(imgs) % 1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        X = np.array(imgs, dtype='float32')
        Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

        with h5py.File('X.h5', 'w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)

    return X,Y

def read_test(root_GTSRB, IMG_SIZE):
    test = pd.read_csv(root_GTSRB + '/GT-final_test.csv', sep=';')

    X_test = []
    y_test = []
    i = 0
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join(root_GTSRB + '/Final_Test/Images/', file_name)
        X_test.append(preprocess_img(io.imread(img_path), IMG_SIZE))
        y_test.append(class_id)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_test, y_test

#===================
def concat_cnn_model(IMG_SIZE, NUM_CLASSES, sgd, isAddLayer1 = True, isTwoOutputs=True):
    # out2 = RightSideLastLayer(banblabala)(x)
    inp = Input(shape=(3, IMG_SIZE, IMG_SIZE,))  # Input((10,))
    x = Conv2D(32, (3, 3), padding='same', input_shape=(3, IMG_SIZE, IMG_SIZE), activation='relu')(inp)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    out2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    out2 = Conv2D(64, (3, 3), activation='relu')(out2)
    out2 = MaxPooling2D(pool_size=(2, 2))(out2)
    out2 = Dropout(0.2)(out2)

    out2 = Flatten()(out2)
    out2 = Dense(512, activation='relu')(out2)
    out2 = Dropout(0.5)(out2)

    if isAddLayer1:
        out1 = Flatten()(x)
        out1 = Dense(128, activation='relu')(out1)
        out1 = Dropout(0.5)(out1)
        if not isTwoOutputs:

            merge_two = concatenate([out1, out2])

            merge_two = Dense(NUM_CLASSES, activation='softmax')(merge_two)
            model2 = Model(inputs=[inp], outputs=merge_two)
            model2.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        else:
            merge_1 = Dense(NUM_CLASSES, activation='softmax', name='out_layer1')(out1)
            merge_2 = Dense(NUM_CLASSES, activation='softmax', name='out_layer2')(out2)

            # model trained on both outputs
            model2 = Model(input=[inp], output=[merge_1, merge_2])
            model2.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
                           optimizer=sgd, #'adam',
                           loss_weights=None,
                           metrics=['accuracy'])
    else:
        merge_two = out2 #concatenate([out2])
        merge_two = Dense(NUM_CLASSES, activation='softmax')(merge_two)
        model2 = Model(inputs=[inp], outputs=merge_two)
        model2.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
    return model2

def cnn_model(IMG_SIZE, NUM_CLASSES, sgd):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))



    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    #128 relu, 128-relu, pool-2x2, dropout(0.2)

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model

def lr_schedule(epoch, lr):
    return lr*(0.1**int(epoch/10))

def run_train_test(XX, YY, model, X_test, y_test, batch_size, nb_epoch, h5name='model.h5'):
    hstry = model.fit(XX, YY,
              verbose=2,  # 2-only epochs 1-batches
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.2,
              shuffle=True,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint(h5name, save_best_only=True)])

    # serialize model to JSON
    model_json = model.to_json()
    with open(h5name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5name)
    print("Saved model to disk - ", h5name)
    print("***************************")

    ntest = min(1000, y_test.shape[0])

    #y_pred = model.predict_classes(X_test[:ntest])#sequential
    y_pred = model.predict(X_test[:ntest])  # sequential
    if y_pred.__class__ == list:
        y_pred = y_pred[1]
    y_pred = np.argmax(y_pred, axis=1)
    acc = np.sum(y_pred[:ntest] == y_test[:ntest]) / np.size(y_pred)
    print("Test accuracy = {}".format(acc))
    print("***************************")
    return hstry

def read_model(X_test, y_test, ntest, modelName_h5 = "modelTT.h5", modelName_json = "modelTT.h5json"):
    # load json and create model
    json_file = open(modelName_json , 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(modelName_h5)
    print("Loaded model from disk")

    y_pred = loaded_model.predict(X_test[:ntest])  # sequential
    if y_pred.__class__ == list:
        y_pred = y_pred[1]
    y_pred = np.argmax(y_pred, axis=1)
    acc = np.sum(y_pred[:ntest] == y_test[:ntest]) / np.size(y_pred)
    return loaded_model, acc
