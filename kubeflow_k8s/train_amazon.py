import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


def load_dataset(file_path, num_samples=-1):
    df = pd.read_csv(file_path, usecols = [6, 9], nrows = num_samples)
    if num_samples > 1:
        df = df.sample(n = num_samples)
    df.columns = ['rating', 'title']

    text = df['title'].tolist()
    text = [str(t).encode('ascii', 'replace') for t in text]
    text = np.array(text, dtype = object)[:]

    labels = df['rating'].tolist()
    labels = [1 if i >= 4 else 0 if i == 3 else -1 for i in labels]
    labels = np.array(pd.get_dummies(labels), dtype = int)[:]

    return labels, text


def get_model():
    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape = [128],
        input_shape = [], dtype = tf.string, name = 'input', trainable = False
    )

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(3, activation = 'softmax', name = 'output'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)  # compare to SGD
    model.compile(
        optimizer = optimizer,  # 'Adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    model.summary()
    return model


def train(NUM_SAMPLES=1000, EPOCHS=5, BATCH_SIZE=32, TRAIN_FILE='train.csv', VAL_FILE='test.csv'):
    WORKING_DIR = os.getcwd()  # use to specify model checkpoint path
    print("Loading training/validation data ...")
    y_val, x_val = load_dataset(VAL_FILE, num_samples = NUM_SAMPLES)
    y_train, x_train = load_dataset(TRAIN_FILE, num_samples = NUM_SAMPLES)

    print("Training the model ...")

    model = get_model()
    model.fit(
        x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1,
        validation_split = 0.2,  # validation_data = (x_val, y_val),
        callbacks = [tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                WORKING_DIR,
                'model_checkpoint'
            ),
            monitor = 'val_loss', verbose = 1,
            save_best_only = True,
            save_weights_only = False,
            mode = 'auto'
        )]
    )
    # Evaluate the model
    loss, acc = model.evaluate(x_val, y_val, verbose = 2)
    print(loss, acc)
    y_val_pred = model.predict(x_val)
    print(y_val_pred)
    return model


def export_model(model, base_path="amazon_review/"):
    path = os.path.join(base_path, str(int(time.time())))
    tf.saved_model.save(model, path)


if __name__ == '__main__':
    model = train(
        TRAIN_FILE = "https://www.dropbox.com/s/tdsek2g4jwfoy8q/train.csv?dl=1",
        VAL_FILE = "https://www.dropbox.com/s/tdsek2g4jwfoy8q/test.csv?dl=1",
    )
    export_model(model)
