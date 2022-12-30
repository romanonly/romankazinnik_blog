import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


def load_dataset(
        url: str,
        num_samples: int,
        output_labels_artifacts: str,
        output_text_artifacts: str
):
    df = pd.read_csv(url, usecols = [6, 9])
    df.columns = ['rating', 'title']
    if num_samples > 1:
        df = df.sample(n = num_samples)

    text = df['title'].tolist()
    text = [str(t).encode('ascii', 'replace') for t in text]
    text = np.array(text, dtype = object)[:]

    labels = df['rating'].tolist()
    labels = [1 if i >= 4 else 0 if i == 3 else -1 for i in labels]
    labels = np.array(pd.get_dummies(labels), dtype = int)[:]

    with open(output_labels_artifacts, "wb") as file:
        pickle.dump(labels, file)

    with open(output_text_artifacts, "wb") as file:
        pickle.dump(text, file)

    # return labels, text


def get_model(output_untrained_model: str):
    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape = [128],
        input_shape = [], dtype = tf.string, name = 'input', trainable = False
    )

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(3, activation = 'softmax', name = 'output'))
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'Adam', metrics = ['accuracy']
    )
    model.summary()
    print("\n\nsave untrained_model.pickle\n\n")
    model.save(output_untrained_model)


def train(
        input_labels_artifacts: str,
        input_text_artifacts: str,
        input_untrained_model: str,
        output_model: str,
        output_history: str,
        EPOCHS=5,
        BATCH_SIZE=32,
):
    print("Training the model ...")

    with open(input_labels_artifacts, "rb") as file:
        y_train = pickle.load(file)

    with open(input_text_artifacts, "rb") as file:
        x_train = pickle.load(file)

    model = tf.keras.models.load_model(input_untrained_model)

    WORKING_DIR = os.getcwd()  # use to specify model checkpoint path
    history = model.fit(
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
    model.save(output_model)
    with open(output_history, "wb") as file:
        pickle.dump(history.history, file)


def eval_model(
        input_model: str,
        input_labels_artifacts: str,
        input_text_artifacts: str,
):
    with open(input_labels_artifacts, "rb") as file:
        y_eval = pickle.load(file)

    with open(input_text_artifacts, "rb") as file:
        x_eval = pickle.load(file)

    model = tf.keras.models.load_model(input_model)

    # Test the model and print loss and mse for both outputs
    loss, acc = model.evaluate(x = x_eval, y = y_eval)
    print("\n\n\n Loss = {}, Acc = {} ".format(loss, acc))


def export_model(model, base_path="kfp_amazon_review/"):
    path = os.path.join(base_path, str(int(time.time())))
    tf.saved_model.save(model, path)


if __name__ == '__main__':
    print("\n\nLoading training data ...")

    # y_val, x_val = 
    load_dataset(
        url = "https://www.dropbox.com/s/tdsek2g4jwfoy8q/train.csv?dl=1",
        num_samples = 1000,
        output_labels_artifacts = "labels_artifacts_train.pickle",
        output_text_artifacts = "text_artifacts_train.pickle"
    )

    get_model(output_untrained_model = "model_untrained.pickle")

    train(
        input_labels_artifacts = "labels_artifacts_train.pickle",
        input_text_artifacts = "text_artifacts_train.pickle",
        input_untrained_model = "model_untrained.pickle",
        output_model = "model.pickle",
        output_history = "history.pickle"
    )

    # export_model(model)

    print("\n\nLoading test data ...")

    load_dataset(
        url = "https://www.dropbox.com/s/tdsek2g4jwfoy8q/test.csv?dl=1",
        num_samples = 1000,
        output_labels_artifacts = "labels_artifacts_test.pickle",
        output_text_artifacts = "text_artifacts_test.pickle"
    )

    eval_model(
        input_model = "model.pickle",
        input_labels_artifacts = "labels_artifacts_test.pickle",
        input_text_artifacts = "text_artifacts_test.pickle"
    )
