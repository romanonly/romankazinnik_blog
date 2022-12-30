import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def process(url: str, num_samples: int, output_model: str, output_history: str):
    # Use pandas excel reader
    df = pd.read_excel(url, engine = 'openpyxl')
    df = df.sample(frac = 1).reset_index(drop = True)
    df.drop('Unnamed: 10', axis = 1, inplace = True)
    df.drop('Unnamed: 11', axis = 1, inplace = True)
    if num_samples > 0:
        df = df.sample(n = min(len(df), num_samples))

    df = df.apply(pd.to_numeric, errors = 'coerce')
    df = df.dropna()

    train, test = train_test_split(df, test_size = 0.2)

    def format_output(data):
        y1 = data.pop('Y1')
        y1 = np.array(y1)
        y2 = data.pop('Y2')
        y2 = np.array(y2)
        return y1, y2

    def norm(x, train_stats):
        return (x - train_stats['mean']) / train_stats['std']

    train_stats = train.describe()

    # Get Y1 and Y2 as the 2 outputs and format them as np arrays
    train_stats.pop('Y1')
    train_stats.pop('Y2')
    train_stats = train_stats.transpose()
    print(train_stats)
    train_Y = format_output(train)
    test_Y = format_output(test)

    # Normalize the training and test data
    norm_train_X = norm(train, train_stats)
    norm_test_X = norm(test, train_stats)

    norm_train_X.to_csv("norm_train_X.csv", index = False)

    def model_builder(train_X):
        # Define model layers.
        input_layer = Input(shape = (len(train_X.columns),))
        first_dense = Dense(units = '128', activation = 'relu')(input_layer)
        second_dense = Dense(units = '128', activation = 'relu')(first_dense)

        # Y1 output will be fed directly from the second dense
        y1_output = Dense(units = '1', name = 'y1_output')(second_dense)
        third_dense = Dense(units = '64', activation = 'relu')(second_dense)

        # Y2 output will come via the third dense
        y2_output = Dense(units = '1', name = 'y2_output')(third_dense)

        # Define the model with the input layer and a list of output layers
        model = Model(inputs = input_layer, outputs = [y1_output, y2_output])

        print(model.summary())

        return model

    model = model_builder(norm_train_X)

    # Specify the optimizer, and compile the model with loss functions for both outputs
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
    model.compile(optimizer = optimizer, loss = {'y1_output': 'mse', 'y2_output': 'mse'},
                  metrics = {'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                             'y2_output': tf.keras.metrics.RootMeanSquaredError()})
    history = model.fit(norm_train_X, train_Y, epochs = 100, batch_size = 10)
    model.save(output_model)

    with open(output_history, "wb") as file:
        pickle.dump(history.history, file)

    model = tf.keras.models.load_model(output_model)

    # Test the model and print loss and mse for both outputs
    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x = norm_test_X, y = test_Y)
    print("\n\nLoss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss,
                                                                                       Y2_rmse))

    return


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
num_samples = -1
process(url, num_samples, output_model = "model_two_head", output_history = "history.pickle")
