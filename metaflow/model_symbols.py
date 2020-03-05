import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import LSTM, Activation
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from matplotlib import pyplot
import numpy as np

def auc(y_true, y_pred):
    auc = metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


#
# LSTM
#
# Train [training data, times-series, features]
def train_lstm(x_train, y_label, n_test_ratio, regul_eps, epochs, model_name, N1_LSTM, N2_LSTM, patience):
    # Hyperparameters

    # minimal number input data points for training larger LSTM model
    min_data_large_model = 20000

    X = np.array(x_train)
    y = np.array(y_label)

    # Simple split into train/test for final validation
    n_test = int(np.floor(n_test_ratio * X.shape[0]))
    X_train1, X_test1, y_train, y_test = (
        X[:-n_test],
        X[-n_test:],
        y[:-n_test],
        y[-n_test:],
    )
    print(X_train1.shape, X_test1.shape, y_train.shape, y_test.shape)

    # TF 2.0 model

    n_steps = X_train1.shape[1]  # time series steps
    n_x_dimension = X_train1.shape[2]  # features at each time step

    # LSTM model:
    model = Sequential()

    if X_train1.shape[0] > min_data_large_model:
        return_sequences = True
    else:
        return_sequences = False

    model.add(
        LSTM(
            N1_LSTM,
            # kernel_regularizer=l2(regul_eps), # not popule in lstm
            recurrent_regularizer=l2(regul_eps),
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(n_steps, n_x_dimension),
            return_sequences=return_sequences,
        )
    )

    if X_train1.shape[0] > min_data_large_model:
        model.add(
            LSTM(
                N2_LSTM,
                activation="relu",
                return_sequences=False,
                kernel_initializer="he_normal",
                # kernel_regularizer=l2(regul_eps),
                recurrent_regularizer=l2(regul_eps),
            )
        )

    model.add(
        Dense(
            N2_LSTM,
            use_bias=False,
            # activation="relu",
            # kernel_initializer="he_normal",
            # kernel_regularizer=l2(regul_eps),
        )
    )
    # model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["mae", tf.keras.metrics.AUC()],
    )

    # simple early stopping
    es1 = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)
    mc = ModelCheckpoint(
        model_name, monitor="val_loss", mode="min", verbose=1, save_best_only=True,
    )

    history = model.fit(
        X_train1,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=1, # 2,
        callbacks=[es1, mc],
        # validation_split=0.3,
        validation_data=(X_test1, y_test),
    )

    # evaluate the model
    mse, mae, auc = model.evaluate(X_test1, y_test, verbose=2)
    print(
        "X_test: MSE: %.3f, RMSE: %.3f, MAE: %.3f AUC: %.3f"
        % (mse, np.sqrt(mse), mae, auc)
    )

    # evaluate the model
    _, train_acc, train_auc = model.evaluate(X_train1, y_train, verbose=0)
    _, test_acc, test_auc = model.evaluate(X_test1, y_test, verbose=0)
    print(
        "Train: mae=%.3f auc=.%3f, Test: mae=%.3f auc=.%3f"
        % (train_acc, train_auc, test_acc, test_auc)
    )

    _, train_acc, train_auc = model.evaluate(X_train1, y_train, verbose=0)
    _, test_acc, test_auc = model.evaluate(X_test1, y_test, verbose=0)

    return X_train1, y_train, X_test1, y_test, history, train_auc, test_auc


def print_validate(model, X_train1, y_train, X_test1, y_test):
    _, train_acc, train_auc = model.evaluate(X_train1, y_train, verbose=0)
    _, test_acc, test_auc = model.evaluate(X_test1, y_test, verbose=0)
    print(
        "Train: mae=%.3f auc=.%3f, Test: mae=%.3f auc=.%3f"
        % (train_acc, train_auc, test_acc, test_auc)
    )

    # errors for train vs test
    def pred(X_train1, y_train, nn, thresh=0.5):
        yhat = model.predict(X_train1[0:nn])
        num_errors = sum(
            np.abs((((yhat > thresh).astype(int)).reshape(nn) - y_train[0:nn]))
        )
        num_goods = sum(
            0 == np.abs((((yhat > thresh).astype(int)).reshape(nn) - y_train[0:nn]))
        )
        print("Errors rate= ", num_errors / nn, " True pos rate=", num_goods / nn)

    # sample from ROC
    thresh = 0.6
    print("Train: example threshold=", thresh)
    pred(X_train1, y_train, y_train.shape[0], thresh)
    print("Test: example threshold=", thresh)
    pred(X_test1, y_test, y_test.shape[0], thresh)

    return train_auc, test_auc


def plot_results(h):
    pyplot.plot(h.history["loss"], label="train")
    pyplot.plot(h.history["val_loss"], label="test")
    pyplot.plot(h.history["auc"], label="AUC train")
    pyplot.legend()
    pyplot.show()
