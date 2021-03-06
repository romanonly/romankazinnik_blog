from tensorflow.keras.models import load_model
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from model_data import create_train_labels
from model_symbols import train_lstm, plot_results, print_validate
from model_data_symbols import time_series_csv_to_lstm_input, train_test_time_series

if __name__ == "__main__":

    # Hyperparameters
    model_name = "data/best_model_1_PP_001.h5"
    y_name = "y_1day_future_price_change_pct"
    # x5  # length of sequences in LSTM
    Nt_backward = 20
    # 5  # 10  # number of features for each x_t datapoint in LSTM
    Num_features = 5
    # 1-1000 training for portfolios not independently may improve prediction
    Num_portfolios = 1
    # remove high correlated features
    Threshhold_cos = 0.95
    # 0.01  # regularisation
    regul_eps = 0.02  # 0.02
    # lstm neaurons, can be also X.shape[2]
    N1_LSTM = 5  # 10
    N2_LSTM = math.floor(N1_LSTM / 2)
    patience = 200
    epochs = 50  # 3000
    # validate for latest time-series points
    n_test_ratio = 0.20

    is_use_features_not_symbols = False
    # Data
    if is_use_features_not_symbols:
        x_train, y_label, list_portfolios = create_train_labels(
            y_name, Num_portfolios, Threshhold_cos, Num_features, Nt_backward
        )
    else:
        mat_features, y = time_series_csv_to_lstm_input(nt_backward=Nt_backward)

        x_train, y_label = train_test_time_series(mat_features, y, Nt_backward)

        list_portfolios = []
        Num_features = x_train[0].shape[1]  # use all p-symbols

    dim_x = x_train[0].shape[1]

    print(
        " num samples, nut_times_back, num_feaures_at_each_time=",
        x_train.__len__(),
        x_train[0].shape,
    )
    assert x_train[0].shape[0] == Nt_backward
    assert dim_x == Num_features + len(list_portfolios) or dim_x == Num_features

    X_train1, y_train, X_test1, y_test, history, train_auc, test_auc = train_lstm(
        x_train,
        y_label,
        n_test_ratio,
        regul_eps,
        epochs,
        model_name,
        N1_LSTM,
        N2_LSTM,
        patience,
    )

    # plot training history
    plot_results(history)

    # load saved model
    saved_model = load_model(model_name)  # "best_model.h5")

    # evaluate the model
    train_auc, test_auc = print_validate(
        saved_model, X_train1, y_train, X_test1, y_test
    )
