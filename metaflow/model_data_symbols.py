import pandas as pd
import copy
from sklearn import preprocessing


# time series columns: data, P1, P2, ,,,. P15, P1_sign, P1_1dayahead, P1_2daysahead
def time_series_csv_to_lstm_input(
    nt_backward=10, target="P1_sign", fn="data/symbols_pp.csv", top_features=None
):
    """create time-series sequences Nt_backword long, labels for list_portfolios using top_features"""
    d_pp = pd.read_csv(fn)

    if top_features is None:
        top_features = d_pp.columns[1:-3]

    # for each portfolio create list of sequences, labels
    # num_times_backward
    dim_Xt = len(top_features)

    # y-axis: time and x-axis: top-features

    y = d_pp[target]

    d0 = copy.copy(d_pp[top_features])
    print(" shape=", d0.shape, " nan rows=", d0[d0.isnull().any(axis=1)])
    d_top_features = d0.dropna(thresh=1)
    nt = d_top_features.shape[0]
    print(
        " shape=",
        d_top_features.shape,
        " nan rows=",
        d_top_features[d_top_features.isnull().any(axis=1)],
    )

    x_scaled = preprocessing.scale(d_top_features._values)
    print("preprocessing.scale")
    print(
        "\n old mean=",
        d_top_features.values.mean(axis=0),
        "\n\n old std=",
        d_top_features.values.std(axis=0),
    )
    print("\n new mean=", x_scaled.mean(axis=0), "\n\n new std=", x_scaled.std(axis=0))
    mat_features = x_scaled  # d_pp[top_features].values
    assert mat_features.shape == (nt, dim_Xt)
    return mat_features, y


def train_test_time_series(mat_features, y, nt_backward=10):
    y_label = []
    x_train = []

    nt = mat_features.shape[0]
    # create Nt-long sequences from matrix, here i-axis is time
    for indt in range(nt_backward, nt):
        y_label.append(y[indt])
        x = mat_features[indt - nt_backward : indt, :]
        x_train.append(x)
    return x_train, y_label

