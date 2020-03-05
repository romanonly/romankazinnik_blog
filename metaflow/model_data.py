import _pickle as cPickle
import numpy as np
import pandas as pd
import copy

from sklearn import preprocessing


def vec_cos(x, y):
    """ cosine similarity vectors c and y"""

    def vec_norm(dYP1):
        YP1norm2 = np.sqrt((dYP1 * dYP1).sum())
        return dYP1 / YP1norm2, YP1norm2

    xn, _ = vec_norm(x)
    yn, _ = vec_norm(y)
    return (xn * yn).sum()


def select_top_features(d, f_names, ybin, list_portfolios, symbols, Num_features):
    """ select top Num_features features using cosine-correlation"""

    def top_features_from_dic(dic_feat_scores):
        """return sorted features list based on its dictionary values"""
        farr_sort1 = np.sort(list(dic_feat_scores.items()), axis=0)
        return [f_cos[0] for f_cos in farr_sort1[-Num_features:]]

    dic_feat_cos = {}
    dic_feat_scores = {}
    for f in f_names:
        dic_feat_scores[f] = 0

    # For each portfolio, for each feature compute cosine-correlation with target
    for P1 in list_portfolios:

        yP1 = ybin[symbols == P1].astype(float)
        # N = yP1.size
        farr = []
        for f in f_names:
            xP1 = d[f].values[symbols == P1]
            cos_xy = vec_cos(xP1, yP1)
            farr.append((f, cos_xy))
            dic_feat_cos[f] = dic_feat_cos.get(f, 0) + cos_xy

        farr_sort = np.sort(farr, axis=0)
        top_features = [f_cos[0] for f_cos in farr_sort[-Num_features:]]
        for f in top_features:
            dic_feat_scores[f] = dic_feat_scores[f] + 1

    # Select features based on (1) ranking sorted by cosine-correlations and
    # (2) sum of its cosine correlations with target, per each portfolio
    top_features1 = top_features_from_dic(dic_feat_scores)
    top_features2 = top_features_from_dic(dic_feat_cos)
    print(
        "select top features sum-cos vs ranking: 100perc same?",
        100 * len(set(top_features1).intersection(set(top_features2))) / Num_features,
    )
    top_features = list(set(top_features1).union(set(top_features2)))
    print(
        "union n_features= %d n_feats1=%d n_feats2=%d"
        % (len(top_features), len(top_features1), len(top_features2))
    )
    return top_features[:Num_features]


def train_labels(d, ybin, list_portfolios, symbols, top_features, Nt_backward):
    """create time-series sequences Nt_backword long, labels for list_portfolios using top_features"""
    y_label = []
    x_train = []
    one_hot_encod = np.ndarray(len(list_portfolios), dtype=float)

    # for each portfolio create list of sequences, labels
    for indP1, P1 in enumerate(list_portfolios):
        # each time sample to have one_hot
        one_hot_encod[:] = 0.0
        one_hot_encod[indP1] = 1.0

        yP1 = ybin[symbols == P1].astype(float)

        N = yP1.size
        Dim_Xt = len(top_features) + len(one_hot_encod)

        # y-axis: time and x-axis: top-features and one-hot Portfolio symbol
        mat_features = np.ndarray((N, Dim_Xt), dtype=float)

        if Dim_Xt > len(top_features):
            # each sequence will have features and one-hot encoded portfolio symbol
            Dim_Xt == len(top_features) + len(one_hot_encod)
            for k in range(N):
                mat_features[k, len(top_features) : Dim_Xt] = one_hot_encod[:]

        for indf, f in enumerate(top_features):
            mat_features[:, indf] = d[f].values[symbols == P1]
        # create Nt-long sequences from matrix, here i-axis is time
        for indt in range(Nt_backward, N):
            y_label.append(yP1[indt])
            x = mat_features[indt - Nt_backward : indt, :]
            x_train.append(x)

    return x_train, y_label


def read_data(is_use_pkl=False):
    """ read csv files, sort by datetime, rank features"""
    if is_use_pkl == False:
        d1 = pd.read_csv("interview_challenge.csv")
        d2 = pd.read_csv("interview_challenge_2.csv")

        # small
        d1["date"] = pd.to_datetime(d1.date)
        d1.sort_values(by=["date"], inplace=True, ascending=True)
        d1["month"] = d1["date"].map(lambda x: 12 * (x.year - 2015) + x.month)
        d1["year"] = d1["date"].map(lambda x: (x.year - 2015))

        # large
        d2["date"] = pd.to_datetime(d2.date)
        d2.sort_values(by=["date"], inplace=True, ascending=True)
        d2["month"] = d2["date"].map(lambda x: x.month)
        d2["year"] = d2["date"].map(lambda x: (x.year - 2015))

        # rank features with d1 or d2 (slower)
        d = d2

        # columns to feature names
        f_names = [
            f for f in d.columns if f[:2] != "y_" and f != "symbol" and f != "date"
        ]  # dont include 1st 'date' and last 'symbol'

        # Takes long time: select best features
        arr_f_maxcos = [(f_names[-1], 0.0)]
        for indf1, f1 in enumerate(f_names[:-1]):
            xP1 = d[f1].values
            max_cos_xy = max(
                [
                    vec_cos(xP1, d[f2].values)
                    for indf2, f2 in enumerate(f_names[indf1 + 1 :])
                ]
            )
            print(indf1, f1, max_cos_xy)
            arr_f_maxcos.append((f1, max_cos_xy))

        with open(".\interview_challenge_all.pkl", "wb") as file_obj:
            cPickle.dump([d1, d2, arr_f_maxcos, f_names], file_obj)

    objects = []
    with (open("interview_challenge_all.pkl", "rb")) as openfile:
        while True:
            try:
                objects.append(cPickle.load(openfile))
            except EOFError:
                break
    d1 = objects[0][0] # small
    d2 = objects[0][1]
    arr_f_maxcos = objects[0][2]
    f_names = objects[0][3]

    # Select csv input data: small d1 ot large d2
    return d1, arr_f_maxcos, f_names


def save_train_data_csv(
    d,
    top_f_names,
    ybin,
    list_portfolios,
    symbols,
    f_names,
    y_name,
    csv_num_PP=1,
    csv_num_features=10,
):
    csv_top_features = select_top_features(
        d, top_f_names, ybin, list_portfolios, symbols, Num_features=csv_num_features
    )
    csv_fname = "f" + str(csv_num_features) + "_PP" + str(csv_num_PP) + ".csv"

    # P1 = symbols[10]
    csv_d = d[f_names[:-2] + ["date", "symbol"] + [y_name]]
    csv_d2 = csv_d[csv_d["symbol"] == symbols[0]]
    csv_d3 = csv_d2.apply(
        lambda x: (x > 0.0).astype(int) if x.name in [y_name] else x, axis=0
    )
    print("csv_d3.shape=", csv_d3.shape)
    csv_d3.to_csv("sym1_187features.csv")

    fcsv_d = d[csv_top_features + ["date", "symbol"] + [y_name]]
    fcsv_d2 = fcsv_d[fcsv_d["symbol"].isin(list(np.unique(symbols)[-csv_num_PP:]))]
    fcsv_d3 = fcsv_d2.apply(
        lambda x: (x > 0.0).astype(int) if x.name in [y_name] else x, axis=0
    )
    print("csv_d3_top_features.shape=", fcsv_d3.shape)
    fcsv_d3.to_csv(csv_fname)

    d_date_pp = d[["date", "symbol", y_name]]
    # ddate = pd.unique(d_date_pp.date)
    pp = pd.unique(d_date_pp.symbol)
    # d3 = pd.DataFrame(columns=['date'] + list(pp))

    p0 = pp[0]
    d0 = fcsv_d[fcsv_d["symbol"].isin(list([p0]))][["date", "symbol", y_name]]
    d0.set_index(d0["date"], inplace=True)
    d0.rename(columns={y_name: p0}, inplace=True)
    d01 = d0[p0]

    for ipp in pp[1:]:
        p2 = ipp
        d1 = fcsv_d[fcsv_d["symbol"].isin(list([p2]))][["date", "symbol", y_name]]
        d1.set_index(d1["date"], inplace=True)
        d1.rename(columns={y_name: p2}, inplace=True)
        d01 = pd.merge(d01, d1[p2], how="inner", left_index=True, right_index=True)
    d01.sort_values(by=["date"], inplace=True, ascending=True)
    vals_p0 = list(d01["P1"].values) + [None, None]
    d01["P1_1d"] = vals_p0[1:-1]
    d01["P1_2d"] = vals_p0[2:]
    ybin = (d01["P1"].values > 0).astype(int)
    d01["P1_sign"] = ybin
    d01.to_csv("symbols_hiplot.csv")


def create_train_labels(
    y_name, Num_portfolios, Threshhold_cos, Num_features, Nt_backward
):
    d, arr_f_maxcos, f_names = read_data(is_use_pkl=True)
    print("input data d.shape =", d.shape)
    #
    symbols = d["symbol"].values  # np.unique(symbol)
    symbols_unique = np.unique(symbols).shape
    list_portfolios = np.unique(symbols)[-Num_portfolios:]
    print(" Modeling portfolios: ", list_portfolios)

    # Create target as sign
    yvals = d[y_name].values
    ybin = (yvals > 0).astype(int)
    print("ybin ", ybin.shape, "PP=", symbols_unique)
    print("f_names ", len(f_names))

    # features
    top_f_names = [
        f_cos[0] for f_cos in arr_f_maxcos if float(f_cos[1]) < Threshhold_cos
    ]
    # key:val pairs:
    # feature_name: feature-cosine-correlation index
    dic_f_maxcos = {}
    for f in arr_f_maxcos:
        dic_f_maxcos[f[0]] = f[1]

    top_features = select_top_features(
        d, top_f_names, ybin, list_portfolios, symbols, Num_features=Num_features
    )
    print(" top_features=", top_features)
    print([(f, dic_f_maxcos[f]) for f in top_features])

    if False:
        save_train_data_csv(
            d,
            top_f_names,
            ybin,
            list_portfolios,
            symbols,
            f_names,
            y_name,
            csv_num_PP=1,
            csv_num_features=10,
        )

    # scale input data
    d_top_features = copy.copy(d[top_features])
    X_scaled = preprocessing.scale(d_top_features._values)
    d_top_features.loc[:, :] = X_scaled[:, :]
    print("preprocessing.scale")
    print(X_scaled.mean(axis=0), X_scaled.std(axis=0))

    x_train, y_label = train_labels(
        d_top_features,
        ybin,
        list_portfolios,
        symbols,
        top_features,
        Nt_backward=Nt_backward,
    )

    return x_train, y_label, list_portfolios
