import pandas as pd
import logging
from sklearn import preprocessing
from sklearn.utils import shuffle
import copy
import pickle
from textdistance import levenshtein
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import random
from multiprocessing import Pool
import gc
from metrics.settings import MAX_TRAIN, TRAIN_VALID_SPLIT, NUM_RANDOM_BLOCKS_SELECT, LABELS_NOISE_NUM, \
    LABELS_SELECT_EACH_SIDE, LABELS_MIN_THRESHOLD, LABELS_MAX_THRESHOLD, LABELS_WEIGHT, N_ESTIMATORS, \
    MAX_DEPTH, LABELS_STRATEGY, LABELS_POSITIVE_MAX_DIST,  LABELS_NEGATIVE_MIN_DIST, PROCESS_PARALLEL, \
    PARALLEL_NUM_CONCUR, NUM_CPU_THREADS, PATH,  PLOT_DEBUG_FLAG, LABELS_COLS_STRINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


def predict_proba(clf, X_train, cols_pred):
    preds_train = clf.predict_proba(X_train[cols_pred])
    if preds_train.shape[1] == 1:  # all '1' , 0 dropped
        preds_train = preds_train
    else:
        preds_train = preds_train[:, 1]
    return preds_train


def create_clf(total_df_k_k_list):
    df_true, df_false, df_unlabeled = create_training_from_blocks(
        total_df_k_k_list, num_random_blocks=NUM_RANDOM_BLOCKS_SELECT
    )
    y_labels_arr = np.array([True] * len(df_true) + [False] * len(df_false))
    x_labels_arr = pd.concat(objs=[df_true, df_false], axis=0)
    assert len(df_true) > 0 and len(df_false) > 0
    logger.info(f" Labels True ={len(df_true)} False={len(df_false)}")
    logger.info(f" UNLabeled ={len(df_unlabeled)}")
    # logger.info(f" Example 5-positives \n{X_Labeled.head(5)}")

    # Train Loop
    x_labels_arr, y_labels_arr = shuffle(x_labels_arr, y_labels_arr)

    # Columns: scaled predictors and manual labeling
    cols_pred = [col for col in x_labels_arr.columns if col.startswith("f_")]
    # cols_view  [col for col in X_Labeled.columns if not col.startswith("f_")]
    cols_view = ["name_x", "name_y", "platform_x", "platform_y", "dist"]
    logger.info(f" Features {len(cols_pred)} scaled = {cols_pred}")
    logger.info(f" Orig columns (non-scaled) = {cols_view}")

    # Split: Train, Valid, Unlabeled
    indices = np.random.permutation(len(x_labels_arr))
    train_n = min(MAX_TRAIN, int(len(x_labels_arr) * TRAIN_VALID_SPLIT))
    idx1, idx2 = indices[:train_n], indices[train_n:]
    x_train_arr, y_train_arr = copy.copy(x_labels_arr.iloc[idx1, :]), y_labels_arr[idx1]
    x_valid_arr, y_valid_arr = copy.copy(x_labels_arr.iloc[idx2, :]), y_labels_arr[idx2]

    # Scale by Train
    scaler = preprocessing.RobustScaler().fit(x_train_arr[cols_pred])

    x_train_arr.loc[:, cols_pred] = scaler.transform(x_train_arr[cols_pred])
    x_valid_arr.loc[:, cols_pred] = scaler.transform(x_valid_arr[cols_pred])
    df_unlabeled.loc[:, cols_pred] = scaler.transform(df_unlabeled[cols_pred])

    assert x_train_arr.isna().sum().sum() == 0
    # Manually labeled will have higher weights
    weights = np.array([1] * len(x_train_arr))
    new_labels = "continue"
    pred_thresh_list = [0]
    iter_manual = 0
    clf = None
    while new_labels != "":
        iter_manual += 1
        # manual labels
        logger.info(
            f"Positive labels (DUPLICATES): train-valid={np.sum(y_train_arr)},{np.sum(y_valid_arr)}"
        )
        logger.info(
            f"Lengths Train={len(x_train_arr)} X_valid={len(x_valid_arr)} Unlabeled={len(df_unlabeled)}"
        )

        clf, preds_train, preds_valid, preds_unlabeled, s1 = create_classifier_preds(
            cols_pred, x_train_arr, y_train_arr, x_valid_arr, y_valid_arr, weights, df_unlabeled
        )
        logger.info(f"{s1}")
        logger.info(f"\n\nManual label iteration = {iter_manual}\n\n")
        # Entropy
        x_train_arr["odds_5050"] = -0.5 + preds_train
        x_valid_arr["odds_5050"] = -0.5 + preds_valid
        predictions_05_level = -0.5 + preds_unlabeled
        df_unlabeled["odds_5050"] = predictions_05_level
        # Manual labeling: select 1 data point from each side
        logger.info(f"STOPPING CRITERIA\n")
        logger.info(
            f" LABELED   DUPLICATES: MEAN  = {np.mean(preds_train[preds_train>0.5]):.2f}"
        )
        logger.info(
            f" UNLABELED DUPLICATES: MEAN  = {np.mean(preds_unlabeled[preds_unlabeled>0.5]):.2f} - QUIT WHEN NEAR 1.0"
        )
        if LABELS_STRATEGY == "low_noise":
            ind = np.argpartition(np.abs(predictions_05_level), LABELS_NOISE_NUM)[:LABELS_NOISE_NUM]
            pred_thresh = max(np.abs(predictions_05_level)[ind[:LABELS_NOISE_NUM]])
            pred_thresh_list.append(pred_thresh)
            df_update = df_unlabeled[predictions_05_level < pred_thresh]
            df_unlabeled = df_unlabeled[predictions_05_level > pred_thresh]
        elif LABELS_STRATEGY == "high_noise":
            df_update, df_unlabeled = active_strategy(predictions_05_level, df_unlabeled)

        new_arr_y = None
        while new_arr_y is None or not ((len(new_arr_y) == len(df_update)) or new_labels == ""):
            logger.info(f"\n{df_update[cols_view + ['odds_5050']]}")
            logger.info(
                f"Please input {len(df_update)} labels 1(DUPPLICATED) or 0, comma separated. Example: 1,0  ENTER to quit:\n"
            )
            new_labels = input()
            try:
                new_arr_y = np.array([np.bool_(np.int_(x)) for x in new_labels.split(",")])
            except Exception as e:
                logger.info(f" *** END MANUAL ENTER")
                new_arr_y = []
        if new_arr_y is not None and len(new_arr_y) > 0:
            create_plots(
                preds_train, preds_valid, preds_unlabeled, y_train_arr, y_valid_arr,
            )
            logger.info(f"new_Y = {new_arr_y}")
            weights = np.concatenate((weights, np.array([LABELS_WEIGHT] * len(new_arr_y))))
            y_train_arr = np.concatenate((y_train_arr, new_arr_y))
            x_train_arr = pd.concat(objs=[x_train_arr, df_update], axis=0)
    return clf, scaler, cols_pred, cols_view


def self_merge(df_in: pd.DataFrame):
    """ self merge and return upper triangle """
    df_in["_idx"] = range(1, len(df_in) + 1)
    df_in["key"] = 0
    df_out = df_in.merge(df_in, how="left", on="key")
    df_out = df_out[df_out["_idx_x"] < df_out["_idx_y"]]
    if len(df_out) == 0:
        return None

    #df_out.drop(["key", "_idx_x", "_idx_y"], 1, inplace=True)
    new_cols = df_out.columns.values.tolist()
    new_cols.remove("key")
    new_cols.remove("_idx_x")
    new_cols.remove("_idx_y")
    df_out = df_out[new_cols]
    assert 2 * len(df_out) == len(df_in) * (len(df_in) - 1)
    return df_out


def create_features(df_in: pd.DataFrame, cols_str: str):
    """features names start cf_  """
    df_in[["active_x", "active_y"]] = df_in[["active_x", "active_y"]].astype(bool)
    df_in["f_active"] = df_in["active_x"] == df_in["active_y"]
    df_in["f_platform"] = df_in["platform_x"] == df_in["platform_y"]
    df_in[f"f_1st_word"] = df_in.apply(
        lambda x: (
            (x["standardized_name_x"].split("_")[0] == x["standardized_name_y"].split("_")[0])
            & (x["standardized_name_x"].split("_")[0] != 'the')
        ),
        axis=1,
    )
    for col in cols_str:
        if col != "active" and col != "platform":
            df_in[f"f_{col}_len"] = df_in.apply(
                lambda x: 1 + max(len(x[col + "_x"]), len(x[col + "_y"])), axis=1
            )
            df_in[f"f_{col}_dist_int"] = df_in.apply(
                lambda x: levenshtein.distance(x[col + "_x"], x[col + "_y"]), axis=1
            )
            df_in[f"f_{col}_dist_rel"] = df_in[f"f_{col}_dist_int"] / df_in[f"f_{col}_len"]
    df_in["f_latitude"] = 0.5 * (df_in["latitude_x"] + df_in["latitude_y"])
    df_in["f_longitude"] = 0.5 * (df_in["longitude_x"] + df_in["longitude_y"])
    df_in["f_dist_lat"] = (df_in["latitude_x"] - df_in["latitude_y"]).abs()
    df_in["f_dist_long"] = (df_in["longitude_x"] - df_in["longitude_y"]).abs()
    df_in["f_dist_long"] = df_in["f_dist_long"] * np.abs(
        1e-4 + np.cos(np.pi / 180 * df_in["f_latitude"]) / 360
    )
    df_in["f_dist"] = (df_in["f_dist_long"] * df_in["f_dist_long"]) + (
        df_in["f_dist_lat"] * df_in["f_dist_lat"]
    )
    df_in["f_dist"] = np.sqrt(df_in["f_dist"])
    cols_drop = ["f_dist_lat", "f_dist_long", "f_latitude", "f_longitude"]
    cols_drop += ["latitude_x", "latitude_y", "longitude_x", "longitude_y"]
    df_in.drop(cols_drop, axis=1, inplace=True)
    # NON-SCALE columns for MANUAL LABELING
    df_in["dist"] = df_in["f_dist"]
    df_in["name_dist_int"] = df_in["f_name_dist_int"]
    return df_in


def create_train_rows(
    df, cols_str=LABELS_COLS_STRINGS, cols_num=["latitude", "longitude", "idx"],
):
    df_m = self_merge(df[cols_str + cols_num])
    if df_m is None:
        return None
    # create features from distances
    new_cols = []
    for col in cols_str + cols_num:
        new_cols += [col + "_x", col + "_y"]
    df_m = df_m[new_cols]
    df_m = create_features(df_m, cols_str=cols_str)
    return df_m


def create_classifier_preds(
    cols_pred, X_train, Y_train, X_valid, Y_valid, weights, df_Unlabeled
):

    # clf = LogisticRegressionCV(class_weight = "balanced", max_iter=500).fit(X_train[cols_pred], Y_train, sample_weight=weights)

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,  # 100 default
        class_weight="balanced",
        max_depth=MAX_DEPTH,
        n_jobs=-1,
    ).fit(X_train[cols_pred], Y_train, sample_weight=weights)

    preds_train = predict_proba(clf, X_train, cols_pred)
    preds_valid = predict_proba(clf, X_valid, cols_pred)
    preds_unlabeled = predict_proba(clf, df_Unlabeled, cols_pred)
    s1 = "auc: "
    try:
        s1 += f" train = {roc_auc_score(y_true=Y_train, y_score=preds_train):.2f} "
        s1 += f" valid = {roc_auc_score(y_true=Y_valid, y_score=preds_valid):.2f} "
    except Exception as e:
        logger.error(f" *** {e}")
    return clf, preds_train, preds_valid, preds_unlabeled, s1


def create_plots(
    preds_train, preds_valid, preds_unlabeled, Y_train, Y_valid,
):
    """ plot hist probs """
    pos_train, neg_train = preds_train[Y_train == True], preds_train[Y_train == False]
    pos_valid, neg_valid = preds_valid[Y_valid == True], preds_valid[Y_valid == False]
    neg_unlabeled, pos_unlabeled = (
        preds_unlabeled[preds_unlabeled < 0.5],
        preds_unlabeled[preds_unlabeled > 0.5],
    )

    if PLOT_DEBUG_FLAG:
        plt.close("all")
        ifig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), dpi=100)
        plt.subplot(2, 1, 1)
        plt.hist(pos_train, bins=50)
        plt.hist(pos_valid, bins=50)
        plt.title(
            f"DUBPLICATES train-valid-unlabeled: means={np.mean(pos_train):.2f}, {np.mean(pos_valid):.2f}"
        )
        plt.subplot(2, 1, 2)
        s = f"NON-DUPLICATES train-valid margin: means={np.mean(neg_train):.2f}, {np.mean(neg_valid):.2f}"
        plt.hist(neg_train, bins=10)
        plt.hist(neg_valid, bins=10)
        plt.title(s)
        s = f"DUBPLICATES  unlabeled mean={np.mean(pos_unlabeled):.2f}"
        plt.hist(pos_unlabeled, bins=50)
        plt.title(s)
        plt.show()


def active_strategy(predictions_level05, df_Unlabeled):
    pos_ind, neg_ind = [], []
    pos_mask = (predictions_level05 > LABELS_MIN_THRESHOLD) & (predictions_level05 < LABELS_MAX_THRESHOLD)
    neg_mask = (predictions_level05 > -LABELS_MAX_THRESHOLD) & (predictions_level05 < -LABELS_MIN_THRESHOLD)
    pos_ind = np.where(pos_mask)[0]
    neg_ind = np.where(neg_mask)[0]
    if len(list(pos_ind)) >= LABELS_SELECT_EACH_SIDE:
        pos_ind = list(random.sample(list(pos_ind), LABELS_SELECT_EACH_SIDE))
    else:
        pos_ind = [np.argmin(pos_mask)]
    if len(list(neg_ind)) >= LABELS_SELECT_EACH_SIDE:
        neg_ind = list(random.sample(list(neg_ind), LABELS_SELECT_EACH_SIDE))
    else:
        neg_ind = [np.argmax(pos_mask)]
    kep_rows = set(range(len(predictions_level05))).difference(set(list(pos_ind) + list(neg_ind)))
    df_update = df_Unlabeled.iloc[list(pos_ind) + list(neg_ind)]
    df_unlabl = df_Unlabeled.iloc[list(kep_rows)]
    assert len(df_unlabl) + len(df_update) == len(df_Unlabeled)
    return df_update, df_unlabl


def process_block(ind_block):
    """ list of blocks as DataFrames"""
    ind = ind_block[0]
    logger.info(f"parallel process {ind} (chunks of {PARALLEL_NUM_CONCUR})")
    df_k_li = ind_block[1]
    return [create_train_rows(d) for d in df_k_li if d is not None]


def run_parallel(blocks):
    blocks_li_li = [blocks[i:i + PARALLEL_NUM_CONCUR] for i in range(0, len(blocks), PARALLEL_NUM_CONCUR)]
    logger.info(f"create multiprocessing Pool: *** {len(blocks_li_li)} *** parallel block lists")
    pool = Pool(processes=NUM_CPU_THREADS)
    df_k_k_li_li = pool.map(process_block, zip(range(len(blocks_li_li)), blocks_li_li))
    df_k_k_list = [item for sublist in df_k_k_li_li for item in sublist if item is not None]
    logger.info(f"multiprocessing Pool finished blocks={len(df_k_k_list)} from {len(df_k_k_li_li)} processes")
    return df_k_k_list


def create_training_from_blocks(
    total_df_k_k_list, num_random_blocks: int = -1, min_name_dist: int = 2
):
    """ Create x3 Data sets and Dx x Dk Train for True, Train for False, Unlabeled"""
    if num_random_blocks > 0:
        blocks = random.sample(total_df_k_k_list, num_random_blocks)
    else:
        blocks = total_df_k_k_list

    df_k_k = None
    if PROCESS_PARALLEL:
        # Create a multiprocessing Pool
        df_k_k_list = run_parallel(blocks)
        for df_k_k_0 in df_k_k_list:
            if df_k_k_0 is not None and len(df_k_k_0) > 1:
                if df_k_k is None:
                    df_k_k = df_k_k_0
                else:
                    df_k_k = pd.concat(objs=[df_k_k, df_k_k_0], axis=0)
        logger.info(f"df_k_k={len(df_k_k)}")
    else:
        for ind, df_k in enumerate(blocks):
            if ind % int(1 + 0.01*len(blocks)) == 0:
                logger.info(
                    f"block={ind} rows={len(df_k)} progress={int(100*ind/len(blocks)):.2f} %"
                )
            df_k_k_0 = create_train_rows(df_k)
            if df_k_k_0 is not None and df_k_k_0.shape[0] > 1:
                if df_k_k is None:
                    df_k_k = df_k_k_0
                else:
                    df_k_k = pd.concat(objs=[df_k_k, df_k_k_0], axis=0)
    n = gc.collect()
    logger.info(
        f" collect: Unreachable objects collected by GC:{n}. uncollectable garbage {gc.garbage}"
    )
    logger.info(f"create simple labels")
    df_true, df_false, df_unlabeled = None, None, df_k_k
    if min_name_dist > 0:
        # DUPLICATES
        mask_true = (df_k_k["name_dist_int"] < min_name_dist) & (
            df_k_k["dist"] < LABELS_POSITIVE_MAX_DIST
        )
        # NON-DUPLICATES

        mask_false = df_k_k["dist"] > LABELS_NEGATIVE_MIN_DIST
        df_true, df_false, df_unlabeled = (
            df_k_k[mask_true],
            df_k_k[mask_false],
            df_k_k[~mask_true & ~mask_false],
        )
    return df_true, df_false, df_unlabeled


if __name__ == "__main__":
    try:
        logger.info(f"\n... read blocks all records\n")
        pickle_off = open(f"{PATH}/blocks.pickle", "rb")
    except Exception as e:
        logger.info(f"\n... blocks.pickle does not exist ({e}), read blocks small subset 20K records\n")
        pickle_off = open(f"{PATH}/blocks_small.pickle", "rb")

    total_df_k_k_list, df_all, df_anom = pickle.load(pickle_off)

    # Create clf
    clf, scaler, cols_pred, cols_view = create_clf(total_df_k_k_list)
    pickling_on = open(f"{PATH}/clf.pickle", "wb")
    pickle.dump([clf, scaler, cols_pred, cols_view], pickling_on)
    pickling_on.close()
