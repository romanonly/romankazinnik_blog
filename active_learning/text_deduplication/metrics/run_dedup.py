import pandas as pd
import pickle
import logging
import numpy as np
import random
import gc
from multiprocessing import Pool
import warnings
from metrics.run_clf import predict_proba, create_train_rows
from metrics.settings import DEDUP_BATCH_SIZE, DEDUP_PROB_THRESHOLD, DEDUP_NUM_RANDOM_BLOCKS_SELECT, \
    DEDUP_BLOCKS_PER_PROCESS, PATH, NUM_CPU_THREADS


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'


pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


cols_save = [
    "idx_x",
    "idx_y",
    "name_x",
    "name_y",
    "platform_x",
    "platform_y",
    "dist",
    "probability_duplicate",
]


def write_duplicates(df_k_k_list: pd.DataFrame, fn1: str):
    try:
        df_duplicates = pd.DataFrame()
        df_unlabeled_concat = pd.DataFrame()
        # scale-score in large chunks
        for ind, df_unlabeled_ind in enumerate(df_k_k_list):
            if df_unlabeled_ind is not None and len(df_unlabeled_ind) > 0:
                df_unlabeled_concat = pd.concat(
                    objs=[df_unlabeled_concat, df_unlabeled_ind], axis=0
                )
            # Write Batch Inferences
            if ind == len(df_k_k_list) - 1 or len(df_unlabeled_concat) > DEDUP_BATCH_SIZE:
                # df_Unlabeled_concat has bool cols and df_a miust be float
                df_a = df_unlabeled_concat.copy()
                df_a[cols_pred] = df_a[cols_pred].astype(float)
                df_a.loc[:, cols_pred] = scaler.transform(df_unlabeled_concat[cols_pred])
                df_unlabeled_concat = pd.DataFrame()
                # Scoring and Threshold at 0.5
                pred_prob = predict_proba(clf, df_a, cols_pred)
                pos_mask = pred_prob > DEDUP_PROB_THRESHOLD
                pos_ind = np.where(pos_mask)[0]
                df_a["probability_duplicate"] = pred_prob
                df_pos = df_a.iloc[pos_ind]
                if len(df_pos) > 0:
                    df_pos.reset_index(drop=True, inplace=True)
                    df_duplicates = pd.concat(
                        objs=[df_duplicates, df_pos[cols_save]], axis=0
                    )
                    # logger.info(f"\n{df_pos[cols_save]}\n") # df_pos[cols_view]
                    df_duplicates.drop_duplicates(
                        ["idx_x", "idx_y"], keep="last", inplace=True
                    )
                    df_duplicates.to_csv(fn1)
        df_duplicates.sort_values(
            by=["probability_duplicate"], ascending=False, inplace=True
        )
        df_duplicates.reset_index(drop=True, inplace=True)
        df_duplicates.to_csv(fn1)
        logger.info(f"writing {fn1}")
    except Exception as e:
        logger.info(f"\nException occurred:{type(e)} (type), {e}\n")
    logger.info(f"{fn1} done\n")


def process_block(ind_block):
    """ list of blocks each block is DataFrame"""
    ind = ind_block[0]
    df_k_list = ind_block[1]
    logger.info(f"parallel process {ind} ({DEDUP_BLOCKS_PER_PROCESS} blocks)")
    df_k_k_list = [create_train_rows(d) for d in df_k_list if d is not None]
    write_duplicates(df_k_k_list, fn1=f"{PATH}/duplicates_big_{ind}.csv")
    return ind


def run_parallel(blocks):
    blocks_li_li = [
        blocks[i : i + DEDUP_BLOCKS_PER_PROCESS]
        for i in range(0, len(blocks), DEDUP_BLOCKS_PER_PROCESS)
    ]
    logger.info(f"multiprocessing Pool: *** {len(blocks_li_li)} *** parallel lists")
    pool = Pool(processes=NUM_CPU_THREADS)  # 8 cores/16
    df_k_k_li_li = pool.map(
        process_block, zip(range(len(blocks_li_li)), blocks_li_li)
    )
    # df_k_k_list = [i for sub in df_k_k_li_li for i in sub if item is not None]
    logger.info(f"multiprocessing Pool finished: {len(df_k_k_li_li)} processes")
    return df_k_k_li_li


if __name__ == "__main__":
    try:
        logger.info(f"\n...open blocks all records\n")
        pickle_off = open(f"{PATH}/blocks.pickle", "rb")
    except Exception as e:
        logger.info(f"\n.../blocks.pickle not exsit {e}, open blocks small subset 20K records\n")
        pickle_off = open(f"{PATH}/blocks_small.pickle", "rb")

    total_df_k_k_list, df_all, df_anom = pickle.load(pickle_off)
    logger.info(f"reading {PATH}/clf.pickle")
    pickle_off = open(f"{PATH}/clf.pickle", "rb")
    clf, scaler, cols_pred, cols_view = pickle.load(pickle_off)
    # create duplicates list
    if DEDUP_NUM_RANDOM_BLOCKS_SELECT > 0:
        blocks = random.sample(total_df_k_k_list, NUM_RANDOM_BLOCKS_SELECT)
    else:
        blocks = total_df_k_k_list
    n = gc.collect()
    logger.info(f"Unreachable objects collected by GC:{n}. un-collectable {gc.garbage}")
    _ = run_parallel(blocks)
    exit(1)
