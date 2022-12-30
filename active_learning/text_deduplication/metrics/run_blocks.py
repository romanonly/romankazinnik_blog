import logging
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics.settings import BLOCKS_NUMBER_LAT, BLOCKS_NUMBER_LON, BLOCKS_EPS, BLOCKS_MAX_ROWS, \
    BLOCKS_MAX_RECURSIVE_DEPTH, BLOCKS_NANS_DROP_COLS, BLOCKS_ANOMALY_STD_NUM, PATH, PATH_CSV, PLOT_DEBUG_FLAG

fn_csv = "css_public_all_ofos_locations.csv"

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

plt.close("all")


def read_data(fn: str, is_remove_id=True):
    df = pd.read_csv(f"{PATH_CSV}/{fn}", sep = "\x01", dtype = str, index_col = 0)
    df.reset_index(inplace = True)
    # missing values
    logger.info(f"null values:\n{df.isnull().sum()}")
    logger.info(f"Num rows={len(df)}")
    logger.info(f"dropping NaN cols={BLOCKS_NANS_DROP_COLS}")
    if not is_remove_id:
        BLOCKS_NANS_DROP_COLS.remove("restaurant_id")
    df = df.drop(BLOCKS_NANS_DROP_COLS, axis = 1)
    logger.info(f"dropping NaN rows")
    df["sub_platform"] = df["sub_platform"].fillna("")
    if not is_remove_id:
        logger.info(f"nan num df[restaurant_id] = {df['restaurant_id'].isnull().sum()}")
        df["restaurant_id"] = df["restaurant_id"].fillna(-1)
    df.dropna(inplace = True)
    logger.info(f"Num rows={len(df)}")

    cols_numeric = ["latitude", "longitude"]
    df["latitude"] = pd.to_numeric(df["latitude"], errors = "coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors = "coerce")
    df[cols_numeric] = df[cols_numeric].astype(float)
    cols_str = [
        "name",
        "standardized_name",
        "platform",
        "sub_platform",
        "country",
        "active",
    ]
    df[cols_str] = df[cols_str].astype("string")
    df["active"] = df["active"].astype(bool)
    logger.info(f"add idx - index column")
    df["idx"] = range(1, len(df) + 1)
    df["idx"] = df["idx"].astype(int)
    return df


def remove_anomalies(df_all):
    mask_anom = None
    for col in ["longitude", "latitude"]:
        min_c = df_all[col].mean() - BLOCKS_ANOMALY_STD_NUM * df_all[col].std()
        max_c = df_all[col].mean() + BLOCKS_ANOMALY_STD_NUM * df_all[col].std()
        mask_anom_0 = (df_all[col] < min_c) | (df_all[col] > max_c)
        if mask_anom is None:
            mask_anom = mask_anom_0
        else:
            mask_anom = mask_anom | mask_anom_0
    df_anom = df_all[mask_anom]
    df_non_anom = df_all[~mask_anom]
    assert len(df_all) == len(df_anom) + len(df_non_anom)
    return df_anom, df_non_anom


def recursive_split(
        df_recursive: pd.DataFrame,
        col1: str,
        x1: float,
        x2: float,
        col2: str,
        y1: float,
        y2: float,
        depth,
):
    """ df_recursive """
    df_k = df_recursive[
        (
                (
                        (df_recursive[col1] > -BLOCKS_EPS + x1)
                        & (df_recursive[col1] < BLOCKS_EPS + x2)
                )
                & (
                        (df_recursive[col2] > -BLOCKS_EPS + y1)
                        & (df_recursive[col2] < BLOCKS_EPS + y2)
                )
        )
    ]
    if len(df_k) < BLOCKS_MAX_ROWS or depth > BLOCKS_MAX_RECURSIVE_DEPTH:
        if df_k.shape[0] > 0:
            return [df_k]
        return []
    # split
    x1_2 = x1 + (x2 - x1) / 2
    y1_2 = y1 + (y2 - y1) / 2
    df_list_1 = recursive_split(df_recursive, col1, x1, x1_2, col2, y1, y1_2, depth + 1)
    df_list_2 = recursive_split(df_recursive, col1, x1, x1_2, col2, y1_2, y2, depth + 1)
    df_list_3 = recursive_split(df_recursive, col1, x1_2, x2, col2, y1, y1_2, depth + 1)
    df_list_4 = recursive_split(df_recursive, col1, x1_2, x2, col2, y1_2, y2, depth + 1)
    return df_list_1 + df_list_2 + df_list_3 + df_list_4


def create_blocks(df0):
    e = BLOCKS_EPS
    dlat_i = (df0["latitude"].max() - df0["latitude"].min() + e) / (
            BLOCKS_NUMBER_LAT - 1
    )
    lati_i = df0["latitude"].min() + dlat_i * range(BLOCKS_NUMBER_LAT)

    total_df_k_k_list = []
    max_rows = []
    for i in range(1, BLOCKS_NUMBER_LAT):
        if i % (1 + int(BLOCKS_NUMBER_LAT / 100)) == 0:
            logger.info(f"blocking processed {int(100 * i / BLOCKS_NUMBER_LAT)} %")
        df_i = df0[
            (df0["latitude"] > -e + lati_i[i - 1]) & (df0["latitude"] < e + lati_i[i])
            ]
        if df_i.shape[0] > 0:
            if df_i.shape[0] < BLOCKS_MAX_ROWS:
                total_df_k_k_list.append(df_i.copy())
                max_rows.append(len(df_i))
            else:
                # Length in meters of 1° of longitude = 40075 km * cos( latitude ) / 360
                # cos_lat = np.cos(np.pi*df_i["latitude"].mean()/180)
                long_dist = df_i["longitude"].max() - df_i["longitude"].min()
                dlon_i = (long_dist + e) / (BLOCKS_NUMBER_LON - 1)
                long_i = df_i["longitude"].min() + dlon_i * range(BLOCKS_NUMBER_LON)
                for k in range(1, BLOCKS_NUMBER_LON):
                    df_recursive = df_i[
                        (df_i["longitude"] > -e + long_i[k - 1])
                        & (df_i["longitude"] < e + long_i[k])
                        ]
                    df_k_k_list = recursive_split(
                        df_recursive,
                        "longitude",
                        long_i[k - 1],
                        long_i[k],
                        "latitude",
                        lati_i[i - 1],
                        lati_i[i],
                        depth = 0,
                    )
                    if len(df_k_k_list) > 0:
                        all_blocks = [dff.shape[0] for dff in df_k_k_list]
                        total_df_k_k_list.extend(df_k_k_list)
                        max_rows.extend(all_blocks)
                        if max(all_blocks) > BLOCKS_MAX_ROWS:
                            logger.info(f"max NAMES per block={max(all_blocks)}")
    return max_rows, total_df_k_k_list


def get_data(fn):
    # Nan
    logger.info("read csv remove cols with missing values")
    df = read_data(fn)
    logger.info(f"\n{df.dtypes}")
    logger.info(f"cols = {list(df.columns)}")
    # remove NaN in latitude, longitude
    # latitude=-90.0 and longitude=0.0 1.6K - just NaN.
    # X, Y treat differently, by name only
    logger.info("remove anomalies")
    df_anom, df_clean = remove_anomalies(df)
    logger.info(f"\nanomalies number rows = {len(df_anom)}")
    if PLOT_DEBUG_FLAG:
        plt.hist(df["latitude"], bins = 20)
        plt.title(f"latitude anomalies")
        plt.show()
        plt.hist(df["longitude"], bins = 20)
        plt.title(f"longitude anomalies")
        plt.show()
        plt.hist(df_clean["latitude"], bins = 20)
        plt.hist(df_clean["longitude"], bins = 20)
        plt.show()

    return df_clean, df_anom


if __name__ == "__main__":
    df_clean, df_anom = get_data(fn_csv)
    # Blocks
    logger.info(f"running blocks margin={BLOCKS_EPS} BLOCKS_NUMBER={BLOCKS_NUMBER_LAT}x{BLOCKS_NUMBER_LON}")
    # total arguments
    if 1 == len(sys.argv):
        logger.info(f"\n subsample quick test\n")
        df_clean = df_clean.sample(n = 20000, replace = False)  # sample 20K rows
    else:
        logger.info(f"\n blocking all records (arg = {sys.argv[1]})\n")
    max_rows, total_df_k_k_list = create_blocks(df_clean)
    logger.info(f"Number blocks: {len(total_df_k_k_list)}")
    logger.info(
        f"Blocks: Max/Mean/std rows per block = {max(max_rows)}, {np.mean(max_rows):.2f}, {np.std(max_rows):.2f}")
    # pickle unpickle
    if 1 == len(sys.argv):
        pickling_on = open(f"{PATH}/blocks_small.pickle", "wb")
    else:
        pickling_on = open(f"{PATH}/blocks.pickle", "wb")
    pickle.dump([total_df_k_k_list, df_clean, df_anom], pickling_on)
    pickling_on.close()
