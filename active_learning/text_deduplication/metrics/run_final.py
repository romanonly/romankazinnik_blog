import logging

import numpy as np
import pandas as pd
from metrics.run_blocks import read_data, fn_csv
from metrics.settings import PATH

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

fn_clean = f'{PATH}/css_clean.csv'
fn_duplicates = f'{PATH}/css_duplicates.csv'
fn_duplicates_pairs = f'{PATH}/css_duplicates_pairs.csv'

if __name__ == "__main__":
    """ select proba threshold and create x2 csv files: duplicates and clean """
    logger.info(f"reading {fn_csv} ")
    df = read_data(fn_csv, is_remove_id = False)
    df_pos_concat = pd.DataFrame()
    for ind in range(100000):
        try:
            fn1 = f"{PATH}/duplicates_big_{ind}.csv"
            df_pos = pd.read_csv(f"{fn1}", sep = ",", dtype = str, index_col = 0)
            df_pos.reset_index(inplace = True, drop = True)
            df_pos['probability_duplicate'] = df_pos['probability_duplicate'].astype(float)
            df_pos['idx_x'] = df_pos['idx_x'].astype(int)
            df_pos['idx_y'] = df_pos['idx_y'].astype(int)
            df_pos_concat = pd.concat(objs = [df_pos_concat, df_pos], axis = 0)
            df_pos_concat.drop_duplicates(['idx_x', 'idx_y'], keep = "last", inplace = True)
            logger.info(f"read dedup {fn1} total rows={len(df_pos_concat)}")
        except Exception as e:
            pass  # logger.info(f" reading {ind} files finished ({e}).")
    # Threshold
    # sort by probability and select proba_threshold
    df_pos = df_pos_concat
    iterate = 0
    df_pos.sort_values(by = ['probability_duplicate'], ascending = False, inplace = True)
    logger.info(f" write pairs sorted by predicted probability {fn_duplicates_pairs} rows={len(df_pos)}")
    df_pos.to_csv(fn_duplicates_pairs)
    new_Y = None
    ind1, ind2 = 0, int(0.5 * len(df_pos) - 1)
    NUM_PAIRS_SHOW = 5
    while new_Y is None or not (ind2 - ind1 < 3 * NUM_PAIRS_SHOW or new_labels == ""):
        ind_1_2 = int(0.5 * (ind1 + ind2))
        iterate += 1
        # select 5 pts at id_1_2, and go up and down
        df_update = df_pos.iloc[ind_1_2 - NUM_PAIRS_SHOW:ind_1_2]
        prob_min = df_pos.iloc[ind_1_2]['probability_duplicate']
        logger.info(
            f"\niteration={iterate} (binary len={ind_1_2 - ind1}) prob_min={prob_min:.3f}"
            f"\n\nPlease input 1 (num mistakes <=1) or 0 (mistakes>1). ENTER to quit:\n"
        )
        logger.info(f"\n{df_update}")
        new_labels = input()
        try:
            new_Y = np.array([np.bool_(np.int_(x)) for x in new_labels.split(",")])
        except Exception as e:
            logger.info(f" *** END MANUAL ENTER ({e}).")
        if new_Y == 1:  # go down, increase proba
            ind1 = ind_1_2
        elif new_Y == 0:
            ind2 = ind_1_2

    prob_min = df_pos.iloc[ind_1_2]['probability_duplicate']
    df_pos = df_pos[df_pos['probability_duplicate'] > prob_min]
    idx_set = set(df_pos["idx_x"])
    idx_set = idx_set.union(set(df_pos["idx_y"]))
    idx_list = list(idx_set)
    idx_list.sort()
    df_exam = df_pos[df_pos["idx_x"].isin(idx_list[:3]) | df_pos["idx_y"].isin(idx_list[:3])]
    logger.info(f"example duplicates idx={idx_list[:3]}: \n{df[df['idx'].isin(idx_list[:3])]}")
    logger.info(f"writing: {fn_clean}, {fn_duplicates}")
    mask = df['idx'].isin(idx_list)
    df_duplicates = df[mask]
    df_clean = df[~mask]
    logger.info(f"\nnum duplicates={len(df_duplicates)}\nnum clean={len(df_clean)}")
    df_clean.to_csv(fn_clean)
    df_duplicates.to_csv(fn_duplicates)
