NUM_CPU_THREADS = 64

# run blocks

BLOCKS_NUMBER_LAT = 100  # 1 km blocks
BLOCKS_NUMBER_LON = 100  # recursive splitting
BLOCKS_EPS = 0.001  # 100K blocks 100 m BLOCK MARGIN for LAT
BLOCKS_MAX_ROWS = 100
BLOCKS_MAX_RECURSIVE_DEPTH = 10  # allow 2^10 = 1000th fraction of LONG-LAT length
BLOCKS_NANS_DROP_COLS = [
    "geom",
    "restaurant_chain",
    "city",
    "delivery_radius",
    "restaurant_id",
]
BLOCKS_ANOMALY_STD_NUM = 3
PATH = "metrics"
PATH_CSV = "metrics"
PLOT_DEBUG_FLAG = False

# run clf

MAX_TRAIN = 300000
TRAIN_VALID_SPLIT = 0.8
NUM_RANDOM_BLOCKS_SELECT = 200  # 10min / 32K total, 10 blocks per min
LABELS_NOISE_NUM = 10
LABELS_SELECT_EACH_SIDE = 1
LABELS_MIN_THRESHOLD = 0.05
LABELS_MAX_THRESHOLD = 0.20
LABELS_WEIGHT = 10
N_ESTIMATORS = 200
MAX_DEPTH = 3
LABELS_STRATEGY = "high_noise"
LABELS_POSITIVE_MAX_DIST = 1e-4
LABELS_NEGATIVE_MIN_DIST = 0.01
PROCESS_PARALLEL = True
PARALLEL_NUM_CONCUR = 4

DEBUG_PLOT = True
LABELS_COLS_STRINGS = [
    "name",
    "standardized_name",
    "platform",
    # "sub_platform",
    # "country",
    "active",
]

# run dedup

DEDUP_BATCH_SIZE = 200000
DEDUP_PROB_THRESHOLD = 0.5
DEDUP_NUM_RANDOM_BLOCKS_SELECT = -1
DEDUP_BLOCKS_PER_PROCESS = 19
