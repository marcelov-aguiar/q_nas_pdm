import os
from typing import List
from pathlib import Path



FD001: str = "FD001"
FD002: str = "FD002"
FD003: str = "FD003"
FD004: str = "FD004"

DEFAULT_MAX_NAME: str = "max"



PREFIX_DATA_TRAIN: str = "train_"
PREFIX_DATA_TEST: str = "test_"
PREFIX_DATA_RUL: str = "RUL_"



FILE_NAME_TRAIN_FD001: str = f'{PREFIX_DATA_TRAIN}{FD001}.txt'
FILE_NAME_TRAIN_FD002: str = f'{PREFIX_DATA_TRAIN}{FD002}.txt'
FILE_NAME_TRAIN_FD003: str = f'{PREFIX_DATA_TRAIN}{FD003}.txt'
FILE_NAME_TRAIN_FD004: str = f'{PREFIX_DATA_TRAIN}{FD004}.txt'

FILE_NAME_TEST_FD001: str = f'{PREFIX_DATA_TEST}{FD001}.txt'
FILE_NAME_TEST_FD002: str = f'{PREFIX_DATA_TEST}{FD002}.txt'
FILE_NAME_TEST_FD003: str = f'{PREFIX_DATA_TEST}{FD003}.txt'
FILE_NAME_TEST_FD004: str = f'{PREFIX_DATA_TEST}{FD004}.txt'

FILE_NAME_RUL_FD001 = f'{PREFIX_DATA_RUL}{FD001}.txt'
FILE_NAME_RUL_FD002 = f'{PREFIX_DATA_RUL}{FD002}.txt'
FILE_NAME_RUL_FD003 = f'{PREFIX_DATA_RUL}{FD003}.txt'
FILE_NAME_RUL_FD004 = f'{PREFIX_DATA_RUL}{FD004}.txt'

CalcRULTrain: str = "CalcRULTrain"

CalcRULTest: str = "CalcRULTest"

# Path
base_path = Path(__file__).resolve().parent.parent

SOURCE_RAW = os.path.join(base_path, "turbofan_engine", "data", "raw")

SOURCE_PROCESSED = os.path.join(base_path, "turbofan_engine", "data", "processed")

RAW_PATH_FD001_TRAIN = os.path.join(SOURCE_RAW, FD001, FILE_NAME_TRAIN_FD001)


RAW_PATH_FD001_TEST = os.path.join(SOURCE_RAW, FD001, FILE_NAME_TEST_FD001)


RAW_PATH_FD001_RUL = os.path.join(SOURCE_RAW, FD001, FILE_NAME_RUL_FD001)

PROCESSED_PATH_FD001_TRAIN = os.path.join(SOURCE_PROCESSED, FD001, FILE_NAME_TRAIN_FD001)

PROCESSED_PATH_FD001_TEST = os.path.join(SOURCE_PROCESSED, FD001, FILE_NAME_TEST_FD001)

FEATURE_TIME: str = "time"

FEATURE_UNIT_NUMBER: str = "unit_number"

FEATURES_NAME: List[str] = [FEATURE_UNIT_NUMBER, FEATURE_TIME, "setting_1", "setting_2",
                         "setting_3", "sensor_1", "sensor_2", "sensor_3",
                         "sensor_4", "sensor_5", "sensor_6", "sensor_7",
                         "sensor_8", "sensor_9", "sensor_10", "sensor_11",
                         "sensor_12", "sensor_13", "sensor_14", "sensor_15",
                         "sensor_16", "sensor_17", "sensor_18", "sensor_19",
                         "sensor_20", "sensor_21"]

FEATURES_TO_NORMALIZER = FEATURES_NAME[1:]

TARGET: str = "RUL"
TOTAL_RUL: str = "TOTAL_RUL"

LENGHT_ROI: int = 150

RULFilter: str = "RULFilter"
