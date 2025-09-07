import logging
import pandas as pd

DATA_DIR = '../data/'
TRAIN_CSV_FILE = DATA_DIR + 'train_V2.csv'
TEST_CSV_FILE = DATA_DIR + 'test_V2.csv'
TRAIN_FEATHER_FILE = DATA_DIR + 'train_V2.feather'
TEST_FEATHER_FILE = DATA_DIR + 'test_V2.feather'

def convert_data_to_feather():
    logging.info('Start converting csv data into feather')
    logging.info('Load csv data')
    train = pd.read_csv(TRAIN_CSV_FILE)
    test = pd.read_csv(TEST_CSV_FILE)

    # 将数据集存储为feather格式，方便后续多次读取。Feather格式读取性能优于csv格式
    logging.info('Convert csv data to feather')
    train.to_feather(DATA_DIR + 'train_V2.feather')
    test.to_feather(DATA_DIR + 'test_V2.feather')

def load_train_feather():
    logging.info('Load train dataset feather file: {}'.format(TRAIN_FEATHER_FILE))
    train = pd.read_feather(TRAIN_FEATHER_FILE)
    return train

def load_test_feather():
    logging.info('Load test dataset feather file: {}'.format(TEST_FEATHER_FILE))
    test_df = pd.read_feather(TEST_FEATHER_FILE)
    return test_df

def remove_null_data(train):
    logging.info("Dataset shape before remove null values: {}".format(train.shape))
    logging.info('There are null winPlacePerc value in dataset with Index: {}'.format(
                 train[train['winPlacePerc'].isnull()].index))
    train.drop(train[train['winPlacePerc'].isnull()].index, inplace=True)
    logging.info("Dataset shape after remove null values: {}".format(train.shape))

def feature_engineering_model_one(train):
    remove_null_data(train)
    logging.info('Start feature engineering with model one: ((100-train[''playerJoined''])/100 + 1)')
    train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
    param = (100 - train['playersJoined']) / 100 + 1
    train['killsNorm'] = train['kills'] * param
    train['damageDealtNorm'] = train['damageDealt'] * param
    train['maxPlaceNorm'] = train['maxPlace'] * param
    train['matchDurationNorm'] = train['matchDuration'] * param
    logging.info('Feature engineering done')
    logging.info(train.head())

def feature_engineering_model_two(train):
    remove_null_data(train)
    logging.info('Start feature engineering with model two: 100 / train[''playersJoined'']')
    train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
    param = 100 / train['playersJoined']
    train['killsNorm'] = train['kills'] * param
    train['damageDealtNorm'] = train['damageDealt'] * param
    train['maxPlaceNorm'] = train['maxPlace'] * param
    train['matchDurationNorm'] = train['matchDuration'] * param
    logging.info('Feature engineering done')
    logging.info(train.head())

def remove_outliers(train):
    logging.info("Dataset shape before remove outliers: {}".format(train.shape))
    # Create feature totalDistance
    train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
    # Create feature killsWithoutMoving
    train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
    train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)

    # Create headshot_rate feature
    train['headshot_rate'] = train['headshotKills'] / train['kills']
    train['headshot_rate'] = train['headshot_rate'].fillna(0)
    # train.drop(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].index, inplace=True)

    # Drop roadKill 'cheaters'
    train.drop(train[train['roadKills'] > 10].index, inplace=True)

    # Remove outliers
    train.drop(train[train['kills'] > 30].index, inplace=True)

    # Remove outliers
    train.drop(train[train['longestKill'] >= 1000].index, inplace=True)

    # Remove outliers
    train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)
    train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
    train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)

    # Remove outliers
    train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
    train.drop(train[train['heals'] >= 40].index, inplace=True)

    logging.info("Dataset shape after remove outliers: {}".format(train.shape))

def one_hot_match_type(train):
    logging.info("Dataset shape before one_hot_match_type: {}".format(train.shape))
    # One hot encode matchType
    train = pd.get_dummies(train, columns=['matchType'])
    logging.info("Dataset shape after one_hot_match_type: {}".format(train.shape))
    logging.info(train.head())
    return train

def category_match_group_id(train):
    logging.info("Dataset shape before category_match_group_id: {}".format(train.shape))
    # Turn groupId and match Id into categorical types
    train['groupId'] = train['groupId'].astype('category')
    train['matchId'] = train['matchId'].astype('category')

    # Get category coding for groupId and matchID
    train['groupId_cat'] = train['groupId'].cat.codes
    train['matchId_cat'] = train['matchId'].cat.codes

    # Get rid of old columns
    train.drop(columns=['Id', 'groupId', 'matchId'], inplace=True)

    logging.info("Dataset shape after category_match_group_id: {}".format(train.shape))
    logging.info(train.info())
