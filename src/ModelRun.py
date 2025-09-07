from sklearn.model_selection import train_test_split

import DataPrepare as dataPrepare
import logging
import MachineLearn as machineLearn
import pandas as pd

logging.basicConfig(level=logging.INFO)

def main():
    # dataPrepare.convert_data_to_feather()
    train = dataPrepare.load_train_feather()
    dataPrepare.feature_engineering_model_one(train)
    dataPrepare.remove_outliers(train)
    train = dataPrepare.one_hot_match_type(train)
    dataPrepare.category_match_group_id(train)

    df_sample = train.sample(1000000)
    df = df_sample.drop(columns = ['winPlacePerc'])
    y = df_sample['winPlacePerc']

    x_train, x_valid, y_train, y_valid = train_test_split(df, y, test_size = 0.2, random_state = 42)
    m1 = machineLearn.random_forest_regressor(x_train, y_train, x_valid, y_valid,
                                         n_estimators=40,
                                         min_samples_leaf=3,
                                         max_features='sqrt')

    ## 抽取特征
    fi = pd.DataFrame({
        'cols': df.columns,
        'importance': m1.feature_importances_
    }).sort_values(by='importance', ascending=False)
    logging.info(fi)

    to_keep = fi[fi.importance > 0.005].cols
    logging.info('Significant features: {}'.format(to_keep))

    # Make a DataFrame with only significant features
    df_keep = df[to_keep].copy()
    x_train, x_valid, y_train, y_valid = train_test_split(df_keep, y, test_size=0.2, random_state = 42)
    machineLearn.random_forest_regressor(x_train, y_train, x_valid, y_valid,
                                         n_estimators=40,
                                         min_samples_leaf=3,
                                         max_features='sqrt')

if __name__ == '__main__':
    main()