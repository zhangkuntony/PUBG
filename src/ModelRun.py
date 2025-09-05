import DataPrepare as dataPrepare
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # dataPrepare.convert_data_to_feather()
    train = dataPrepare.load_train_feather()
    dataPrepare.feature_engineering_model_one(train)
    dataPrepare.remove_outliers(train)
    train = dataPrepare.one_hot_match_type(train)
    dataPrepare.category_match_group_id(train)


if __name__ == '__main__':
    main()