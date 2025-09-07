import logging
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def random_forest_regressor(x_train, y_train, x_valid, y_valid, n_estimators, min_samples_leaf, max_features):
    m = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)
    start_time = datetime.now()
    logging.info("Random Forest regressor start: {}".format(start_time))
    m.fit(x_train, y_train)
    end_time = datetime.now()
    time_difference = end_time - start_time
    logging.info("Random Forest regressor end: {}".format(end_time))
    logging.info(f"Random Forest regressor duration: {time_difference.total_seconds():.2f} 秒")

    y_pre = m.predict(x_valid)
    logging.info('Score: {}'.format(m.score(x_valid, y_valid)))
    logging.info('MAE: {}'.format(mean_absolute_error(y_true=y_valid, y_pred=y_pre)))
    return m