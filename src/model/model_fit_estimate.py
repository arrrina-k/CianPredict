"""
implements trivial way to fit catboost regressor and to evaluate all metrics on processed cian dataset
"""

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class CianPredictionModel(CatBoostRegressor):
    def __init__(self):
        super().__init__()

    def fit_estimate(self, processed_data_path: str) -> None:
        """
        simplify the process of fitting and prints all the metrics associated with the model.
        :param processed_data_path:
        :return:
        """
        df = pd.read_csv(processed_data_path)
        categorical = ["author_type", "underground", "district"]
        x = df.drop(columns=["price", "lats", "lons", "street"])
        y = df["price"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)
        self.fit(x_train, y_train, cat_features=categorical)
        y_pred = self.predict(x_test)
        print(f"MSE: {mean_squared_error(y_test, y_pred)}")
        print(f"R2: {r2_score(y_test, y_pred)}")
