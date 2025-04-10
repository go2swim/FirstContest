import numpy as np
from catboost import CatBoostRegressor
from src.const import RANDOM_STATE
from src.utils import load_data, save_predictions


def process_data(train_df, test_df):
    duplicates = [(4, 12), (5, 13), (6, 14), (7, 15)]
    cols_to_drop = [pair[0] for pair in duplicates]
    return (
        train_df.drop(columns=cols_to_drop),
        test_df.drop(columns=cols_to_drop)
    )


def train_and_predict(X_train, y_train, X_test):
    model_params = {
        'iterations': 30000,
        'learning_rate': 0.01946,
        'depth': 7,
        'l2_leaf_reg': 3,
        'random_state': RANDOM_STATE,
        'verbose': 0
    }

    model = CatBoostRegressor(**model_params)
    model.fit(X_train, y_train.values.ravel())
    return model.predict(X_test)


if __name__ == "__main__":
    df_x_train, df_y_train, df_test = load_data()

    X_train, X_test = process_data(df_x_train, df_test)

    predictions = train_and_predict(X_train, df_y_train, X_test)

    save_predictions(predictions)