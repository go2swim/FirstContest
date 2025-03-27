import os
import pandas as pd
from src.const import PREDICTS_DIR, PATH_TO_DATA


def load_data(path_to_data=PATH_TO_DATA):
    df_x_train = pd.read_csv(os.path.join(path_to_data, "X_train.csv"),
                             delimiter=',',
                             header=None)
    df_y_train = pd.read_csv(os.path.join(path_to_data, "y_train.csv"),
                             delimiter=',',
                             header=None)
    df_test = pd.read_csv(os.path.join(path_to_data, "X_test.csv"),
                          delimiter=',',
                          header=None)

    df_x_train.columns = [i for i in range(df_x_train.shape[1])]
    df_test.columns = df_x_train.columns

    return df_x_train, df_y_train, df_test


def save_predictions(predictions, predicts_dir=PREDICTS_DIR, file_prefix="predictions"):
    os.makedirs(predicts_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(predicts_dir)
                      if f.startswith(file_prefix + '_') and f.endswith('.csv')]

    max_num = 0
    for f in existing_files:
        try:
            num = int(f[len(file_prefix) + 1:-4])
            if num > max_num:
                max_num = num
        except ValueError:
            continue

    new_filename = f"{file_prefix}_{max_num + 1}.csv"
    file_path = os.path.join(predicts_dir, new_filename)

    submission = pd.DataFrame({
        'y': predictions,
        'ID': range(len(predictions)),
    })
    submission.to_csv(file_path, index=False)

    print(f"Предсказания сохранены в: {file_path}")
    return file_path