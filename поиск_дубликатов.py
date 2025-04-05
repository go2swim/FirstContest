import pandas as pd

# Читаем CSV-файл. Если в файле нет заголовков, можно добавить параметр header=None
df = pd.read_csv('data_pizdata/X_train.csv')

# Поиск «почти дубликатов» с помощью корреляции
# Вычисляем абсолютную корреляционную матрицу
corr_matrix = df.corr().abs()
threshold = 0.99  # порог, выше которого считаем столбцы почти идентичными

# Находим пары столбцов с корреляцией выше порога
duplicate_pairs = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if corr_matrix.iloc[i, j] >= threshold:
            duplicate_pairs.append((i+1, j+1))


print(f"\nПары столбцов с корреляцией >= {threshold}:")
for pair in duplicate_pairs:
    print(pair)
print()

df = pd.read_csv('data_pizdata/X_test.csv')
for col1, col2 in duplicate_pairs:
    # Преобразуем номера столбцов в индексы (0-indexed)
    idx1 = col1 - 1
    idx2 = col2 - 1

    zero_ratio1 = (df.iloc[:, idx1] == 0).mean()
    zero_ratio2 = (df.iloc[:, idx2] == 0).mean()

    print(f"Пара ({col1}, {col2}):")
    print(f"  Столбец {col1}: {zero_ratio1:.2%} нулей")
    print(f"  Столбец {col2}: {zero_ratio2:.2%} нулей")

    if zero_ratio1 > zero_ratio2:
        print(f"  => Столбец {col1} повреждён\n")
    else:
        print(f"  => Столбец {col2} повреждён\n")
