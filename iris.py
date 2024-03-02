import pandas as pd
from sklearn.datasets import load_iris

# Загрузка данных Iris из sklearn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Сохранение в CSV
df.to_csv('iris.csv', index=False)
