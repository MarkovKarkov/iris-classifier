# main.py
from sklearn.datasets import load_iris
import pandas as pd

# Carica il dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Mappa i nomi delle specie
target_names = dict(enumerate(iris.target_names))
y_named = y.map(target_names)

# Stampa un campione
print(X.head())
print(y_named.head())
