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

#print(X.head())
#print(y_named.head())
#print(X.describe())
#print(y_named.value_counts())
#print(X.info())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Divisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializza e addestra il classificatore
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predizioni sul test set
y_pred = clf.predict(X_test)

# Valuta l’accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuracy:.2f}")

import seaborn as sns
import matplotlib.pyplot as plt

# Aggiungi la colonna con i nomi delle classi
df = X.copy()
df['species'] = y.map(target_names)

#  Istogramma delle classi
sns.countplot(data=df, x='species')
plt.title("Distribuzione delle classi Iris")
plt.xlabel("Specie")
plt.ylabel("Frequenza")
plt.tight_layout()
plt.show()

# Scatter plot tra due feature
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.title("Petal length vs width per specie")
plt.tight_layout()
plt.show()

