import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

previsores = pd.read_csv('/content/drive/MyDrive/Breast cancer/entradas_breast.csv')
classe = pd.read_csv('/content/drive/MyDrive/Breast cancer/saidas_breast.csv')

# Convertendo para arrays numpy
X = previsores.values
Y = classe.values.ravel()  # Utilizando .ravel() para transformar em array 1D

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

breast = pd.concat([classe, previsores], axis=1)

# Exibir o dataframe concatenado
print(breast)

plt.figure(figsize=(20, 10))
sns.heatmap(previsores.corr(), annot=True, cmap='rocket', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show

# Criar uma instância do modelo SVM
modelo = SVC(kernel='linear')  # Usando kernel linear para SVM de classificação

# Treinar o modelo usando os dados de treino
modelo.fit(X_train, Y_train)

#make it binary classification problem
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]

# Fazer previsões no conjunto de teste
Y_pred = modelo.predict(X_test)

# Avaliar a acurácia do modelo
acuracia = accuracy_score(Y_test, Y_pred)
print(f'Acurácia do modelo SVM: {acuracia}')

# Outras métricas de avaliação
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))


