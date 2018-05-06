
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('bitcoin.csv')
Y = np.array(data['Y'])
X = np.array(data['X']).reshape((len(data['X']), 1))


X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=1)

mlp = MLPRegressor(solver='adam',
                    hidden_layer_sizes=(1000,),
                    random_state=1,
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    max_iter=4000,
                    activation='logistic',
                    momentum=0.1,
                    verbose=True,
                    early_stopping=True,
                    validation_fraction=0.3,
                    tol=0.0001)

mlp.fit(X_treino, Y_treino)

r = mlp.predict(X_teste)

print('Score', mlp.score(X_teste, Y_teste ))