from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()

X, Y = iris.data, iris.target

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=1)


mlp = MLPClassifier(solver='adam',
                    hidden_layer_sizes=(5,),
                    random_state=1,
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    max_iter=400,
                    activation='logistic',
                    momentum=0.1,
                    verbose=True,
                    early_stopping=True,
                    validation_fraction=0.3,
                    tol=0.0001)

mlp.fit(X_treino, Y_treino)
result = mlp.predict(X_teste)
print(result)