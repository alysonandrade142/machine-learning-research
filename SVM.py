# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm

data = pd.read_csv('TBLCENSO2011_v4_Regressao.csv', delimiter=";")

#      ------------ PRÉ-PROCESSAMENTO --------------

#
# ** COEF DE CORRELAÇÃO
#
# coefs = []
# refVector = []
#
# ** PEGANDO OS CAMPOS COM MAIOR COEF DE CORRELAÇÃO
#
#  Foi necessário criar 2 vetores, pois o retorno do personr são tuplas (que não podem ser alteradas),
#  por este motivo precisei criar um vector de "referência", contendo a tupla de retorno e o nome da coluna
#
# for column in data.columns:
#     if column != data.columns[-1]:
#         coef = stats.pearsonr(data[column], data[data.columns[-1]])
#         # print(column + " " + data.columns[-1])
#         coefs.append(coef)
#         refVector.append({"coef": coef, "column": column})
#
# ** ORDENANDO A LISTA E PEGANDO AS 10 COLUNAS COM MAIOR COEF
#
# coefSorted = sorted(coefs, key=lambda tup: (tup[0], -tup[1]), reverse=True)
# coefSorted = coefSorted[:10]
#
# ** PEGANDO O NOME DAS COLUNAS COM MAIOR COEF
#
# for value in refVector:
#     for tupla in coefSorted:
#         if tupla == value["coef"]:
#             print value["column"]
#
# ** RESULTADO
#
# ID_DEPENDENCIA_ADM
# ID_LABORATORIO_CIENCIAS
# ID_REG_INFANTIL_PREESCOLA
# ID_REG_MEDIO_MEDIO
# ID_EDUCACAO_FISICA
# AG_BIN_ID_EDUCACAO_FISICA_DOC
# IDADE_MESES
# REL_NUM_COMP_ADMINISTRATIVOS
# HORARIO
# AG_MED_NUM_TOTAL_ESCOLAS

columns = ['ID_DEPENDENCIA_ADM', 'ID_LABORATORIO_CIENCIAS', 'ID_REG_INFANTIL_PREESCOLA',
           'ID_REG_MEDIO_MEDIO', 'ID_EDUCACAO_FISICA', 'AG_BIN_ID_EDUCACAO_FISICA_DOC',
           'IDADE_MESES', 'REL_NUM_COMP_ADMINISTRATIVOS', 'HORARIO', 'AG_MED_NUM_TOTAL_ESCOLAS']

X = pd.DataFrame(data, columns=columns)
Y = data["EVADIU"]

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=1)

clf = svm.SVC(kernel="rbf")
clf.fit(X_treino, Y_treino)
result = clf.predict(X_teste)

cm = confusion_matrix(Y_teste, result)

print(cm)

print('Score', clf.score(X_teste, Y_teste))

