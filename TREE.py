import graphviz
from sklearn.datasets import load_breast_cancer
from sklearn import tree

breast = load_breast_cancer()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(breast.data, breast.target)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("teste_final")


























