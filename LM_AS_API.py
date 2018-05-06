import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
reg = linear_model.LinearRegression()

data = pd.read_csv('drug.csv', skipinitialspace=True)

X = data[['alcohol-use']]
y = pd.factorize(data['alcohol-frequency'].values)[0].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg.fit(X_train, y_train)

//SAVING THE MODEL
//joblib.dump(reg, 'model.pkl')
//LOADING THE MODEL
//joblib_model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.get_json()
    query_df = pd.DataFrame(json_)
    model = joblib.load('model.pkl')
    prediction = model.predict(value)
    return jsonify({'prediction': list(prediction[0])})

if __name__ == '__main__':
    app.run(port=8080)
