from flask import Flask, jsonify,  request, render_template
import numpy as np
import sklearn.externals
import joblib


app = Flask(__name__)
model_load = joblib.load("./logistic_regression_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        print('data', request.form.values())

        checkbox_values = []

        for i in range(1, 41):  # Assuming you have checkboxes named 'checkbox1', 'checkbox2', 'checkbox3'
            checkbox_name = f'checkbox{i}'
            checkbox_value = request.form.get(checkbox_name, '0')
            checkbox_values.append(checkbox_value)

        int_features = [int(x) for x in checkbox_values]

        final_features = [np.array(int_features)]
        print('final features', final_features)
        output = model_load.predict(final_features).tolist()
        print('final output', output)
        return render_template('index.html', prediction_text='{}'.format(output[0]))
        #return render_template('index.html', prediction_text='hello')
    else :
        return render_template('index.html')

@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)