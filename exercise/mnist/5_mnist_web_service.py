from exercise.mnist import mnist_model
import numpy as np
from flask import Flask
from flask import request, render_template

# Model Load
model = mnist_model.MNISTModel()

# Server Start
app = Flask("MNIST_Server", static_url_path='')


@app.route('/')
def index_page():
    return render_template('mnist.html')


@app.route('/mnist', methods=['POST'])
def do_predict():
    client_data = request.form['handwritten']
    client_data = client_data.replace("%2C", ",")
    matrix = np.matrix(client_data)
    predicted_number, probability = model.do_predict(matrix, image_processing=False)
    return str(predicted_number) + "|" + str(probability)

app.run()
