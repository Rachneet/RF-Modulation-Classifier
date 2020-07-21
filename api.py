import logging
import os
import torch
from deployment import Model

from flask import Flask, request

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

model = "trained_cnn_no_intf_vsg_all"

@app.before_first_request
def load_model():
    global model
    # model_path = os.environ['REMOTE_MODEL_PATH']
    model_path = "/home/rachneet/thesis_results/"
    logging.info('Loading model: {}'.format(model_path))
    model = torch.load(model_path+model, map_location='cuda:0')

@app.route('/predict', methods=['POST'])
def predict():
    """Return a machine learning prediction."""
    global model
    data = request.get_json()
    logging.info('Incoming data: {}'.format(data))
    prediction = model.predict(data)
    inp_out = {'input': data, 'prediction': prediction}
    logging.info(inp_out)
    return inp_out


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)