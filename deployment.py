import logging
import argparse

logging.basicConfig(level=logging.INFO)


class Model():

    def __init__(self, model):
        self.model = model


    def predict_sample(self, input):
        '''
            Function that accepts a model and input data and returns a prediction.

            Args:
            ---
            model: a machine learning model.
            input_features: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

            Returns:
            --------
            prediction: Prediction of the model. Numpy array of shape (1,).
            '''
        pass


    def predict_batch(self, batch_input):
        '''
            Function that predicts a batch of samples.

            Args:
            ---
            model: a machine learning model.
            batch_input_features: A batch of features required by the model to
            generate predictions. Numpy array of shape (m, n) where m is the
            number of instances and n is the dimension of the feature vector.

            Returns:
            --------
            predictions: Predictions of the model. Numpy array of shape (m,).
            '''
        pass


    def to_remote(self):
        '''
        serializes a trained model and uploads it to a remote file system like
        S3 or Google Cloud Storage
        :return:
        returns the path to the serialized model
        '''


def run_batch_inference(remote_model_path):
    '''
    Generate and store a batch of predictions.

    Args:
    ---
    remote_model_path - Path to serialized model stored on remote object store.
    '''
    logging.info('Running batch inference.')
    raw_data = get_raw_inference_data()
    ''' 
    We first retrieve the raw input data for prediction. 
    This logic usually retrieves data from a database and can be parameterized as needed.
    '''

    logging.info('Retrieve serialized model from {}.'.format(
        remote_model_path))
    model = Model.from_remote(remote_model_path)
    X = model.preprocess(raw_data)
    predictions = model.predict_batch(X)

    logging.info('Writing predictions to database.')
    write_to_db(raw_data, predictions)
    '''
     function is responsible for writing the predictions to a database. I passed both the raw_data 
     and the predictions to that function because raw_data usually contains necessary metadata such as ID fields. 
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieve arguments for batch inference.')
    parser.add_argument('--remote_model_path', type=str, required=True,
                        help='Remote path to serialized model.')
    args = parser.parse_args()
    run_batch_inference(remote_model_path=args.remote_model_path)