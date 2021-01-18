from __future__ import print_function

import argparse
import numpy as np
import keras
import os

import load
import util

def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs = predict(args.data_json, args.model_path)
    
    #probs = predict("examples/cinc17/val.json", "saved/cinc17/1607813643-299/0.404-0.859-008-0.364-0.870.hdf5")
    output_path = os.path.join("saved", "preds", args.model_path.split("/")[-2])

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.savetxt(os.path.join(output_path, "preds_train.txt"), probs.reshape((probs.shape[0], -1)))
