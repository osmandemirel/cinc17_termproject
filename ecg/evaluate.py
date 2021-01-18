import json
import numpy as np
from sklearn.metrics import classification_report

import load


def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []
    for d in data:
        labels.append(d['labels'])
    return labels


classes = ["A", "N", "O", "~"]
classes = sorted(set(classes))

int_to_class = dict(zip(range(len(classes)), classes))
class_to_int = {c : i for i, c in int_to_class.items()}

path_preds = "saved/preds/1607813643-299/preds.txt"
pred_porbs = np.loadtxt(path_preds)
pred_porbs = pred_porbs.reshape((pred_porbs.shape[0], pred_porbs.shape[1]/len(classes), len(classes)))
y_pred = np.argmax(pred_porbs, axis=2)
#preds = np.array([int_to_class[x] for x in preds[0]])


path_actual = "examples/cinc17/val.json"
actual = load.load_dataset(path_actual)

preproc = load.Preproc(*actual)
y_true = preproc.process_y(actual[1])
y_true = np.argmax(y_true, axis=2)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, confusion_matrix, recall_score


m = MultiLabelBinarizer().fit(y_true)

f1_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')

recall_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')

confusion_matrix(m.transform(y_true),
         m.transform(y_pred),
         average='macro')

classification_report(y_true, preds, labels=classes)
