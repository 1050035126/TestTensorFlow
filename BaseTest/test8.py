from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np


IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# If the training and test sets aren't stored locally, download them.
if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb+") as f:
                f.write(raw)

if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb+") as f:
                f.write(raw)

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

# # Data sets
# IRIS_TRAINING = "dataSet/iris_training.csv"
# IRIS_TEST = "dataSet/iris_test.csv"
#
# # Load datasets.
# training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TRAINING,
#     target_dtype=np.int,
#     features_dtype=np.float32)
#s test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TEST,
#     target_dtype=np.int,
#     features_dtype=np.float32)
#
# from sklearn.datasets import load_iris
#
# dataSet = load_iris()
#
# train_feature=np.array(dataSet.data[np.s_[0:100]])
# train_target=np.array(dataSet.target[np.s_[0:100]])
#
# test_feature=np.array(dataSet.data[np.s_[100:]])
# test_target=np.array(dataSet.target[np.s_[100:]])




# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
