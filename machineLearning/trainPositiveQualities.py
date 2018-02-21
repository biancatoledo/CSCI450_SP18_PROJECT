from __future__ import print_function

import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Get features and labels from csv file

    #Parameters
    num_classes = 6 # The 6 classes of qualities to encourage
    num_features = 25 # Amount of questions we are asking

# Organize data to be used for training

    # Create lists for Positive Qualities Model
        # Input and Target data
        pq_features = tf.placeholder(tf.float32, shape=[None, num_features])
        # For random forest, labels must be integers (the class id)
        pq_classes = tf.placeholder(tf.int32, shape=[None])

# Train and test Models

    # (1) Ensemble with Random Forests

        # Parameters
        num_steps = 500 # Total steps to train (Something to tune)
        batch_size = 1024 # The number of samples per batch (Something to tune)
        num_trees = 10 # Trees in the forest (Something to tune)
        max_nodes = 1000 # Nodes per tree (Something to tune)

    # Random Forest Parameters
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes, regression = True).fill()

    # Build the Random Forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    # Get training graph and loss
    train_op = forest_graph.training_graph(pq_features, pq_classes)
    loss_op = forest_graph.training_loss(pq_features, pq_classes)

    regressor = tf.contrib.learn.TensorForestEstimator(hparams)
    regressor.fit(x=pq_features, y=pq_classes, num_steps)

    # Test Random Forest
    y_new = regressor.predict(x_new)

# Basic Artificial Neural Network


# K Nearest Neighbor?


# Regression Tree? (CART)


# Hybrid NN and Ensemble?
