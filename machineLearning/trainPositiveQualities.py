from __future__ import print_function

import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Get features and labels from csv file

# Organize data to be used for training

    # Create lists for Serving Group Model

# Train and test Models

    # (1) Ensemble with Random Forests (sklearn)

    # Random Forest Parameters

    # Train test splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3 ,random_state=0)

    # Model fitting
    forest = RandomForestClassifier(n_estimators=500)
    forest.fit(X_train,y_train)

    # Test Random Forest
    # Model score
    print(forest.score(X_test,y_test))

# Basic Artificial Neural Network


# K Nearest Neighbor?


# Regression Tree? (CART)


# Hybrid NN and Ensemble?
