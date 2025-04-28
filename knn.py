#-------------------------------------------------------------------------
# AUTHOR: Daniel Appel
# FILENAME: knn.py
# SPECIFICATION: Implementing KNN for weather prediction
# FOR: CS 5990- Assignment #4
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

# Importing necessary Python libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# 11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

# Defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]  # 1 for Manhattan, 2 for Euclidean
w_values = ['uniform', 'distance']

# Reading the training data
df_train = pd.read_csv('weather_training.csv')
X_training = df_train[['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']].values
y_training_original = df_train['Temperature (C)'].values

# Discretize y_training
y_training = np.zeros(len(y_training_original), dtype=int)
for i in range(len(y_training_original)):
    min_diff = float('inf')
    best_class = 0
    for class_value in classes:
        diff = abs(y_training_original[i] - class_value)
        if diff < min_diff:
            min_diff = diff
            best_class = class_value
    y_training[i] = best_class

# Reading the test data
df_test = pd.read_csv('weather_test.csv')
X_test = df_test[['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']].values
y_test_original = df_test['Temperature (C)'].values

# Discretize y_test
y_test = np.zeros(len(y_test_original), dtype=int)
for i in range(len(y_test_original)):
    min_diff = float('inf')
    best_class = 0
    for class_value in classes:
        diff = abs(y_test_original[i] - class_value)
        if diff < min_diff:
            min_diff = diff
            best_class = class_value
    y_test[i] = best_class

# Keep track of the highest accuracy
highest_accuracy = 0.0
best_parameters = None

# Loop over the hyperparameter values (k, p, and w) of KNN
for k in k_values:
    for p in p_values:
        for w in w_values:
            # Fitting the KNN to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf.fit(X_training, y_training)
            
            # Initialize accuracy calculation
            correct_predictions = 0
            total_predictions = len(X_test)
            
            # Make the KNN prediction for each test sample and compute its accuracy
            for i, (x_testSample, y_testSample) in enumerate(zip(X_test, y_test_original)):
                # Make prediction
                y_pred = clf.predict([x_testSample])[0]
                
                # Calculate percentage difference
                percent_diff = 100 * abs(y_pred - y_testSample) / abs(y_testSample) if y_testSample != 0 else 100 * abs(y_pred - y_testSample)
                
                # Check if prediction is considered correct (within ±15%)
                if percent_diff <= 15:
                    correct_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / total_predictions
            
            # Check if this is the highest accuracy so far
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_parameters = (k, p, w)
                print(f"Highest KNN accuracy so far: {highest_accuracy:.2f}, Parameters: k = {k}, p = {p}, weight = {w}")

print(f"Final best parameters: k = {best_parameters[0]}, p = {best_parameters[1]}, weight = {best_parameters[2]}")
print(f"Highest accuracy achieved: {highest_accuracy:.2f}")