#Remy Bikowski deliverable 9
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load in the dataset from the csv file
df = pd.read_csv('boston_housing.csv')

# create data frames for response and features
response = df['MEDV']
features = df[['RM']]

# Split the data into a train and a test set
X_train, X_test, y_train, y_test = train_test_split(features, response, test_size=0.2, random_state=42)

# Model selection and its fitting
lr = LinearRegression()
lr.fit(X_train, y_train)

# Generate predictions based on the test data
y_pred = lr.predict(X_test)

# Evaluate the model using sklearn module
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Interpret the results and print out an explanation to user
mse_sqrt = np.sqrt(mse)
print("The root mean squared error (RMSE) is approximately {:.2f}.".format(mse_sqrt))
print("This means that, on average, this model's predictions are off by about ${:.2f} thousand for the median value of owner-occupied homes.".format(mse_sqrt))

print("The R² score is {:.2f}.".format(r2))
print("This means that {:.2%} of the variation in the median values of the owner-occupied homes can be explained by the average number of rooms per home.".format(r2))
