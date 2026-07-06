import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# tool for preparing data for ML and splits dataset into training and testing data
from sklearn.model_selection import train_test_split

# Load the diabetes dataset - this will include all features unless specified 
calHouse_X, calHouse_y = datasets.fetch_california_housing(return_X_y=True)
'''
# Use 3 features
calHouse_X = calHouse_X[:, [0, 2, 3]]
# Use 1 feature
calHouse_X = calHouse_X[:, np.newaxis, 1]
'''



'''
# Split the data into training/testing sets
calHouse_X_train = calHouse_X[:-20]
calHouse_X_test = calHouse_X[-20:]

# Split the targets into training/testing sets
calHouse_y_train = calHouse_y[:-20]
calHouse_y_test = calHouse_y[-20:]
'''
# Split the data into training/testing sets
calHouse_X_train, calHouse_X_test, calHouse_y_train,
calHouse_y_test = train_test_split(
    calHouse_X,
    calHouse_y,
    test_size=0.01,
    random_state=42
)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(calHouse_X_train, calHouse_y_train)

# Make predictions using the testing set
calHouse_y_pred = regr.predict(calHouse_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(calHouse_y_test, calHouse_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(calHouse_y_test, calHouse_y_pred))

# Plot outputs
'''
plt.scatter(calHouse_X_test, calHouse_y_test, color="black")
plt.plot(calHouse_X_test, calHouse_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

'''
plt.figure(figsize=(6,6))

plt.scatter(calHouse_y_test, calHouse_y_pred, color="blue")

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.grid(True)

plt.show()
