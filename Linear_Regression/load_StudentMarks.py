import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# tool for preparing data for ML and splits dataset into training and testing data
from sklearn.model_selection import train_test_split

# Input feature (Hours Studied)
hours_X = np.array([[1],[2],[3],[4],[5],[7],[8],[10],[13],[14]])
# Target variable (Marks)
marks_y = np.array([50, 55, 60, 65, 70, 90, 100, 120, 150, 160])

print("hours_X:")
print(hours_X.flatten())

print("\nmarks_y:")
print(marks_y)

hours_X_train, hours_X_test, marks_y_train, marks_y_test = train_test_split(
    hours_X,
    marks_y,
    test_size=0.3,
    random_state=42
)
print("hours_X_train:")
print(hours_X_train.flatten())

print("\nhours_X_test:")
print(hours_X_test.flatten())

print("\nmarks_y_train:")
print(marks_y_train)

print("\nmarks_y_test:")
print(marks_y_test)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(hours_X_train, marks_y_train)

# Make predictions using the testing set
marks_y_pred = regr.predict(hours_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(marks_y_test, marks_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(marks_y_test, marks_y_pred))

# Plot outputs

plt.scatter(hours_X_test, marks_y_test, color="black")
plt.plot(hours_X_test, marks_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
