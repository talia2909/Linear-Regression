#linear_regression

## Introduction

The `linear_regression.py` script provides an implementation of the linear regression model using gradient descent to predict the progression of diabetes in patients. The script uses the renowned diabetes dataset from `sklearn` and contrasts the custom implementation with results obtained from scikit-learn's linear regression model.

## Features

1. **Data Preparation**:
    - Automatically loads the diabetes dataset.
    - Splits the dataset into training and test sets for model evaluation.

2. **Linear Regression with Scikit-Learn**:
    - Utilizes the scikit-learn library to create and train a linear regression model.
    - Offers a sanity check by displaying coefficients and the mean squared error (MSE).

3. **Custom Implementation of Linear Regression**:
    - Implements linear regression from scratch using gradient descent.
    - Continuously updates coefficients to minimize the MSE.
    - Displays diagnostic outputs at regular intervals during training.

4. **Result Export**:
    - Exports the predicted values of the test dataset to a text file. The filename corresponds to the student's ID.

## Usage

1. **Dependencies**:
    - Ensure you have the required libraries installed: `numpy`, `sklearn`, `json`, and `matplotlib`.

2. **Student ID**:
    - Update the `ID` variable at the end of the script with the relevant student or user ID.

3. **Execution**:
    - Run the script. This will load the data, train both the scikit-learn and custom models, display results, and export predictions.

4. **Output**:
    - After successful execution, a text file named `<Student_ID>.txt` will be generated, containing the predicted values for the test samples.

## Conclusion

The `linear_regression.py` script offers a hands-on approach to understanding linear regression, emphasizing the gradient descent optimization technique. While it's designed primarily for educational purposes, it can serve as a foundation for more advanced regression tasks.
