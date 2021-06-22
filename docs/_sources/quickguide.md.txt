(quickstartlabel)=
# A Five Minute Tutorial to Using Aklearn

## What is this library for?

Aklearn is a library to apply your favorite machine learning algorithms, with the help of some basic data preparation and model evaluation functionality. It is a personal project of mine to improve my coding and machine learning skills. The code is likely not complete as you are reading this, and there are plenty of bugs and non-functioning code. Nevertheless, I'm writing this for future practice, so I'll pretend everything is working just fine.

## Starting Ingredients

Most of the algorithms have highly similar usage, so let's examine an example workflow.

- Step 1: Identify our problem and obtain the data.
    - We want predict the price of Boston housing using some covariates.
    - We will want to make sure our data are numpy arrays

    ```
    # We'll import the data from sklearn
    from sklearn.datasets import load_boston
    import numpy as np


    # Define feature matrix X and output vector y
    # sklearn will automatically assign them appropriately
    X, y = load_boston(return_X_y=True)
    ```

- Step 2: Choose a model.
    - Let's do a standard linear regression, for simplicity of this tutorial. We will instantiate a `Linear(features, output, split_proportion, standardized)` object. This will require:
        - A feature matrix: set `features = X`.
        - An output vector: set `output = y`.
        - The proportion of data to train on: we'll set 75% for training, so `split_proportion = 0.75`.
        - Whether or not we wish to standardize X before hand (default is yes): set `standardized = True`.

- Step 3: Fit the model and evaluate. 
    - As soon as we instantiate our model, it will fit on the training data, give us the coefficient vector along with the train and test error (the MSE to be precise).

    ```
    model = Linear(features = X, 
                   output = y, 
                   split_proportion = 0.75, 
                   standardize = True)

    coefficients = model.coefficients
    train_error = model.train_error
    test_error = model.test_error
    ```

That's it! For a classification algorithm, the steps are similar. Make sure the output labels are integers. Further, if not all output labels appear in the training or testing data, you will have to 
