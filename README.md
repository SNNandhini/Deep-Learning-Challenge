# Deep Learning Challenge
Charity Funding Predictor

The aim of this challenge is to create a tool for a non-profit foundation (Alphabet Soup) select applicants for funding with the best chance of success in their ventures.

The source CSV file contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. 

Using machine learning and neural networks, the features in dataset above are used to create a binary classifier that can predict (higher than 75% accuracy) whether applicants will be successful if funded by Alphabet Soup.

The steps followed to create the model are as follows:

## 1)   Preprocess the Data
-   Pandas and scikit-learn’s StandardScaler() are used to preprocess the dataset. 
-   Read in the charity_data.csv to a Pandas DataFrame.
    -   Identify target(s) for the model.
    -   Identify feature(s) for the model.
-   Drop the EIN and NAME columns.
-   Determine the number of unique values for each column.
-   For columns that have more than 10 unique values, determine the number of data points for each unique value.
-   Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
-   Use pd.get_dummies() to encode categorical variables.
-   Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
-   Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

## 2)   Compile, Train, and Evaluate the Model
-   Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
-   Create the first hidden layer and choose an appropriate activation function.
-   If necessary, add a second hidden layer with an appropriate activation function.
-   Create an output layer with an appropriate activation function.
-   Check the structure of the model.
-   Compile and train the model.
-   Create a callback that saves the model's weights every five epochs.
-   Evaluate the model using the test data to determine the loss and accuracy.
-   Save and export your results to an HDF5 file.

## 3)   Optimize the Model
-   Optimize the model to achieve a target predictive accuracy higher than 75%.
-   Use any or all of the following methods to optimize your model:
    -   Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
        -   Dropping more or fewer columns.
        -   Creating more bins for rare occurrences in columns.
        -   Increasing or decreasing the number of values for each bin.
    -   Add more neurons to a hidden layer.
    -   Add more hidden layers.
    -   Use different activation functions for the hidden layers.
    -   Add or reduce the number of epochs to the training regimen.

## 4)   Write a Report on the Neural Network Model
A report on the performance of the deep learning model created for Alphabet Soup is written.

The report contains the following:
-   Overview of the analysis
-   Results
    -   Data Preprocessing
        -   What variable(s) are the target(s) for the model?
        -   What variable(s) are the features for the model?
        -   What variable(s) should be removed from the input data because they are neither targets nor features?
    -   Compiling, Training, and Evaluating the Model
        -   How many neurons, layers, and activation functions were selected for the neural network model, and why?
        -   Was the target model performance achieved?
        -   What steps were taken in the attempts to increase model performance?
-   Summary: Summary of the overall results of the deep learning model, include a recommendation for how a different model could solve this classification problem.

## Files Uploaded
-   Source File - charity_data in Resources folder
-   Jupyter Notebooks - 1 for the original model and 4 for optimized models in Notebooks folder
-   Output Files - H5 files (1 for the original model and 4 for optimized models) in Output_files folder
-   Deep Learning Model Performance Report