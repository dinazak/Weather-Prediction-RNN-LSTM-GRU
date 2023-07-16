# Weather Prediction using RNN, LSTM, and GRU Models

This repository contains code for comparing the performance of RNN, LSTM, and GRU models in predicting weather data. The models are trained on historical weather data and used to forecast the maximum temperature for the next day.

## Problem Description
The task of weather prediction is a classic time series forecasting problem. In this project, we aim to predict the maximum temperature for the next day based on historical weather data. The dataset contains various features such as minimum temperature, precipitation, humidity, wind speed, etc., which can be used as input features for training the models. The goal is to build RNN, LSTM, and GRU models that can effectively learn the temporal dependencies in the data and make accurate temperature predictions.

## Dataset
The weather dataset used for this project is sourced from [Zenodo](https://zenodo.org/record/4770937#.ZFA1U3ZBzIW). It consists of historical weather data recorded over several years, including features such as date, minimum temperature, maximum temperature, precipitation, wind speed, and more. The dataset should be downloaded and stored in a specific directory or provided path before running the notebook.

## Data Preprocessing
Before training the models, the dataset needs to be preprocessed. This involves steps such as handling missing values, scaling numerical features, encoding categorical variables (if any), and splitting the data into training and validation sets. The notebook will provide code snippets and instructions on how to perform these preprocessing steps.

## Model Training
The notebook utilizes the Keras library to build and train the RNN, LSTM, and GRU models. Each model is trained using the mean squared error (MSE) loss function, which is a common choice for regression tasks. Hyperparameters such as the number of neurons in the hidden state, learning rate, optimizer, and batch size can be adjusted and experimented with to find the optimal configuration.

## Early Stopping
To prevent overfitting and determine the ideal number of epochs to train the models, early stopping is applied using the validation data loss. Early stopping monitors the validation loss during training and stops the training process when the loss stops improving. This helps to prevent the models from memorizing the training data and ensures better generalization to unseen data.

## Step Sizes
To compare the models' performance for different step sizes, three different step sizes are selected. The step size refers to the number of previous days' data used to predict the maximum temperature for the next day. The step sizes should be carefully chosen to represent small, medium, and large step sizes, allowing us to observe how the models perform under varying temporal dependencies.

## Evaluation and Results
The evaluation metric used for comparing the models is the mean squared error (MSE), which measures the average squared difference between the predicted and actual maximum temperature values. The notebook records and presents the MSE for each model and step size, as well as the epoch at which early stopping occurred. Additionally, the notebook may include visualizations of the training and validation loss curves for each model.

## Conclusion
The conclusion summarizes the key findings and observations from the experiment. It highlights the performance of each model for different step sizes, the impact of early stopping on the models' training, and any notable insights gained from the results. It also discusses which model performed best for each step size and provides recommendations for selecting an appropriate step size based on the dataset and problem requirements.

## Dependencies
The following libraries are required to run the notebook:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

## License
This project is licensed under the [MIT License](LICENSE).
