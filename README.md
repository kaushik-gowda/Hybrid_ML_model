## Hybrid Machine Learning model using Python

Combination of Different Types of Algorithms!!
Improvement in prediction performance

When a single algorithm cannot capture data complexity, build a hybrid model

Combination of:
    - LSTM for Sequencce learning
    - Linear Regression for trend analysis


The dataset is based on the stock market data

1. Converting the date column to a datetime type, setting it as the index and focusing on thr close price.
2. LSTM
    - It effectively captures sequential dependencies and patterns in time-series data.
    - Scaling close price between 0 and 1 using MinMaxScalar
    - Creating sequences of a defined length to predict next 60 days price
    - SPlitting the datset into train and test
    - Building the LSTm model with layers to capture the temporal dependencies
    - Compiling the model using,
        - Appropriate Optimizer
        - Loss function
    - Fit into the training data.

3. Linear Regression
    - It captures Simple linear realationships and long term trends in data.
    - Generating the lagged features for linear regression.
    - splitting data into training and testing.
    - training the model
    

4. Combination
    - lSTM's Ability to model complex time-dependent patterns.
    - Linear Regression's Ability to identify and follow broader trends.
    - Create a more balanced and accurate prediction system.
    - make prediction using LSTM on test set and inverse transform the scaled prediction.
    - make prediction using Linear Regression on test set and inverse transform the scaled prediction.
    - Combine both the models for prediction.
    
