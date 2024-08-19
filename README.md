# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

#### STEP 1:

Loading the dataset

#### STEP 2:

Split the dataset into training and testing

#### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

#### STEP 4:

Build the Neural Network Model and compile the model.

#### STEP 5:

Train the model with the training data.

#### STEP 6:

Plot the performance plot

#### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SHALINI V
### Register Number: 212222240096
#### IMPORT LIBRARIES:
```python
from google.colab import auth
import gspread
from google.auth import default

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

import pandas as pd
import matplotlib.pyplot as plt


```
#### AUTH AND READ GSPREAD:
```python
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('demo').sheet1 
rows = sheet.get_all_values()
```
#### OBJECT AND DATA SPLITTING:
```python
scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)
```
#### COMPLIE AND FIT:
```python
ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')
ai_brain.fit(x_train,y_train,epochs=1000,verbose=0)
```
#### POLT THE LOSS:
```python
loss_plot = pd.DataFrame(ai_brain.history.history)
loss_plot.plot()
```
#### ERROR AND PREDICT:
```python
err = rmse()
preds = ai_brain.predict(x_test)
err(y_test,preds)
```
#### ERROR AND PREDICT:
```python
err = rmse()
preds = ai_brain.predict(x_test)
err(y_test,preds)
x_n1 = [[28]]
x_n_n = scaler.transform(x_n1)
ai_brain.predict(x_n_n)
```
## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction

Include your sample input and output here

## RESULT

Include your result here
