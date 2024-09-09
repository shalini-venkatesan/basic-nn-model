# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Developing a neural network regression model entails a structured process, encompassing phases such as data acquisition, preprocessing, feature selection, model architecture determination, training, hyperparameter optimization, performance evaluation, and deployment, followed by ongoing monitoring for refinement.

## Neural Network Model

![ex1 nn](https://github.com/user-attachments/assets/c8d7679a-1039-49be-b2c9-1e354457b1a1)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Shalini V
### Register Number: 212222240096

```python
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

ws = gc.open('demo').sheet1

rows = ws.get_all_values()
```
```python
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'sno':'float'})
df = df.astype({'marks':'float'})
df.head()

x = df[["sno"]].values
y = df[["marks"]].values
```
```python
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 33)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train1 = scaler.transform(x_train)
```
```python
marks_data = Sequential([Dense(6,activation='relu'),Dense(7,activation='relu'),Dense(1)])
marks_data.compile(optimizer = 'rmsprop' , loss = 'mse')

marks_data.fit(x_train1 , y_train,epochs = 500)

loss_df = pd.DataFrame(marks_data.history.history)
loss_df.plot()

x_test1 = scaler.transform(x_test)
marks_data.evaluate(x_test1,y_test)

X_n1 = [[30]]
X_n1_1 = scaler.transform(X_n1)
marks_data.predict(X_n1_1)
```
## Dataset Information

![image](https://github.com/user-attachments/assets/e35140c1-20b5-48d9-b13d-1ed8e1fcab13)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/1bc80fbb-0081-4dd1-b5e0-2fa21d13cb94)

### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/cbcd55df-da56-4b02-8e3b-05ec6625be44)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/a7c3d673-2143-4ede-aa79-109ec4142392)

## RESULT

Thus the program executed successfully
