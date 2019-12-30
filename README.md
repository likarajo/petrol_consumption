# Petrol Consumption

Predict petrol consumption based on different factors using **regression** with ***Tensorflow2.0***.

## Background

**Tensorflow2.0** is the latest version of ***Google***'s flagship deep learning platform.

* Uses Keras API as its default library for training classification and regression models.
* Problem with earlier versions of TensorFlow was the complexity of model creation. 

## Dependencies

* TensorFlow 2.0
* Pandas
* Scikit-learn
* Numpy

`pip install -r requirements.txt`

To make sure TensorFlow2.0 is installed: `pip install --upgrade tensorflow`

## Dataset

* Petrol Consumption Dataset: https://www.kaggle.com/harinir/petrol-consumption<br>
Saved in *data/petrol_consumption.csv*

* Details:
  * **Petrol_tax**: the buying price of the car)
  * **Average_income**:  the maintenance cost
  * **Paved_Highways**: number of doors
  * **Population_Driver_licence(%)**: the seating capacity
  * **Petrol_Consumption**: the luggage capacity

## Data Preprocessing

### Prepare features and labels

* Features = Petrol_tax, Average_income, Paved_Highways, Population_Driver_licence(%)
* Labels = Petrol_Consumption

### Split data into Training and Test sets

* Training set = 80%
* Test set = 20%

### Normalize the training and test features

For regression problems in general and deep learning, it is highly recommended to normalize the dataset.

## Create model

* Using **Keras** functional API
* Input layer
* 3 Hidden Dense layers with 100, 50 and 25 neurons respoectively and ReLU activation function
* Output with 1 neurons for 1 output value
* Loss function = mean squared error
* Optimizer = Adam
* Evaluation metric = mean squared error

## Train the model

* Epochs = 100
* Validation data = 20% of training data

## Evaluate the model

Using ***Root mean squared error***.  

* Find mean squared error between the predicted and actual values
* Find the square root of the mean squared error  

## Conclusion

* The results show that model performance is better on the training set since the root mean squared error value for training set is less.  
* The model is overfitting.  
  * The reason is obvious, as there are only 48 records in the dataset.  
  * Training the regression model with a larger dataset will give better result.  

