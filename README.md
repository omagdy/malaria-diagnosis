# Malaria Diagnosis
## Introduction
This project utilizes a CNN to detect whether a colored microscopic image contains malaria or not.

## Training
To train the model use the following command:
```
python main.py -lr 0.01 -bs 200 -ep 100 
```
where lr is the learning rate, bs is the batch size and ep is the number epochs for training.

## Possible Improvements
1. Utilize an optimizer to find the optimum hyperparameters for training the model
2. Create a deeper model to obtain better performance and utilize skip connections to minimize the vanishing gradient problem
