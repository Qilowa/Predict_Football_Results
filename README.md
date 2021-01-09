# Predict_Football_Results

## Description

Predict football results using the number of shots on target.

## 
Let's look at Levante's shots on target throughout the 2019-2020 season.

![histogram of levante's shots on target](./imgs/sot.png?raw=true)

We can see that this follows approximatively a normal distribution.

![histogram of levante's shots on target](./imgs/sota.png?raw=true)

We can also do the same with the shots on target allowed by a team.

Using the Kernel Density Estimation from ```scikit-learn```, we will be able to have samples from the distribution. 