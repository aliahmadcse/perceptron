# Binary Classification system using single layer Perceptron

This project goal is to make a sinle layer perceptron that learns from a csv dataset and then make prediction on the data which is given to it. For this purpose, I made a perceptron which will train on any number of rows and columns and then able to make a prediction.

# Data Set

The project uses a dataset which have 5 columns. First four columns specified as inputs labels x1, x2, x3, x4 and 5th one is output label y.

# Instructions to execute the program


1. Download or clone the project from GitHub.

2. Open project in VS-Code (the modern open source editor).

3. Fire up the integrated terminal of vs-code and run command:
``` 
        pipenv install
```
to set the same virtual environment. (You might need to install pipenv before by using:
``` 
        pip install pipenv
```

4. After selecting the virtual environment created by pipenv for your project, run:
``` 
        python script.py –learn
```
to start learning.

5.	Enter the command below to test perceptron.
``` 
        python script.py --test
```

