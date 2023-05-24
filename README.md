<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Cultural Data Science 2023</h1> 
  <h2 align="center">Assignment 2</h2> 
  <h3 align="center">Visual Analytics</h3> 


  <p align="center">
    Aleksandrs Baskakovs
  </p>
</p>


<!-- Assignment instructions -->
## Assignment instructions
For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset.

- Load the Cifar10 dataset
- Preprocess the data (e.g., greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier and one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.

<!-- ABOUT THE PROJECT -->
## About the project
This repository contains two scripts that enable image classification on the ```Cifar10``` dataset using either logistic regression or a neural network classifier built with ```scikit-learn```.

<!-- Data -->
## Data
The [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60K 32x32 colour images in 10 classes, with 6K images per class.

<!-- USAGE -->
## Usage
To use the code you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.76.0 (Universal). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment for the project. The code has been thoroughly tested and verified to work on a Mac machine running macOS Ventura 13.1. However, it should also be compatible with other Unix-based systems such as Linux. If you encounter any issues or have questions regarding compatibility on other platforms, please let me know.

1. Clone repository
2. Run ``setup.sh``
3. Run ```run_lr_classification.sh``` or ```run_nn_classification.sh```

### Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/sashapustota/image-classification
cd image-classification
```

### Run ```setup.sh```

The ``setup.sh`` script is used to automate the installation of project dependencies and configuration of the environment. By running this script, you ensure consistent setup across different environments and simplify the process of getting the project up and running.

The script performs the following steps:

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the required packages
4. Deactivates the virtual environment

### Run ```run_lr_classifier.sh``` or ```run_nn_classifier.sh```

The ```run_lr_classifier.sh``` and ```run_nn_classifier.sh``` scripts are used to run the logistic regression and neural network classifiers respectively. The scripts perform the following steps:

1. Load the Cifar10 dataset
2. Preprocess the data
3. Load the classifier
4. Fit the classifier
5. Save a classification report

To run the scripts, simply run the following commands in your terminal:

```bash
bash run_lr_classifier.sh
```

or

```bash
bash run_nn_classifier.sh
```

## Results

A classification report from the each approach on the data is saved to the folder ```out```.

## Customizing model parameters

The ```run_lr_classifier.sh``` and ```run_nn_classifier.sh``` scripts are designed to run the models with the default parameters. However, it is possible to customize the parameters by changing the values in the scripts or by passing the parameters as arguments in the terminal.

The following parameters can be customized for the **logistic regression classifier**:

* ```--tol -t``` - Tolerance for stopping criteria. Default value is ```0.1```.
* ```--solver -s``` - Algorithm to use in the optimization problem. Default value is ```saga```.
* ```--penalty -p``` - "Specify the norm of the penalty." Default value is ```none```.
* ```--max_iter -i``` - Maximum number of iterations taken for the solvers to converge. Default value is ```35```.

To pass the parameters as arguments in the terminal, simply run the following lines in your terminal:

```bash
python3 src/lr_classifier.py -t <your value> -s <your value> -p <your value> -i <your value>
```

The following parameters can be customized for the **neural network classifier**:

* ```--hidden_layer_sizes -hls``` - The ith element represents the number of neurons in the ith hidden layer. Default value is ```64,10```.
* ```--max_iter -i``` - Maximum number of iterations taken for the solvers to converge. Default value is ```20```.
* ```--learning_rate -l``` - Learning rate schedule for weight updates. Default value is ```adaptive```.
* ```--early_stopping -s``` - Whether to use early stopping to terminate training when validation score is not improving. Default value is ```True```.

To pass the parameters as arguments in the terminal, simply run the following lines in your terminal:

```bash
source ./cifar10_classifiers_venv/bin/activate
python3 src/nn_classifier.py -hls <your value> -i <your value> -l <your value> -s <your value>
deactivate
```

<!-- REPOSITORY STRUCTURE -->
## Repository structure
This repository has the following structure:
```
│   README.md
│   requirements.txt
│   run_lr_classifier.sh
│   run_nn_classifier.sh
│   setup.sh
│
├───out
│
└──src
      lr_classifier.py
      nn_classifier.py

```
<!-- REPOSITORY STRUCTURE -->
## Findings

When running the models with the parameters specified in the scripts, the following results were obtained:

```
lr-classifier
              precision    recall  f1-score   support

    airplane       0.34      0.38      0.36      1000
  automobile       0.36      0.38      0.37      1000
        bird       0.25      0.20      0.22      1000
         cat       0.21      0.15      0.18      1000
        deer       0.24      0.20      0.22      1000
         dog       0.29      0.29      0.29      1000
        frog       0.27      0.30      0.29      1000
       horse       0.29      0.30      0.30      1000
        ship       0.35      0.40      0.37      1000
       truck       0.39      0.45      0.41      1000

    accuracy                           0.31     10000
   macro avg       0.30      0.31      0.30     10000
weighted avg       0.30      0.31      0.30     10000
```
```
nn-classifier
              precision    recall  f1-score   support

    airplane       0.42      0.33      0.37      1000
  automobile       0.43      0.47      0.45      1000
        bird       0.28      0.40      0.33      1000
         cat       0.27      0.16      0.20      1000
        deer       0.32      0.20      0.25      1000
         dog       0.37      0.34      0.35      1000
        frog       0.31      0.53      0.39      1000
       horse       0.45      0.35      0.40      1000
        ship       0.47      0.48      0.47      1000
       truck       0.44      0.46      0.45      1000

    accuracy                           0.37     10000
   macro avg       0.38      0.37      0.37     10000
weighted avg       0.38      0.37      0.37     10000
```