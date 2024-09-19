# About
This repository is divided into the following folders:

```plaintext
│
├── config/ #Folder with configuration files
|
├──controllers/ #Contains the controller files for each experiment in the project. Each controller is responsible for executing every step of the data project, from loading the dataset to model creation and evaluation.
|
├──docs/ # Documentation for the ds_library
|
├── ds_library/ # Data science library designed to be generic enough to be used in other data projects
|
├── test/ # Scripts with unit tests for the project's functions
|
├──turbofan_engine/ # Project dataset with its respective scripts
|
├──main.py # Responsible for calling all controllers and executing all experiments in the project.
|
```


# Turbofan
The NASA Turbofan Engine Degradation Simulation Dataset is widely used for predictive maintenance and prognostics research. It contains simulated data for multiple engine units, where each unit has run until failure under different operating conditions and fault modes. The dataset provides time-series data from various sensors monitoring parameters like temperature, pressure, and flow rates. The goal is often to predict the Remaining Useful Life (RUL) of the engines.

For more details and to access the dataset, you can visit [NASA's Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository).

