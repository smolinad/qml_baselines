# Quantum Machine Learning Models Repository

Sebastian Molina - [smolinad@unal.edu.co](mailto:smolinad@unal.edu.co)  
Sergio Quiroga - [squirogas@unal.edu.co](mailto:squirogasd@unal.edu.co)  
Supervisors: Fabio Gonzalez & Diego Useche.

This repository is a collection of quantum machine learning models implemented using various quantum computing frameworks. 
By the time being, the models were implemented on Tencents's `TensorCircuit` and IBM's `Qiskit`.

The webview of the notebooks can be seen [here](https://smolinad.github.io/quantum_machine_learning/docs).

## Model 1: Quantum Variational Circuit (Farhi & Neven)
- Description: Implements a quantum variational circuit for MNIST Digits classification, as per Farhi & Neven (2018).
- Directory: `/mnist`
- We introduce a companion Marimo notebook. Marimo is a new generation Python notebook that provides reactive and interactive data insights.
  The notebook is located [https://smolinad.github.io/quantum_machine_learning/docs/mnist](here), and it can by run as follows:
    + Install `marimo` with `pip install marimo`.
    + Download the notebook, and in the downloads directory (or any chosen directory) run `marimo edit nameofthenotebook.py`.
    + A local host will promp in your default browser, just as Jupyter notebook. 

## Model 2: Quantum Support Vector Machine with $ZZ$-feature map
- Description: Implements a quantum version of a support vector machine for classification tasks.
- Directory: `/qsvm`

## Poster 
- Description: A poster related to the motivations and objectives of this repository.
- Directory: https://smolinad.github.io/quantum_machine_learning/docs/poster
  
## Paper
- Description: A paper related to the motivations and objectives of this repository.
- Directory: https://smolinad.github.io/quantum_machine_learning/docs/paper
