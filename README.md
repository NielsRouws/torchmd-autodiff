# TorchMD Autodiff Example

## Introduction

This example demonstrates the benefits of automatic differentiation within a molecular dynamics engine.
We first simulate a waterbox and frequently save the coordinates and velocities.
Then we try to infer the force field parameters from this trajectory.



## Requirements

- torchmd
- tqdm
- torch (tested with version 1.4.0)
- numpy
- matplotlib


## Scripts and Files

- waterbox.py -- Definition of the molecular system based on files in the torchmd test suite.
- simulate.ipynb -- Jupyter notebook for simulating the water box.
- learn.ipynb -- Jupyter notebook for learning the parameters from the trajectory.
- integrator.py -- A version of the torchmd integrator without inplace operations
- plot.ipynb -- plotting script

