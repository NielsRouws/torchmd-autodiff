# TorchMD Autodiff Example

## Introduction

This example demonstrates the benefits of automatic differentiation within a molecular dynamics engine.
We first simulate a waterbox and frequently save the coordinates and velocities.
Then we try to infer the force field parameters from these trajectories.
Finally, we try to reproduce the dynamics by a force field with a different functional form (maybe united atom).

## Requirements

- torchmd
- tqdm
- torch
- numpy
- matplotlib


## Scripts and Files

- waterbox.py -- Definition of the molecular system based on files in the torchmd test suite.
- simulate.ipynb -- Jupyter notebook for simulating the water box.
- learn.ipynb -- Jupyter notebook for learning the parameters from the trajectory.


## TODO
Try to fit a united-atom water model to the trajectory.