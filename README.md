# RBM

First implementation and ideas of the Restricted Boltzmann Machine (RBM) algorithm for collaborative filtering. The goal is to implement a recommendation system for the mensa ratings. The project features two main architectural approaches:

1. **Single-User Mode**: Individual RBM models with shared weights for each user
2. **Global Mode**: A unified RBM model handling all users simultaneously

The training of large and sparse data is a major challenge in this project, and different architectural approaches are implemented to address this challenge.

A demo is implemented in order to demonstrate the implementation procedure for the dataset of our meal recommendation system. The corresponding test datasets are available upon request. A toy example on synthetic movie recommendations is also given in the examples folder.