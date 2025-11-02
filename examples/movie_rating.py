import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so 'src' package can be imported when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_handling.data_handler import process_csv, randomly_delete_ratings
from src.rbm.global_rbm import GlobalRBM


def _default_film_csv_path():
	repo_root = Path(__file__).resolve().parents[1]
	return repo_root / "data" / "test" / "FilmTestAndValidation.csv"

# Load data (use repo-relative default)
default_path = _default_film_csv_path()
data = process_csv(default_path)
dropped_data = randomly_delete_ratings(data, 2)
dropped_data[0] = [[0, 0, 1, 0, 0], [], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [], [0, 0, 0, 1, 0], [], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], []]

num_users = len(data)
K = 5
m = 12
hidden_units = 100
global_rbm = GlobalRBM(num_users, K, m, hidden_units)

global_rbm.set_user_data(dropped_data)
global_rbm.set_test_data(data)
print(dropped_data[0])
global_rbm.initialize_RBMs()
global_rbm.train(epochs=50) # it is quite remarkable, how 45 seems to be the magic number for convergence in this case
global_rbm.predict(user=0)