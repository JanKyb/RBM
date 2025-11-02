"""Configuration for RBM experiments and defaults.

Put small, experiment-level hyperparameters and paths here so they are
easy to find and adjust. Modules should import specific names rather than
pulling everything from this file.

This file is deliberately lightweight -- it's just constants, not a
full-featured configuration system.
"""

from pathlib import Path

# Initialization scale for weights and biases (stddev of normal init)
INIT_SCALE = 0.01

# Training defaults
LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 45

# Model defaults
DEFAULT_HIDDEN_UNITS = 100
DEFAULT_RATING_OPTIONS = 5

# Reproducibility
RANDOM_SEED = 42

# Data paths (repo-root relative). Tests/CLI scripts can resolve these
# using this file's location (two parents up -> project root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_TEST_DIR = DATA_DIR / "test"
DATA_PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# Verbosity / logging
VERBOSE = True

# Useful guard: list of keys exported
__all__ = [
	"INIT_SCALE",
	"LEARNING_RATE",
	"DEFAULT_EPOCHS",
	"DEFAULT_HIDDEN_UNITS",
	"DEFAULT_RATING_OPTIONS",
	"RANDOM_SEED",
	"DATA_DIR",
	"DATA_TEST_DIR",
	"DATA_PREPROCESSED_DIR",
	"VERBOSE",
]
