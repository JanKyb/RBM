import numpy as np
from src.rbm import config


class RBMBase:
    """Base class for RBM implementations.

    Stores parameters in a structured form (items, rating_options, hidden_units):
      - weights: shape (items, rating_options, hidden_units)
      - bias_visible: shape (items, rating_options)
      - bias_hidden: shape (hidden_units,)

    Provides helpers for validation, sigmoid/softmax and flattened views used by
    some algorithms that operate on a flattened visible vector of length items*rating_options.
    """

    def __init__(self, rating_options: int, items: int, hidden_units: int, init_scale: float = None):
        if not all(isinstance(x, int) and x > 0 for x in (rating_options, items, hidden_units)):
            raise ValueError("rating_options, items and hidden_units must be positive integers")

        self.rating_options = rating_options
        self.items = items
        self.hidden_units = hidden_units
        # Backwards-compatible flat visible unit count used elsewhere in the code
        self.visible_units = self.items * self.rating_options

        # Determine init_scale: if not provided, use value from config
        if init_scale is None:
            init_scale = getattr(config, "INIT_SCALE", 0.01)

        # Parameters stored in structured form
        self.weights = np.random.randn(self.items, self.rating_options, self.hidden_units) * init_scale
        self.bias_visible = np.random.randn(self.items, self.rating_options) * init_scale
        self.bias_hidden = np.random.randn(self.hidden_units) * init_scale

    def set_weights(self, weights, bias_visible, bias_hidden):
        if not isinstance(weights, np.ndarray) or weights.shape != (self.items, self.rating_options, self.hidden_units):
            raise ValueError(f"weights must be a numpy array of shape ({self.items}, {self.rating_options}, {self.hidden_units}).")
        if not isinstance(bias_visible, np.ndarray) or bias_visible.shape != (self.items, self.rating_options):
            raise ValueError(f"bias_visible must be a numpy array of shape ({self.items}, {self.rating_options}).")
        if not isinstance(bias_hidden, np.ndarray) or bias_hidden.shape != (self.hidden_units,):
            raise ValueError(f"bias_hidden must be a numpy array of shape ({self.hidden_units},).")

        self.weights = weights
        self.bias_visible = bias_visible
        self.bias_hidden = bias_hidden

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
