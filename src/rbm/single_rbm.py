import numpy as np
from src.rbm.base_rbm import RBMBase


class SingleUserRBM(RBMBase):
    def __init__(self, K, m, hidden_units):
        """Initialize a per-user RBM backed by the structured parameter storage in RBMBase."""
        super().__init__(rating_options=K, items=m, hidden_units=hidden_units)

    def sample_hidden(self, data):
        """
        Compute hidden unit probabilities and states.
        
        Parameters:
        - data (numpy.ndarray): Input data of shape (items, rating_options).
        
        Returns:
        - tuple:
            - hidden_probs (numpy.ndarray): Probabilities of hidden units.
            - hidden_states (numpy.ndarray): Binary states of hidden units.
        """
        if not isinstance(data, np.ndarray) or data.shape[0] != self.items:
            raise ValueError(f"data must be a numpy array with {self.items} rows and {self.rating_options} columns.")

        hidden_activations = np.tensordot(data, self.weights, axes=([0,1], [0,1])) + self.bias_hidden
        hidden_probs = self.sigmoid(hidden_activations)
        hidden_states = hidden_probs > np.random.rand(*hidden_probs.shape)
        return hidden_probs, hidden_states.astype(float)

    def sample_visible(self, hidden_states):
        """
        Compute visible unit probabilities and states.
        
        Parameters:
        - hidden_states (numpy.ndarray): Hidden unit states of shape (items, hidden_units).
        
        Returns:
        - tuple:
            - visible_probs (numpy.ndarray): Probabilities of visible units.
            - visible_states (numpy.ndarray): Binary states of visible units.
        """
        if not isinstance(hidden_states, np.ndarray) or hidden_states.shape[0] != self.hidden_units:
            raise ValueError(f"hidden_states must be a numpy array with {self.hidden_units} rows.")

        visible_activations = np.tensordot(hidden_states, self.weights, axes=([0], [2])) + self.bias_visible
        visible_probs = np.zeros((hidden_states.shape[0], self.visible_units))
        visible_probs = np.zeros((self.items, self.rating_options))
        
        for i in range(self.items):
            visible_probs[i, :] = self.softmax(visible_activations[i, :])
        
        # Activate only the highest-probability unit in each K-group
        visible_states = np.zeros_like(visible_probs, dtype=int)
        visible_states[np.arange(visible_probs.shape[0]), visible_probs.argmax(axis=1)] = 1
        
        return visible_probs, visible_states

    def gibbs_sampling(self, hidden_states, iterations=1):
        """
        Perform Gibbs sampling to reconstruct visible and hidden states.
        
        Parameters:
        - hidden_states (numpy.ndarray): Initial hidden states of shape (items, hidden_units).
        - iterations (int): Number of Gibbs sampling iterations.
        
        Returns:
        - tuple:
            - neg_associations (numpy.ndarray): Negative associations for weight updates.
            - visible_probs (numpy.ndarray): Reconstructed probabilities of visible units.
            - hidden_probs (numpy.ndarray): Reconstructed probabilities of hidden units.
        """
        if not isinstance(hidden_states, np.ndarray):
            raise TypeError("hidden_states must be a numpy array.")
        if hidden_states.shape[0] != self.hidden_units:
            raise ValueError(f"hidden_states must have {self.hidden_units} rows.")
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer.")
        
        for _ in range(iterations):
            visible_probs, _ = self.sample_visible(hidden_states)
            hidden_probs, hidden_states = self.sample_hidden(visible_probs)

        neg_associations = np.tensordot(visible_probs, hidden_probs, axes=0)
        return neg_associations, visible_probs, hidden_probs
    
    def print_error(self, data, nvp, epoch=0):
        error = np.sqrt(np.sum((data - nvp) ** 2))
        print(f"Epoch {epoch}: error is {error:.6f}")
    
    def training_iteration(self, data, learning_rate=0.1, epoch=1, error_print = False):
        """
        Perform a single training iteration for the RBM.
        
        Parameters:
        - data (numpy.ndarray): The input data, with shape (items, rating_options).
        - learning_rate (float): The learning rate for the gradient update (default is 0.1).
        - epoch (int): The current epoch number (default is 1).
        
        Returns:
        - numpy.ndarray: The reconstructed visible probabilities after Gibbs sampling.
        
        Raises:
        - ValueError: If input data shape or epoch number is invalid.
        - TypeError: If learning_rate is not a float.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy ndarray.")
        if not isinstance(learning_rate, (float, int)):
            raise TypeError("learning_rate must be a float or int.")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a positive integer.")

        pos_hidden_probs, pos_hidden_states = self.sample_hidden(data)
        pos_associations = np.tensordot(data, pos_hidden_probs, axes=0)

        # Choose iteration value based on epoch range
        iteration_value = 1 if epoch < 15 else 3 if 15 <= epoch <= 25 else 5
        neg_associations, neg_visible_probs, neg_hidden_probs = self.gibbs_sampling(pos_hidden_states, iterations=iteration_value)

        # Compute gradients
        gradients = [
            pos_associations - neg_associations,
            data - neg_visible_probs,
            pos_hidden_probs - neg_hidden_probs
        ]

        if error_print: self.print_error(data, neg_visible_probs, epoch)
        
        return neg_visible_probs, gradients

    def train(self, data, learning_rate=0.1, epochs=500, error_print=False):
        """
        Train the RBM for a number of epochs.
        
        Parameters:
        - data (numpy.ndarray): The training data of shape (items, rating_options).
        - learning_rate (float): The learning rate for the gradient update (default is 0.1).
        - epochs (int): The number of epochs to train the model (default is 500).
        
        Raises:
        - ValueError: If input data shape is invalid.
        - TypeError: If learning_rate is not a float.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy ndarray.")
        if data.ndim != 2 or data.shape[0] != self.items or data.shape[1] != self.rating_options:
            raise ValueError(f"data must have shape ({self.items}, {self.rating_options})")
        if not isinstance(learning_rate, (float, int)):
            raise TypeError("learning_rate must be a float or int.")
        if not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs must be a positive integer.")

        for epoch in range(epochs):
            nvp, gradients = self.training_iteration(data, learning_rate=learning_rate, epoch=epoch, error_print=error_print)
            
            # Update parameters
            self.weights += learning_rate * gradients[0]
            self.bias_visible += learning_rate * gradients[1]
            self.bias_hidden += learning_rate * gradients[2]

if __name__ == '__main__':
    data = [[0, 0, 1, 0, 0], [], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [], [0, 0, 0, 1, 0], [], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [], []]

    np.random.seed(42)
    filtered_data = np.array([item for item in data if item])  # Keep non-empty lists
    filled_indices = [index for index, item in enumerate(data) if item]

    print("Filtered Data:", filtered_data)
    print("Indices of Filled Entries:", filled_indices)

    rating_options = 5
    items = filtered_data.shape[0]
    num_hidden = 10
    epochs = 50
    filtered_data = filtered_data.reshape(-1, rating_options)
    
    rbm = SingleUserRBM(rating_options, items, num_hidden)
    rbm.train(filtered_data, epochs=epochs, error_print=True)