import numpy as np
from src.rbm.single_rbm import SingleUserRBM
from src.data_handling.data_handler import process_csv, randomly_delete_ratings
from src.plotting.plotting import plot_errors
from src.rbm.base_rbm import RBMBase
from tqdm import trange


class GlobalRBM(RBMBase):
    def __init__(self, num_users, K, m, hidden_units):
        """
        Initializes the Global RBM class with the provided parameters.

        Parameters:
            num_users (int): Number of users.
            K (int): Rating depth.
            m (int): Total number of movies.
            hidden_units (int): Number of hidden units.
        """
        if not isinstance(num_users, int) or num_users <= 0:
            raise TypeError(f"Expected 'num_users' to be a positive integer, but got {type(num_users)}.")
        if not isinstance(K, int) or K <= 0:
            raise TypeError(f"Expected 'K' to be a positive integer, but got {type(K)}.")
        if not isinstance(m, int) or m <= 0:
            raise TypeError(f"Expected 'm' to be a positive integer, but got {type(m)}.")
        if not isinstance(hidden_units, int) or hidden_units <= 0:
            raise TypeError(f"Expected 'hidden_units' to be a positive integer, but got {type(hidden_units)}.")

        # initialize base RBM parameters (weights, biases)
        super().__init__(rating_options=K, items=m, hidden_units=hidden_units)

        self.users = num_users
        self.rbm_list = []

        self.filtered_user_data = None
        self.test_data = None
        self.filtered_indices = None
        self.filtered_neg_indices = None

    def set_user_data(self, user_data):
        """
        Filters the user data and stores the filtered results.

        Parameters:
            user_data (list): List of user data sets to be filtered.
        """
        if not isinstance(user_data, list):
            raise TypeError(f"Expected 'user_data' to be a list, but got {type(user_data)}.")

        filtered_data = []
        filtered_indices = []
        filtered_neg_indices = []

        for dataset in user_data:
            # Collect non-empty rows and their indices
            filtered_dataset = np.array([row for row in dataset if row])
            indices = [index for index, row in enumerate(dataset) if row]
            neg_indices = [index for index, row in enumerate(dataset) if not row]
            filtered_data.append(filtered_dataset)
            filtered_indices.append(indices)
            filtered_neg_indices.append(neg_indices)

        self.filtered_user_data = filtered_data
        self.filtered_indices = filtered_indices
        self.filtered_neg_indices = filtered_neg_indices

    def set_test_data(self, test_data):
        """
        Filters the test data and stores the filtered results.

        Parameters:
            test_data (list): List of test data sets to be filtered.
        """
        if not isinstance(test_data, list):
            raise TypeError(f"Expected 'test_data' to be a list, but got {type(test_data)}.")
        
        # make sure that training data was set first
        if self.filtered_user_data is None:
            raise ValueError("Training data must be set before setting test data.")

        filtered_test_data = []
        filtered_test_indices = []

        for i, dataset in enumerate(test_data):
            if not isinstance(dataset, list):
                raise TypeError(f"Each dataset in 'test_data' must be a list, but got {type(dataset)}.")

            # Collect non-empty rows and their indices
            indices = [index for index, row in enumerate(dataset) if row]
            neg_indices = [index for index, row in enumerate(dataset) if not row]
        
            #compare which indices are not included in the training data and only keep those for testing
            unique_test_indices = [idx for idx in set(indices) if idx not in self.filtered_indices[i]]
            filtered_neg_test_indices = [idx for idx in set(neg_indices) if idx not in self.filtered_neg_indices]

            filtered_test_data.append([dataset[j] for j in unique_test_indices])
            filtered_test_indices.append(unique_test_indices)

        self.test_data = filtered_test_data
        self.test_indices = filtered_test_indices

    def initialize_RBMs(self):
        """
        Initializes the RBMs for each user and the global RBM.
        """
        if self.filtered_user_data is None:
            raise ValueError("User data has not been set.")

        for user in range(self.users):
            this_user_data = self.filtered_user_data[user]
            m = int(len(this_user_data))  # Every user gets a Boltzmann machine with the number of movies they have rated
            self.rbm_list.append(SingleUserRBM(self.rating_options, m, self.hidden_units))

        self.globalRBM = SingleUserRBM(self.rating_options, self.items, self.hidden_units)

    def train(self, epochs: int = 45, learning_rate: float = 0.01):
        """
        Trains the global RBM using the user data.

        Parameters:
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for training.
        """
        if not isinstance(epochs, int) or epochs <= 0:
            raise TypeError(f"Expected 'epochs' to be a positive integer, but got {type(epochs)}.")
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise TypeError(f"Expected 'learning_rate' to be a positive number, but got {type(learning_rate)}.")

        avg_weights = np.zeros_like(self.weights)
        avg_bias_visible = np.zeros_like(self.bias_visible)
        avg_bias_hidden = np.zeros_like(self.bias_hidden)

        test_error_list = []
        train_error_list = []
        
        for epoch in range(epochs):
            total_indices_occurence = np.zeros(self.items)  # Track indices of filled entries, necessary for the averaging step
            total_error = 0
            test_error = 0
            train_error = 0
            
            for user in range(self.users):
                current_indices = self.filtered_indices[user]
                total_indices_occurence[current_indices] += 1
                # Generate expanded indices (not needed with structured parameters)
                curr_Weights = self.weights[current_indices, :, :]  # Get the weights for the current user
                curr_bias_visible = self.bias_visible[current_indices, :]  # Get the visible biases for the current user
                curr_bias_hidden = self.bias_hidden  # Get the hidden biases for the current user
                
                self.rbm_list[user].set_weights(curr_Weights, curr_bias_visible, curr_bias_hidden)
                _, current_gradients = self.rbm_list[user].training_iteration(self.filtered_user_data[user], learning_rate=learning_rate, epoch=epoch)

                avg_weights[current_indices, : , :] += current_gradients[0]
                avg_bias_visible[current_indices, :] += current_gradients[1]
                avg_bias_hidden += current_gradients[2]
                
                # Calculate errors
                current_test_indices = self.test_indices[user]
                visible_probs, _ = self.predict(user)
                
                if current_test_indices == []:
                    test_error += 0 # If we have no test data for this user, skip error calculation
                else:
                    test_error += np.sum((self.test_data[user] - visible_probs[current_test_indices,:]) ** 2) / (self.users * self.rating_options * len(current_test_indices))
                
                train_error += np.sum((self.filtered_user_data[user] - visible_probs[current_indices]) ** 2) / (self.users * self.rating_options)
                
                # expanded_neg_indices = []
                # for idx in self.filtered_neg_indices[user]:
                #     expanded_neg_indices.extend([idx * self.rating_depth + i for i in range(self.rating_depth)])
                # test_error += np.sum((self.test_data[expanded_neg_indices, user] - visible_probs[:, expanded_neg_indices]) ** 2) / (self.users * len(expanded_neg_indices))
            
            test_error_list.append(np.sqrt(test_error))
            train_error_list.append(np.sqrt(train_error))

            # Average the weights based on total occurrence
            for i in range(self.items):
                if total_indices_occurence[i] == 0: continue
                avg_weights[i, :, :] /= total_indices_occurence[i]
                avg_bias_visible[i, :] /= total_indices_occurence[i]
                avg_bias_hidden /= self.users

            # Weight update
            self.weights += avg_weights * learning_rate / self.users  # Learning rate depends on the number of users
            self.bias_visible += avg_bias_visible * learning_rate / self.users
            self.bias_hidden += avg_bias_hidden * learning_rate / self.users

            self.globalRBM.set_weights(self.weights, self.bias_visible, self.bias_hidden)
            
            # print RMSE for each epoch
            print(f"Epoch {epoch + 1}: Test Error = {test_error_list[-1]:.4f}, Train Error = {train_error_list[-1]:.4f}")
                
        plot_errors(test_error_list, train_error_list)

    def predict(self, user) -> tuple:
        """
        Predicts the movie ratings for a user.

        Parameters:
            user (int): The index of the user.

        Returns:
            tuple: The predicted visible probabilities and states.
        """
        if not isinstance(user, int):
            raise TypeError(f"Expected 'user' to be an integer, but got {type(user)}.")
        
        hidden_states, _ = self.rbm_list[user].sample_hidden(self.filtered_user_data[user])
        visible_probs, visible_states = self.globalRBM.sample_visible(hidden_states)
        return visible_probs, visible_states