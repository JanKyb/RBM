import numpy as np
from src.rbm.single_rbm import SingleUserRBM

def test_single_user_rbm_initialization():
    # Test valid initialization
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    # New structured storage: items x rating_options x hidden_units
    assert rbm.visible_units == 50  # K * m (backwards-compatible)
    assert rbm.hidden_units == 3
    assert rbm.items == 10
    assert rbm.rating_options == 5
    assert rbm.weights.shape == (10, 5, 3)  # (items, rating_options, hidden_units)
    assert rbm.bias_visible.shape == (10, 5)
    assert rbm.bias_hidden.shape == (3,)

    # Test invalid initialization (negative values)
    try:
        rbm = SingleUserRBM(K=-5, m=10, hidden_units=3)
    except ValueError as e:
        assert str(e) == "rating_options, items and hidden_units must be positive integers"
    
    # Test invalid initialization (non-integer values)
    try:
        rbm = SingleUserRBM(K=5.5, m=10, hidden_units=3)
    except ValueError as e:
        assert str(e) == "rating_options, items and hidden_units must be positive integers"

def test_set_weights():
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    
    # Test valid weight setting
    # Use structured shapes for weights/biases
    weights = np.random.randn(10, 5, 3)
    bias_visible = np.random.randn(10, 5)
    bias_hidden = np.random.randn(3)
    rbm.set_weights(weights, bias_visible, bias_hidden)
    assert np.array_equal(rbm.weights, weights)
    assert np.array_equal(rbm.bias_visible, bias_visible)
    assert np.array_equal(rbm.bias_hidden, bias_hidden)

    # Test invalid weight setting (wrong shape)
    try:
        rbm.set_weights(np.random.randn(40, 3), bias_visible, bias_hidden)
    except ValueError:
        pass
    
    # Test invalid bias setting (wrong shape)
    try:
        rbm.set_weights(weights, np.random.randn(40), bias_hidden)
    except ValueError:
        pass
    
    try:
        rbm.set_weights(weights, bias_visible, np.random.randn(4))
    except ValueError:
        pass
        
        #print dimensions
    print("Weights shape:", rbm.weights.shape)
    print("Bias visible shape:", rbm.bias_visible.shape)
    print("Bias hidden shape:", rbm.bias_hidden.shape)

def test_sigmoid():
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    x = np.array([0, 1, -1])
    result = rbm.sigmoid(x)
    assert np.allclose(result, 1 / (1 + np.exp(-x)))
    
    print("Sigmoid result shape:", result.shape)

def test_softmax():
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    x = np.array([[1, 2, 3], [1, 2, 3]])
    result = rbm.softmax(x, axis=1)
    assert np.allclose(result, np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True))
    
    print("Softmax result shape:", result.shape)

def test_sample_hidden():
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    # New API: sample_hidden expects a single structured sample of shape (items, rating_options)
    data = np.random.randn(rbm.items, rbm.rating_options)
    hidden_probs, hidden_states = rbm.sample_hidden(data)
    assert hidden_probs.shape == (rbm.hidden_units,)  # single sample -> hidden vector
    assert hidden_states.shape == (rbm.hidden_units,)
    assert np.all((hidden_states == 0) | (hidden_states == 1))  # Binary states

    # Test invalid data shape
    try:
        rbm.sample_hidden(np.random.randn(4, 40))  # Wrong shape
    except ValueError:
        pass
        
    print("Hidden probs shape:", hidden_probs.shape)

def test_sample_visible():
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    # New API: sample_visible expects a single hidden vector
    hidden_states = np.random.randn(rbm.hidden_units)
    visible_probs, visible_states = rbm.sample_visible(hidden_states)
    assert visible_probs.shape == (rbm.items, rbm.rating_options)
    assert visible_states.shape == (rbm.items, rbm.rating_options)
    assert np.all((visible_states == 0) | (visible_states == 1))  # Binary states

    # Test invalid hidden_states shape
    try:
        rbm.sample_visible(np.random.randn(4, 2))  # Wrong shape
    except ValueError:
        pass
        
    print("Visible probs shape:", visible_probs.shape)

def test_gibbs_sampling():
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    # Use single hidden vector for new API
    hidden_states = np.random.randn(rbm.hidden_units)
    neg_associations, visible_probs, hidden_probs = rbm.gibbs_sampling(hidden_states, iterations=2)
    assert neg_associations.shape == (rbm.items, rbm.rating_options, rbm.hidden_units)
    assert visible_probs.shape == (rbm.items, rbm.rating_options)
    assert hidden_probs.shape == (rbm.hidden_units,)

    # Test invalid iterations value
    try:
        rbm.gibbs_sampling(hidden_states, iterations=-1)
    except ValueError:
        pass
    
    # Test invalid hidden_states type
    try:
        rbm.gibbs_sampling("invalid", iterations=2)
    except TypeError as e:
        assert str(e) == "hidden_states must be a numpy array."
        
    print("Negative associations shape:", neg_associations.shape)
    
import numpy as np

def test_training_iteration():
    # Create an RBM instance
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    
    # Prepare mock data
    # New API: structured single sample
    data = np.random.randn(rbm.items, rbm.rating_options)
    
    # Test valid training iteration
    nvp, gradients = rbm.training_iteration(data, learning_rate=0.1, epoch=5)
    assert nvp.shape == (rbm.items, rbm.rating_options)
    assert np.all(np.isfinite(nvp))  # Ensure no NaN or inf values in the output

    # # Test invalid input data (wrong shape)
    # try:
    #     rbm.training_iteration(np.random.randn(1, 40), learning_rate=0.1, epoch=5)
    # except ValueError as e:
    #     assert str(e) == f"data must have shape (n_samples, {rbm.visible_units})"
    
    # # Test invalid learning_rate type
    # try:
    #     rbm.training_iteration(data, learning_rate="0.1", epoch=5)
    # except TypeError as e:
    #     assert str(e) == "learning_rate must be a float or int."
    
    # # Test invalid epoch type
    # try:
    #     rbm.training_iteration(data, learning_rate=0.1, epoch="5")
    # except ValueError as e:
    #     assert str(e) == "epoch must be a positive integer."

    # # Test invalid epoch value (negative)
    # try:
    #     rbm.training_iteration(data, learning_rate=0.1, epoch=-5)
    # except ValueError as e:
    #     assert str(e) == "epoch must be a positive integer."

def test_train():
    # Create an RBM instance
    rbm = SingleUserRBM(K=5, m=10, hidden_units=3)
    
    # Prepare mock data
    data = np.random.randn(rbm.items, rbm.rating_options) 
    
    # Test valid training process
    rbm.train(data, learning_rate=0.1, epochs=3)  # Should print error for each epoch
    assert True  # If no exception is raised, the test passes

    # # Test invalid input data (wrong shape)
    # try:
    #     rbm.train(np.random.randn(1, 40), learning_rate=0.1, epochs=3)
    # except ValueError as e:
    #     assert str(e) == f"data must have shape (n_samples, {rbm.visible_units})"
    
    # # Test invalid learning_rate type
    # try:
    #     rbm.train(data, learning_rate="0.1", epochs=3)
    # except TypeError as e:
    #     assert str(e) == "learning_rate must be a float or int."
    
    # # Test invalid epochs type
    # try:
    #     rbm.train(data, learning_rate=0.1, epochs="3")
    # except ValueError as e:
    #     assert str(e) == "epochs must be a positive integer."
    
    # # Test invalid epochs value (negative)
    # try:
    #     rbm.train(data, learning_rate=0.1, epochs=-3)
    # except ValueError as e:
    #     assert str(e) == "epochs must be a positive integer."

    # Test running training for 0 epochs (should not raise an error but should not train)
    rbm.train(data, learning_rate=0.1, epochs=0)
    # No assertion needed here, just ensure no exception is thrown
    assert True  # Training with 0 epochs should not throw errors
    
if __name__ == "__main__":
    test_single_user_rbm_initialization()
    test_set_weights()
    test_sigmoid()
    test_softmax()
    test_sample_hidden()
    test_sample_visible()
    test_gibbs_sampling()
    test_training_iteration()
    test_train()
    print("All tests passed!")