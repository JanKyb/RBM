import numpy as np
from src.rbm.global_rbm import GlobalRBM
from src.rbm.single_rbm import SingleUserRBM

data1 = [[0, 0, 1, 0, 0], [], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [], [0, 0, 0, 1, 0], [], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], []]
data2 = [[0, 0, 1, 0, 0], [0,0,0,0,1], [], [0, 0, 0, 0, 1], [], [0, 1, 0, 0, 0], [], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], []]
data3 = [[0, 0, 1, 0, 0], [1,0,0,0,0], [], [1, 0, 0, 0, 0], [], [0, 0, 0, 1, 0], [], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], []]

data_test = [data1, data2, data3]

# Test 1: test initialization of the globalRBM object
def test_initialization():
    num_users = 3
    K = 5
    total_items = 10
    hidden_units = 3
    globalRBM = GlobalRBM(num_users, K, total_items, hidden_units)
    
    # Check initial values
    assert globalRBM.users == num_users
    assert globalRBM.rating_options == K
    assert globalRBM.items == total_items
    assert globalRBM.hidden_units == hidden_units
    assert globalRBM.weights.shape == (total_items, K, hidden_units)
    assert globalRBM.bias_visible.shape == (total_items, K)
    assert globalRBM.bias_hidden.shape == (hidden_units,)

# Test 2: test set_user_data
def test_set_user_data():
    num_users = 3
    K = 5
    total_items = 10
    hidden_units = 3
    globalRBM = GlobalRBM(num_users, K, total_items, hidden_units)
    
    valid_data = data_test.copy()
    
    # Test setting valid data
    globalRBM.set_user_data(valid_data)
    # filtered_user_data is a list with one entry per user
    assert len(globalRBM.filtered_user_data) == len(valid_data)

    # Test invalid data type
    try:
        globalRBM.set_user_data("invalid_type")
        assert False, "Expected TypeError for invalid data type"
    except TypeError:
        pass

    # Test invalid shape
    invalid_data = [np.random.rand(10, 10), np.random.rand(5, 5)]
    try:
        globalRBM.set_user_data(invalid_data)
        assert False, "Expected ValueError for mismatched data dimensions"
    except ValueError:
        pass

# Test 3: test set_test_data
def test_set_test_data():
    num_users = 3
    K = 5
    total_items = 10
    hidden_units = 3
    globalRBM = GlobalRBM(num_users, K, total_items, hidden_units)
    
    valid_test_data = data_test
    
    # Test setting valid test data
    globalRBM.set_user_data(valid_test_data)  # Set training data first
    globalRBM.set_test_data(valid_test_data)
    assert len(globalRBM.test_data) == len(valid_test_data)
    
    #Test case when training data is not set
    globalRBM_no_train = GlobalRBM(num_users, K, total_items, hidden_units)
    try:
        globalRBM_no_train.set_test_data(valid_test_data)
        assert False, "Expected ValueError for setting test data before training data"
    except ValueError:
        pass

    # Test invalid data type
    try:
        globalRBM.set_test_data("invalid_type")
        assert False, "Expected TypeError for invalid test data type"
    except TypeError:
        pass

# Test 4: test initialize_RBMs method
def test_initialize_rbms():
    num_users = 3
    K = 5
    total_items = 10
    hidden_units = 3
    globalRBM = GlobalRBM(num_users, K, total_items, hidden_units)

    valid_data = data_test.copy()
    globalRBM.set_user_data(valid_data)
    
    globalRBM.initialize_RBMs()
    assert len(globalRBM.rbm_list) == num_users
    
    # Test without user data
    empty_rbm = GlobalRBM(num_users, K, total_items, hidden_units)
    try:
        empty_rbm.initialize_RBMs()
        assert False, "Expected ValueError for no user data"
    except ValueError:
        pass

# Test 5: test training_iteration method
def test_training_iteration():
    num_users = 3
    K = 5
    total_items = 10
    hidden_units = 3
    globalRBM = GlobalRBM(num_users, K, total_items, hidden_units)

    valid_data = data_test.copy()
    globalRBM.set_user_data(valid_data)
    globalRBM.set_test_data(valid_data)
    globalRBM.initialize_RBMs()


    globalRBM.train(epochs=1, learning_rate=0.01)

# Test 6: test train method with valid input
def test_train_valid():
    num_users = 3
    K = 5
    total_items = 10
    hidden_units = 3
    globalRBM = GlobalRBM(num_users, K, total_items, hidden_units)

    valid_data = data_test.copy()
    globalRBM.set_user_data(valid_data)
    globalRBM.set_test_data(valid_data)
    globalRBM.initialize_RBMs()

    globalRBM.train(epochs=45, learning_rate=0.01)

# Test 7: test predict method
def test_predict():
    num_users = 3
    K = 5
    total_items = 10
    hidden_units = 3
    globalRBM = GlobalRBM(num_users, K, total_items, hidden_units)

    valid_data = data_test.copy()
    globalRBM.set_user_data(valid_data)
    globalRBM.set_test_data(valid_data)
    globalRBM.initialize_RBMs()

    visible_probs, visible_states = globalRBM.predict(0)
    # New API: visible_probs is (items, rating_options)
    assert visible_probs.shape == (total_items, K)
    assert visible_states.shape == (total_items, K)

if __name__ == "__main__":
    # Run the individual tests first
    test_initialization()
    test_set_user_data()
    test_set_test_data()
    test_initialize_rbms()
    test_training_iteration()
    test_train_valid()
    test_predict()

    # Run the full functionality test (train method together with other methods)
    print("All tests passed.")