import matplotlib.pyplot as plt

def plot_errors(test_errors, train_errors):
    """
    Plots the total error and test error over iterations.
    
    Args:
        test_errors (list): List of test errors (numeric values).
        train_errors (list): List of train errors (numeric values).
    """
    # Convert numeric values to floats for plotting
    test_errors_np = [error.item() if hasattr(error, 'item') else float(error) for error in test_errors]
    train_errors_np = [error.item() if hasattr(error, 'item') else float(error) for error in train_errors]
    
    # Create a range for iterations
    iterations = range(1, len(train_errors) + 1)

    plt.plot(iterations, train_errors_np, label='Train Error', marker='s', linestyle=':')
    # Add labels, title, and legend
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('RMSE Error Over Iterations of Training Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot errors
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, test_errors_np, label='Test Error', marker='x', linestyle='--')
    
    # Add labels, title, and legend
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()