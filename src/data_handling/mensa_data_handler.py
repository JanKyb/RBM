import pandas as pd
import numpy as np
from pathlib import Path

# Function to sort the CSV by email (second column) and then by the third column,
# and also identify unique emails and the largest number in the third column.
def sort_csv_by_email_and_column(file_path):
    """
    Process the CSV file to extract unique emails, sort data, count unique emails, 
    find the largest food ID, and generate a numpy array of ratings.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        dict: A dictionary containing unique emails, sorted data, and the email indices array.
    """
    # Accept either Path or str
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    data = pd.read_csv(file_path)
    
    # Extract unique emails and sort by the second column (email) and third column (meal ID)
    unique_emails_data = data.drop_duplicates(subset=[data.columns[1]])
    sorted_data = unique_emails_data.sort_values(by=[data.columns[1], data.columns[2]], ascending=[True, True])
    
    # Count unique emails and find the maximum meal ID
    unique_emails_count = unique_emails_data[data.columns[1]].nunique()
    max_food_id = data[data.columns[2]].max()  # Get the max meal ID from the full dataset
    
    # Extract unique emails into a list
    unique_emails_list = unique_emails_data[data.columns[1]].sort_values().unique().tolist()
    
    # Generate a numpy array with rows for unique emails and columns for meal IDs, initialized to zeros
    email_indices_array = np.zeros((unique_emails_count, max_food_id), dtype=float)
    
    # Map emails to their indices
    email_to_index = {email: idx for idx, email in enumerate(unique_emails_list)}
    
    # Populate the numpy array with ratings
    for _, row in data.iterrows():
        email = row[data.columns[1]]
        meal_id = int(row[data.columns[2]])  # Ensure meal ID is an integer
        rating = int(row[data.columns[3]] )      # Assuming the rating is in the 4th column
        
        # Get the index of the email and write the rating into the numpy array
        email_index = email_to_index[email]
        email_indices_array[email_index, meal_id - 1] = rating  # Use meal_id - 1 for zero-based indexing
        
    print("Unique emails count:", unique_emails_count)
    
    # Return the numpy array of ratings (rows: unique emails, cols: meal ids)
    # and the list of unique emails in the same row order
    return email_indices_array, unique_emails_list
    
def convert_array_to_list_of_lists(email_indices_array):
    """
    Convert the NumPy array of ratings to a list of lists of lists format.
    
    Parameters:
        email_indices_array (np.ndarray): A 2D NumPy array of ratings.
        
    Returns:
        list: A list of lists of lists representation of the ratings.
    """
    # Get the maximum rating value to determine the length of rating lists
    max_rating = 5  # NaN-safe maximum value # Ensure at least length 1 for empty arrays
    
    # Initialize the result as a list of lists
    ratings_list = []
    
    # Iterate over rows (emails) in the array
    for row in email_indices_array:
        user_ratings = []
        for rating in row:
            if rating == 0:  # Empty rating
                user_ratings.append([])
            else:  # Convert rating to one-hot representation
                rating_list = [0] * max_rating
                rating_list[int(rating) - 1] = 1
                user_ratings.append(rating_list)
        ratings_list.append(user_ratings)
    
    return ratings_list


def _default_mensa_csv_path():
    """Return the default path to the mensa CSV inside the repository `data/test` folder.

    This resolves the repository root relative to this file (two parents up: project root)
    and returns data/test/mensa_bot.csv. Adjust if your data lives elsewhere.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "test" / "mensa_bot.csv"


if __name__ == "__main__":
    # Example usage when executed directly. Uses a repo-relative default path.
    default_path = _default_mensa_csv_path()
    try:
        rating_array = sort_csv_by_email_and_column(default_path)
    except FileNotFoundError:
        print(f"Default CSV not found at {default_path}. Please provide a valid file path.")
    else:
        final_list = convert_array_to_list_of_lists(rating_array)
        print(f"Converted ratings for {len(final_list)} users.")

