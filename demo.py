#!/usr/bin/env python3
"""Run a Global RBM on mensa ratings and allow simple interactive predictions.

Usage:
    python run_global_rbm.py [--csv PATH] [--hidden N] [--epochs E]

If no CSV is provided the script will look for the default path configured in
`src.rbm.config.DATA_TEST_DIR / 'mensa_bot.csv'`.

The script:
- Loads the mensa CSV using the mensa data handler
- Converts the rating matrix to the project's list-of-lists format
- Creates and trains a GlobalRBM
- Prompts the user to pick an email (or index) and a meal id to see predicted ratings

This is a simple demo harness for development and experimentation.
"""
from pathlib import Path
import sys

from src.data_handling.mensa_data_handler import sort_csv_by_email_and_column, convert_array_to_list_of_lists
from src.rbm.global_rbm import GlobalRBM
from src.rbm import config
import numpy as np


# This demo reads configuration values directly from `src.rbm.config`.
# If you want to override any value for quick experimentation, edit the
# variables below or change the values in the config module.

# Optional override values (uncomment to change):
# CSV_OVERRIDE = Path('/path/to/your/mensa_bot.csv')
# HIDDEN_OVERRIDE = 128
# EPOCHS_OVERRIDE = 100
# LR_OVERRIDE = 0.01

# Resolve demo parameters from config (or overrides above)
CSV_OVERRIDE = None
HIDDEN_OVERRIDE = None
EPOCHS_OVERRIDE = None
LR_OVERRIDE = None


def main():
    # Resolve CSV path (use override if provided, otherwise config default)
    csv_path = CSV_OVERRIDE or (config.DATA_TEST_DIR / "mensa_bot.csv")
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    # Load and transform data
    rating_array, emails = sort_csv_by_email_and_column(csv_path)
    # rating_array shape: (num_users, num_meals) with integers (0 meaning missing)
    ratings_list = convert_array_to_list_of_lists(rating_array)

    num_users = len(ratings_list)
    total_meals = rating_array.shape[1]
    K = config.DEFAULT_RATING_OPTIONS
    hidden_units = HIDDEN_OVERRIDE or config.DEFAULT_HIDDEN_UNITS

    print(f"Loaded {num_users} users, {total_meals} meals, rating depth K={K}, hidden_units={hidden_units}")

    # Create GlobalRBM
    grbm = GlobalRBM(num_users=num_users, K=K, m=total_meals, hidden_units=hidden_units)

    # For this demo we'll use the same data for training and testing (no held-out set).
    grbm.set_user_data(ratings_list)
    grbm.set_test_data(ratings_list)
    grbm.initialize_RBMs()

    epochs = EPOCHS_OVERRIDE or config.DEFAULT_EPOCHS
    learning_rate = LR_OVERRIDE or config.LEARNING_RATE

    print("Starting training... (this may take a while)")
    grbm.train(epochs=epochs, learning_rate=learning_rate)
    print("Training finished.")

    # Interactive prediction loop
    while True:
        print("\nUsers (index: email):")
        for i, e in enumerate(emails):
            print(f"  {i}: {e}")
        sel = input("Enter user index (or email) to predict for, or 'q' to quit: ").strip()
        if sel.lower() in ("q", "quit", "exit"):
            break

        # Resolve selection to index
        try:
            user_idx = int(sel)
            if user_idx < 0 or user_idx >= num_users:
                print("Invalid user index")
                continue
        except ValueError:
            # treat as email lookup (case-insensitive)
            matches = [i for i, e in enumerate(emails) if sel.lower() in e.lower()]
            if not matches:
                print("No matching emails found")
                continue
            if len(matches) > 1:
                print("Multiple matches found, please be more specific:")
                for i in matches:
                    print(f"  {i}: {emails[i]}")
                continue
            user_idx = matches[0]

        meal_sel = input(f"Enter meal id (1-{total_meals}) to predict for, or 'a' to show all: ").strip()
        visible_probs, visible_states = grbm.predict(user_idx)
        # visible_probs shape: (items, K)
        if meal_sel.lower() in ("a", "all"):
            print("Predicted rating probabilities for each meal (showing argmax rating):")
            for meal_id in range(total_meals):
                probs = visible_probs[meal_id]
                pred_rating = int(np.argmax(probs) + 1) 
                #round the rating to 2 decimal places
                pred_rating = round(pred_rating, 2)
                print(f" Meal {meal_id+1}: predicted rating {pred_rating} with probs {np.round(probs,3)}")
            continue

        try:
            meal_id = int(meal_sel)
            if meal_id < 1 or meal_id > total_meals:
                print("Meal id out of range")
                continue
        except ValueError:
            print("Invalid meal id")
            continue

        probs = visible_probs[meal_id - 1]
        pred_rating = int(np.argmax(probs) + 1)
        print(f"Predicted rating for user {emails[user_idx]} on meal {meal_id}: {pred_rating}")
        print(f"Probability distribution (ratings 1..{K}): {np.round(probs, 4)}")


if __name__ == "__main__":
    main()
