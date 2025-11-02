import random
import pandas as pd
from pathlib import Path


def generate_synthetic_data(num_people=30, num_dishes=20, ratings_fraction=0.2):
    ratings_per_person = int(num_dishes * ratings_fraction)
    data = []
    for person_id in range(1, num_people + 1):
        rated_dishes = random.sample(range(1, num_dishes + 1), ratings_per_person)
        for dish_id in rated_dishes:
            rating = random.randint(1, 5)
            data.append({'Person': person_id, 'Dish': dish_id, 'Rating': rating})
    return pd.DataFrame(data)


def _default_output_path():
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / 'data' / 'preprocessed'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / 'dish_ratings.csv'


if __name__ == '__main__':
    df = generate_synthetic_data()
    print(df.head())
    out_path = _default_output_path()
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic data to {out_path}")
