import csv
import numpy as np
import random
from pathlib import Path

random.seed(0)

# Funktion zum Konvertieren von Rating in 5-Bit-Darstellung
def rating_to_bits(rating):
    bits = [0] * 5
    if 1 <= rating <= 5:
        bits[rating - 1] = 1
    return bits

# Datei einlesen und Ratings extrahieren
def process_csv(file_path):
    all_ratings = []
    # Accept Path or str
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    # CSV-Datei öffnen
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)  # Überspringt die Kopfzeile

        for row in reader:
            # Ratings extrahieren (alle Werte ab Spalte 2)
            ratings = row[1:]
            ratings = [int(r) for r in ratings if r.isdigit()]  # Nur Zahlen berücksichtigen
            # Ratings in 5-Bit-Darstellung konvertieren
            bits = [rating_to_bits(r) for r in ratings]
            if bits != []:
                all_ratings.append(bits)

    # Umwandeln in ein dreidimensionales Array
    #all_ratings = np.array(all_ratings)  # (Benutzer, Bewertungen, Bits)
    return all_ratings

def randomly_delete_ratings(data, x):
    """
    Randomly deletes x ratings for each user in the dataset.
    Replaces the deleted ratings with an empty list.
    
    Args:
        data (list): List of lists of ratings for each user.
        x (int): Number of ratings to delete per user.
        
    Returns:
        list: Updated dataset with x ratings deleted per user.
    """
    updated_data = []
    
    for user_ratings in data:
        # Get the indices of all ratings
        indices = list(range(len(user_ratings)))
        # Randomly select x indices to delete
        indices_to_delete = random.sample(indices, min(x, len(indices)))
        
        # Replace selected ratings with empty lists
        updated_user_ratings = [
            [] if i in indices_to_delete else rating 
            for i, rating in enumerate(user_ratings)
        ]
        updated_data.append(updated_user_ratings)
    
    return updated_data

def _default_film_csv_path():
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "test" / "FilmTestAndValidation.csv"


if __name__ == "__main__":
    file_path = _default_film_csv_path()
    try:
        binary_ratings = process_csv(file_path)
    except FileNotFoundError:
        print(f"Default CSV not found at {file_path}. Please provide a valid file path.")
    else:
        print("Dreidimensionale Struktur der Ratings (Benutzer x Bewertungen x Bits):")
        print(binary_ratings)
        updated_data = randomly_delete_ratings(binary_ratings, 2)
        print(updated_data)