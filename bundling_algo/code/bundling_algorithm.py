import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import haversine_distances
from math import radians

# Set a random seed for reproducibility
np.random.seed(42)

# Function to generate random hexanumeric MoveID of length 6
def generate_move_id():
    hexanumeric_characters = '0123456789ABCDEF'
    return ''.join(np.random.choice(list(hexanumeric_characters), size=6))

# Generate synthetic data for 1000 moves
num_moves = 500
cities_germany = ['Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 'Stuttgart', 'Dusseldorf', 'Dortmund']
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

moves_data = pd.DataFrame({
    'MoveID': [generate_move_id() for _ in range(num_moves)],
    'MoveDate': np.random.choice(date_range, size=num_moves),
    'Distance': np.random.uniform(50, 500, size=num_moves),
    'Volume': np.random.uniform(10, 20, size=num_moves),
    'Price': np.random.uniform(500, 2000, size=num_moves),
    'Cost': np.random.uniform(300, 1500, size=num_moves),
})

# Generate synthetic cities in Germany
moves_data['City'] = np.random.choice(cities_germany, size=num_moves)

# Helper function to calculate haversine distance between two points
def haversine_distance(coord1, coord2):
    coord1 = [radians(_) for _ in coord1]
    coord2 = [radians(_) for _ in coord2]
    result = haversine_distances([coord1, coord2])
    return result[0][1] * 6371000  # Radius of Earth in meters

# Generate synthetic coordinates for cities in Germany
city_coordinates_germany = {
    'Berlin': (52.5200, 13.4050),
    'Hamburg': (53.5511, 9.9937),
    'Munich': (48.8566, 2.3522),
    'Cologne': (50.9375, 6.9603),
    'Frankfurt': (50.1109, 8.6821),
    'Stuttgart': (48.7758, 9.1829),
    'Dusseldorf': (51.2277, 6.7735),
    'Dortmund': (51.5136, 7.4653)
}

# Add coordinates to moves_data
moves_data['Coordinates'] = moves_data['City'].map(city_coordinates_germany)

# Bundling algorithm
def bundle_moves(moves_data):
    bundled_moves = []
    total_cost_before = 0
    total_cost_after = 0
    total_bundles = 0
    bundles_per_label = {'co-load move': 0, 'back-load move': 0, 'round-trip move': 0}

    # Cost savings variables
    cost_savings_co_load = 0.3
    cost_savings_back_load = 0.4
    cost_savings_round_trip = 0.5

    def calculate_cost_savings(move_type, total_cost):
        if move_type == "co-load move":
            return cost_savings_co_load * total_cost
        elif move_type == "back-load move":
            return cost_savings_back_load * total_cost
        elif move_type == "round-trip move":
            return cost_savings_round_trip * total_cost
        else:
            return 0

    # Optimize routes from city A to city B
    for _, group in moves_data.groupby(['MoveDate', 'City']):
        if len(group) > 1:  # Check if there are multiple moves on the same day between the same pair of cities
            distances = []
            for move1, move2 in combinations(group['MoveID'], 2):
                coord1 = moves_data.loc[moves_data['MoveID'] == move1, 'Coordinates'].iloc[0]
                coord2 = moves_data.loc[moves_data['MoveID'] == move2, 'Coordinates'].iloc[0]
                distances.append((move1, move2, haversine_distance(coord1, coord2)))

            if distances:  # Check if distances list is not empty
                distances_df = pd.DataFrame(distances, columns=['MoveID1', 'MoveID2', 'Distance'])
                sorted_distances = distances_df.sort_values(by='Distance')
                move_id_1 = sorted_distances.iloc[0]['MoveID1']
                move_id_2 = sorted_distances.iloc[0]['MoveID2']

                cost_before = moves_data.loc[moves_data['MoveID'] == move_id_1, 'Cost'].iloc[0] + \
                               moves_data.loc[moves_data['MoveID'] == move_id_2, 'Cost'].iloc[0]

                bundled_moves.append([move_id_1, move_id_2])

                cost_after = moves_data.loc[moves_data['MoveID'] == move_id_1, 'Cost'].iloc[0] + \
                             moves_data.loc[moves_data['MoveID'] == move_id_2, 'Cost'].iloc[0]

                total_cost_before += cost_before
                total_cost_after += cost_after

                move_type = 'co-load move'
                bundles_per_label[move_type] += 1
                total_bundles += 1
                print(f"Bundle {total_bundles} - {[move_id_1, move_id_2]}")
                print(f"Type: {move_type}")
                print(f"Cost Savings: {calculate_cost_savings(move_type, total_cost_before)}\n")

    # Identify and bundle back-load moves
    for _, group in moves_data.groupby('City'):
        distances = []
        for move1, move2 in combinations(group['MoveID'], 2):
            coord1 = moves_data.loc[moves_data['MoveID'] == move1, 'Coordinates'].iloc[0]
            coord2 = moves_data.loc[moves_data['MoveID'] == move2, 'Coordinates'].iloc[0]
            distances.append((move1, move2, haversine_distance(coord1, coord2)))

        if distances:  # Check if distances list is not empty
            distances_df = pd.DataFrame(distances, columns=['MoveID1', 'MoveID2', 'Distance'])
            sorted_distances = distances_df.sort_values(by='Distance')
            move_id_1 = sorted_distances.iloc[0]['MoveID1']
            move_id_2 = sorted_distances.iloc[0]['MoveID2']

            cost_before = moves_data.loc[moves_data['MoveID'] == move_id_1, 'Cost'].iloc[0] + \
                           moves_data.loc[moves_data['MoveID'] == move_id_2, 'Cost'].iloc[0]

            bundled_moves.append([move_id_1, move_id_2])

            cost_after = moves_data.loc[moves_data['MoveID'] == move_id_1, 'Cost'].iloc[0] + \
                         moves_data.loc[moves_data['MoveID'] == move_id_2, 'Cost'].iloc[0]

            total_cost_before += cost_before
            total_cost_after += cost_after

            move_type = 'back-load move'
            bundles_per_label[move_type] += 1
            total_bundles += 1
            print(f"Bundle {total_bundles} - {[move_id_1, move_id_2]}")
            print(f"Type: {move_type}")
            print(f"Cost Savings: {calculate_cost_savings(move_type, total_cost_before)}\n")

    # Identify and bundle round-trip moves
    round_trip_moves = moves_data.groupby('City')['MoveID'].apply(list).reset_index()
    for _, group in round_trip_moves.iterrows():
        if len(group['MoveID']) >= 3:
            bundled_moves.append(group['MoveID'][:5])  # Keep at most 5 moves in a round trip bundle
            total_cost_before += sum(moves_data.loc[moves_data['MoveID'].isin(group['MoveID']), 'Cost'])
            total_cost_after += sum(moves_data.loc[moves_data['MoveID'] == group['MoveID'][0], 'Cost'])

            move_type = 'round-trip move'
            bundles_per_label[move_type] += 1
            total_bundles += 1
            print(f"Bundle {total_bundles} - {group['MoveID'][:5]}")
            print(f"Type: {move_type}")
            print(f"Cost Savings: {calculate_cost_savings(move_type, total_cost_before)}\n")

    print("Total Cost Before Optimization:", total_cost_before)
    print("Total Cost After Optimization:", total_cost_after)
    print("Total Bundles:", total_bundles)
    print("Bundles per Label:", bundles_per_label)

    # Additional bundling logic can be added for other types of bundling

    return bundled_moves

# Function to find bundling combinations for a specific MoveID
def find_bundling_combinations(moves_data, target_move_id):
    combinations_found = False

    print(f"Searching for bundling combinations for MoveID: {target_move_id}")

    for move_id in moves_data['MoveID']:
        if move_id != target_move_id:
            coord1 = moves_data.loc[moves_data['MoveID'] == target_move_id, ['Latitude', 'Longitude']].iloc[0].tolist()
            coord2 = moves_data.loc[moves_data['MoveID'] == move_id, ['Latitude', 'Longitude']].iloc[0].tolist()
            distance = haversine_distance(coord1, coord2)

            if distance <= 3:  # Adjust as needed
                print(f"Potential bundling combination found with MoveID: {move_id}")
                combinations_found = True

    if not combinations_found:
        print("No bundling possibility found for the given MoveID.")

# Example to find bundling combinations for a specific MoveID
target_move_id = 'ABC123'  # Change to a valid MoveID for testing
find_bundling_combinations(moves_data, target_move_id)

# Run the bundling algorithm
bundled_moves = bundle_moves(moves_data)

# Display bundled moves
print("\nFinal Bundled Moves:")
for bundle in bundled_moves:
    print(bundle)
