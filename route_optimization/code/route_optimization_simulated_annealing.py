!pip install googlemapsimport numpy as np
import random
import math
import googlemaps
import folium
from datetime import datetime
from folium.plugins import FloatImage

# Read API key from file or environment variable
def read_api_key(filename='api_key.txt'):
    try:
        with open(filename, 'r') as f:
            api_key = f.read().strip()
        return api_key
    except FileNotFoundError:
        print(f"API key file '{filename}' not found.")
        return None

api_key = read_api_key()  # Read API key from file
if api_key is None:
    exit()

# Define delivery locations with real names and addresses in Germany
delivery_locations = {
    'Berlin, Germany': 'Berlin, Germany',
    'Hamburg, Germany': 'Hamburg, Germany',
    'Munich, Germany': 'Munich, Germany',
    'Cologne, Germany': 'Cologne, Germany',
    'Frankfurt, Germany': 'Frankfurt, Germany',
    'Stuttgart, Germany': 'Stuttgart, Germany',
    'Düsseldorf, Germany': 'Düsseldorf, Germany',
    'Dortmund, Germany': 'Dortmund, Germany',
    'Essen, Germany': 'Essen, Germany',
    'Leipzig, Germany': 'Leipzig, Germany',
    'Nuremberg, Germany': 'Nuremberg, Germany',
    'Dresden, Germany': 'Dresden, Germany',
    'Bremen, Germany': 'Bremen, Germany',
    'Hanover, Germany': 'Hanover, Germany',
    'Duisburg, Germany': 'Duisburg, Germany',
    'Bochum, Germany': 'Bochum, Germany',
    'Wuppertal, Germany': 'Wuppertal, Germany',
    'Bielefeld, Germany': 'Bielefeld, Germany',
    'Bonn, Germany': 'Bonn, Germany',
    'Münster, Germany': 'Münster, Germany',
}

# Fetch coordinates using Google Maps Geocoding API
def fetch_coordinates(locations, api_key):
    gmaps = googlemaps.Client(key=api_key)
    coordinates = {}
    for loc_name, address in locations.items():
        try:
            result = gmaps.geocode(address)
            if result:
                coords = result[0]['geometry']['location']
                coordinates[loc_name] = (coords['lat'], coords['lng'])
            else:
                print(f"Could not fetch coordinates for {loc_name}")
                coordinates[loc_name] = None
        except Exception as e:
            print(f"Error fetching coordinates for {loc_name}: {str(e)}")
            coordinates[loc_name] = None
    return coordinates

# Fetch coordinates for delivery locations
coordinates = fetch_coordinates(delivery_locations, api_key)

# Print coordinates for debugging
print("Coordinates:")
print(coordinates)

# Fetch distances using Google Maps API and construct distance matrix
def fetch_distances(coordinates, api_key):
    gmaps = googlemaps.Client(key=api_key)
    num_locations = len(coordinates)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i, (loc1, coord1) in enumerate(coordinates.items()):
        for j, (loc2, coord2) in enumerate(coordinates.items()):
            if i != j:
                try:
                    result = gmaps.distance_matrix(coord1, coord2, mode='driving', departure_time=datetime.now())
                    distance_matrix[i][j] = result['rows'][0]['elements'][0]['distance']['value'] / 1000  # Convert to kilometers
                except Exception as e:
                    print(f"Error fetching distance between {loc1} and {loc2}: {str(e)}")
                    distance_matrix[i][j] = np.inf
    return distance_matrix

# Fetch distances and construct distance matrix
distance_matrix = fetch_distances(coordinates, api_key)

# Print distance matrix for debugging
print("Distance Matrix:")
print(distance_matrix)

# Simulated Annealing Algorithm
def simulated_annealing(distance_matrix):
    current_route = list(range(len(delivery_locations)))  # Initial route as indices
    random.shuffle(current_route)  # Randomly shuffle initial route
    current_distance = total_distance(current_route, distance_matrix)
    best_route = current_route.copy()
    best_distance = current_distance
    temperature = 1000
    cooling_rate = 0.003

    while temperature > 1:
        new_route = generate_neighbor(current_route)
        new_distance = total_distance(new_route, distance_matrix)
        if acceptance_probability(current_distance, new_distance, temperature) > random.random():
            current_route = new_route
            current_distance = new_distance
            if current_distance < best_distance:
                best_route = current_route.copy()
                best_distance = current_distance
        temperature *= 1 - cooling_rate

    return best_route, best_distance

# Helper functions
def total_distance(route, distance_matrix):
    total_dist = 0
    for i in range(len(route) - 1):
        total_dist += distance_matrix[route[i]][route[i+1]]
    return total_dist

def acceptance_probability(old_dist, new_dist, temperature):
    if new_dist < old_dist:
        return 1
    return math.exp((old_dist - new_dist) / temperature)

def generate_neighbor(route):
    neighbor = route.copy()
    if len(route) > 2:
        i, j = random.sample(range(1, len(route) - 1), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    elif len(route) == 2:
        neighbor[1], neighbor[0] = neighbor[0], neighbor[1]  # Swap the only two locations
    return neighbor

# Optimize route using simulated annealing
best_route, best_distance = simulated_annealing(distance_matrix)

# Print optimized route and distance
print("Optimized route:", [list(delivery_locations.keys())[i] for i in best_route])
print("Total distance:", best_distance)

# Create an interactive map
m = folium.Map(location=list(coordinates.values())[0], zoom_start=6)

# Add markers for delivery locations
for location, coords in coordinates.items():
    folium.Marker(location=coords, popup=location).add_to(m)

# Add polyline for the optimized route with arrows
route_coords = [list(coordinates.values())[i] for i in best_route]
for i in range(len(route_coords) - 1):
    folium.Marker(location=route_coords[i], icon=folium.Icon(color='green')).add_to(m)
    folium.PolyLine(locations=[route_coords[i], route_coords[i+1]], color='blue').add_to(m)
    # Calculate arrow position
    arrow_pos = [(route_coords[i][0] + route_coords[i+1][0]) / 2, (route_coords[i][1] + route_coords[i+1][1]) / 2]
    folium.RegularPolygonMarker(location=arrow_pos, fill_color='red', number_of_sides=3, radius=10,
                                rotation=math.degrees(math.atan2(route_coords[i+1][1]-route_coords[i][1],
                                                                route_coords[i+1][0]-route_coords[i][0]))).add_to(m)

# Add marker and polyline for the last location
folium.Marker(location=route_coords[-1], icon=folium.Icon(color='green')).add_to(m)

# Display map in Google Colab
m
