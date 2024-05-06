import os
import zipfile
from zipfile import ZipFile
from google.colab import files

# Upload your Kaggle API key (kaggle.json) using the Colab interface
files.upload()

# Move the uploaded key to the correct location
!mkdir ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

from kaggle.api.kaggle_api_extended import KaggleApi

# Instantiate the Kaggle API client
api = KaggleApi()
api.authenticate()

# Define the directory path where you want to download the dataset
download_dir = "/content"

# Download the dataset into the specified directory
api.dataset_download_files(dataset="sujaykapadnis/crossword-puzzles-and-clues", path=download_dir, unzip=True)

import pandas as pd

# File paths for the two CSV files
file1_path = '/content/times.csv'
file2_path = '/content/big_dave.csv'

# Read the CSV files into pandas DataFrames
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Union the two DataFrames
union_df = pd.concat([df1, df2], ignore_index=True)

# Remove duplicates from the union DataFrame
data = union_df.drop_duplicates()

# Save the merged DataFrame to a CSV file
output_file = '/content/data.csv'
data.to_csv(output_file, index=False)

# Display the union DataFrame
data.head()
