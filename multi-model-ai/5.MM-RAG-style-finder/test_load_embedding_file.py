# This file is just to load data in embedding file to inspect.

import pandas as pd
import os

file_path = "swift-style-embeddings.pkl" # 192 images of taylor swift, embedded into vectors

try:
    if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Open the file in binary read mode ('rb')
    data = pd.read_pickle(file_path)

    # Now 'data' contains the deserialized Python object.
    # You can print it or use it as needed.
    print("Data successfully loaded:")
    print(data)
    # print(data["Image"][0])

# except FileNotFoundError:
#     print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
