import pandas as pd

file_path = 'data/hdf5/LOB.h5'

dataset_key = 'lob'

# Load the DataFrame from the HDF5 file
df = pd.read_hdf(file_path, key=dataset_key)

# Convert 'type' back to 'bid' and 'ask' 
df['type'] = df['type'].map({0: 'bid', 1: 'ask'})

print(df.head())  
