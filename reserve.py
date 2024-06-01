import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('reservation_data.csv')

df.loc[28, 'reservation_success'] = 'no'

# Write the DataFrame back to the CSV file
df.to_csv('reservation_data.csv', index=False)
