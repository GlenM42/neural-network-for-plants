import pandas as pd

# Load the dataset
df = pd.read_csv("output_data.csv")

# Step 1: Clean the Answer column
df['Answer'] = df['Answer'].str.replace('[\[\}\]]', '', regex=True)

# Step 2: Convert all answers to lowercase
df['Answer'] = df['Answer'].str.lower()

# Step 3: Group answers by filenames into a list
# Step 3.1: Create a DataFrame with unique filenames
unique_filenames = df['Filename'].unique()
filename_df = pd.DataFrame({'Filename': unique_filenames})

# Step 3.2: Group answers by filename into a list
grouped_answers = df.groupby('Filename')['Answer'].apply(list).reset_index()

# Step 3.3: Merge filename_df with grouped_answers
result_df = pd.merge(filename_df, grouped_answers, on='Filename', how='left')

print(result_df)
