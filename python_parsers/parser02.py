import pandas as pd

# Load the dataset
df = pd.read_csv("pieces_of_datasets/output_data.csv")

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

# Remove "Copy of" from the filenames
result_df['Filename'] = result_df['Filename'].str.replace('^Copy of ', '', regex=True)

# Group answers by filename into a list again
grouped_answers = result_df.groupby('Filename')['Answer'].sum().reset_index()

# Resetting index to avoid multi-index
grouped_answers = grouped_answers.reset_index(drop=True)

print(grouped_answers)

# An attempt at getting rid of the 100 copies in the grouped_answer

plant_images_df = pd.read_csv("pieces_of_datasets/plant_images.csv")
column_names = ['image_id', 'metadata']
new_df = plant_images_df[column_names]
# print(new_df)

