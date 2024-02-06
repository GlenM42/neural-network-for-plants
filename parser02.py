import pandas as pd

# Load the dataset
df = pd.read_csv("output_data.csv")

# Step 1: Clean the Answer column
df['Answer'] = df['Answer'].str.replace('[\[\}\]]', '', regex=True)

# Step 2: Convert all answers to lowercase
df['Answer'] = df['Answer'].str.lower()

# Step 3: Group answers by filenames into a dictionary
answers_dict = df.groupby('Filename')['Answer'].apply(list).to_dict()

print(answers_dict)
print("==============")
