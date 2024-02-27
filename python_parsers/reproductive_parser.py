import pandas as pd
import csv
import json
from statistics import mode
from collections import defaultdict


# Function to parse the line into columns, handling JSON properly
def parse_line(line):
    columns = []
    inside_json = False
    current_column = ''
    open_braces_count = 0

    for char in line:
        if char == ',' and not inside_json:
            columns.append(current_column.strip())
            current_column = ''
        else:
            current_column += char
            if char == '{':
                inside_json = True
                open_braces_count += 1
            elif char == '}':
                open_braces_count -= 1
                if open_braces_count == 0:
                    inside_json = False

    columns.append(current_column.strip())
    return columns


def check_filename(row):
    if "Filename" not in row[2]:
        return row[2]
    else:
        return row[3]


def extract_filename(json_str):
    try:
        data = json.loads(json_str)
        return data[list(data.keys())[0]]['Filename']
    except (json.JSONDecodeError, KeyError):
        return None


# Read the CSV file and parse each line
parsed_data = []
with open('pieces_of_datasets/Reproductive.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\n')
    for row in reader:
        line = row[0]
        columns = parse_line(line)
        parsed_data.append(columns)

# Convert parsed data into a DataFrame
df = pd.DataFrame(parsed_data)
pd.set_option('display.max_columns', None)

# print(df)

print("\n==============\n")

# Apply the function to create a new column based on the condition
df[3] = df.apply(check_filename, axis=1)

clean_df = pd.DataFrame()
# Apply the function to the second column
clean_df['Filename'] = df[1].apply(extract_filename)
clean_df['Answer'] = df[3]

# Assuming df is your DataFrame containing the parsed data
# clean_df.to_csv("output_data.csv", index=False)
# THIS IS PERFORMED; NO NEED TO RUN SO FAR

print("\n==============\n")

df = pd.read_csv("pieces_of_datasets/output_data.csv", dtype=str, low_memory=False)

unique_answers = df['Answer'].unique()

# Print unique values
print(unique_answers)

# Fix capitalization
df['Answer'] = df['Answer'].str.lower()
# Remove 'Female and Male' part from the values inside lists in the "Answer" column
df['Answer'] = df['Answer'].str.replace('female and male', '', regex=False)
# Remove leading and trailing whitespace from values in the "Answer" column
df['Answer'] = df['Answer'].str.strip()

print("\n==============\n")

unique_answers = df['Answer'].unique()

# Print unique values
print(unique_answers)

print("\n======ALMOST=READY========\n")

grouped_df = df.groupby('Filename').agg(list)

# Reset index to make "Filename" a regular column instead of an index
grouped_df.reset_index(inplace=True)

# Print the grouped DataFrame
print(grouped_df)

print("\n======FIRST=APPROACH=JUST=MODE========\n")

# Create a new column to store the most frequent answer
grouped_df['Most_Frequent_Answer'] = grouped_df['Answer'].apply(lambda x: mode(x) if len(x) > 0 else None)

# Print the DataFrame with the most frequent answer
print(grouped_df)

print("\n=====SECOND=APPROACH=DOING=DICTIONARIES=======\n")

# Initialize an empty list to store dictionaries of probabilities
probabilities_list = []

# Iterate over each row in the DataFrame
for index, row in grouped_df.iterrows():
    # Initialize a dictionary to store probabilities for the current row
    probabilities = defaultdict(float)

    # Calculate probabilities for each answer in the current row
    total_answers = len(row['Answer'])
    if total_answers > 0:
        for answer in row['Answer']:
            probabilities[answer] += 1 / total_answers

    # Round probabilities to 3 digits after the decimal point
    rounded_probabilities = {key: round(value, 3) for key, value in probabilities.items()}

    # Append rounded probabilities to the list
    probabilities_list.append(rounded_probabilities)

# Create a new column to store the dictionaries of probabilities
grouped_df['Answer_Probabilities'] = probabilities_list

# Print the DataFrame to verify the changes
print(grouped_df)

print("\n====ADD=LINKS=AND=IMAGE=ID==========\n")

data_images_df = pd.read_csv('pieces_of_datasets/plant_images_cleaned.csv')
print(data_images_df)

# Merge the datasets on the 'Filename' and 'image_name' columns
combined_df = pd.merge(grouped_df, data_images_df, left_on='Filename', right_on='image_name', how='left')

# Drop the redundant 'image_name' column
combined_df.drop(columns=['image_name'], inplace=True)
# combined_df.drop(columns=['Answer'], inplace=True)

# Print the combined DataFrame
print("\n=====SHOW=THIS=TO=EVERYBODY=========\n")
print(combined_df)
combined_df.to_csv("reproductive_almost_rtu.csv", index=False)

print("\n=====MERGING=EVERYTHING=======\n")

# Group the combined dataset by the 'image_id'
grouped_by_image_id = combined_df.groupby('image_id')

# Initialize lists to store the new data
new_filenames = []
new_locations = []
new_combined_answers = []

# Iterate over each group
for image_id, group in grouped_by_image_id:
    # Concatenate 'Filename' and 'locations'
    new_filenames.append(group['Filename'].iloc[0])
    new_locations.append(group['locations'].iloc[0])

    # Merge lists from 'Answer'
    merged_answers = [item for sublist in group['Answer'] for item in sublist]

    # Store the combined answers
    new_combined_answers.append(merged_answers)

# Create a new DataFrame with the updated information
new_df = pd.DataFrame({
    'Filename': new_filenames,
    'locations': new_locations,
    'Combined_Answers': new_combined_answers
})

# Initialize an empty list to store dictionaries of probabilities
probabilities_list = []

# Iterate over each row in the DataFrame
for index, row in new_df.iterrows():
    # Initialize a dictionary to store probabilities for the current row
    probabilities = defaultdict(float)

    # Calculate probabilities for each answer in the current row
    total_answers = len(row['Combined_Answers'])
    if total_answers > 0:
        for answer in row['Combined_Answers']:
            probabilities[answer] += 1 / total_answers

    # Round probabilities to 3 digits after the decimal point
    rounded_probabilities = {key: round(value, 3) for key, value in probabilities.items()}

    # Append rounded probabilities to the list
    probabilities_list.append(rounded_probabilities)

# Create a new column to store the dictionaries of probabilities
new_df['Answer_Probabilities'] = probabilities_list

# Adds most frequent answer
new_df['Most_Frequent_Answer'] = new_df['Combined_Answers'].apply(lambda x: mode(x) if len(x) > 0 else None)

# Drops 'Combined_Answers' column; CAN BE COMMENTED IF NEEDED
new_df.drop(columns=['Combined_Answers'], inplace=True)

# Print the DataFrame to verify the changes
print(new_df)

new_df.to_csv('reproductive_rtu.csv', index=False)
