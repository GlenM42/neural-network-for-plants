import pandas as pd
import csv
import json


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
with open('Reproductive.csv', newline='') as csvfile:
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

df = pd.read_csv("output_data.csv", dtype=str, low_memory=False)

grouped_df = df.groupby('Filename').agg(list)

# Reset index to make "Filename" a regular column instead of an index
grouped_df.reset_index(inplace=True)

# Print the grouped DataFrame
print(grouped_df)
