import pandas as pd
import matplotlib.pyplot as plt

file_path = "user_data_cleaned.csv"
df = pd.read_csv(file_path, dtype=str, low_memory=False)

# Initial look
print(df.head())
print(df.describe())
column_names = df.columns.tolist()
print("Column names: ", column_names)

# As much as I understand it, the 'Unnamed' columns are
# the stuff that is relevant only to drawing the boxes

# Here, creating a new data frame with usefull ones
column_names = ['user_name', 'workflow_name', 'subject_data', 'question', 'answer', 'another_answer']
new_df = df[column_names]
print(new_df.head(2))
print(new_df.describe())

# Analyzing the user_name column
user_name_frequency = df['user_name'].value_counts()
print(user_name_frequency)

# Since we do not care about 'not-logged-in' ppl, I am creating a filtered_users list:
filtered_users = user_name_frequency[~user_name_frequency.index.str.contains('not-logged-in')]
print(filtered_users)

# I think it would be usefull to see the distribution on the graph, so:
filtered_users.plot.bar(x='user_name', y='count', figsize=(12, 6), color='skyblue', rot=45)
plt.xlabel('User Name')
plt.ylabel('Count')
plt.title('User Activity')
# Display the chart
plt.show()

# Obviously, we need not care about those dudes who made ultra low contributions
# Therefore, I will do a test to find out 80% of the distribution
print("80th Percentile Count: ", filtered_users.quantile(0.8))

# Great. Now, let's limit the graph to those who made > 76 guesses.
threshold_count = 76
active_users = filtered_users[filtered_users > threshold_count]
print(active_users)

ax = active_users.plot.bar(x='user_name', y='count', figsize=(12, 9), color='skyblue', rot=90)
plt.xlabel('User Name')
plt.ylabel('Count')
plt.title('User Activity')
plt.legend()
# Display the bar chart
plt.show()
