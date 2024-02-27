import csv
import os
import shutil

# Define the path to the CSV file and the images folder
csv_file_path = 'pieces_of_datasets/reproductive_rtu.csv'
images_folder_path = 'images'

# Define the target folders for each category
categories = ['sterile', 'both', 'female', 'male', 'not sure']

# Create the category folders if they don't exist
for category in categories:
    os.makedirs(os.path.join(images_folder_path, category), exist_ok=True)

# Read the CSV file
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        # Extract the image filename and category from each row
        image_filename, _, _, category = row

        # Check if the category is one of the predefined categories
        if category in categories:
            # Define the source and destination paths for the image
            source_path = os.path.join(images_folder_path, image_filename)
            destination_path = os.path.join(images_folder_path, category, image_filename)

            # Move the image to the appropriate category folder
            shutil.move(source_path, destination_path)
