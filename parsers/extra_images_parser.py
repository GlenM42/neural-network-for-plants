# The extra_images.txt file includes images that had "Copy Of " in front of them. That has been deleted. 
# So some of the image names in the list are repeats, which we can get rid of. 
# Some of them are also actually in the list now that we have removed the "Copy of " problem. 
# So this program will just trim the list down to ONLY images we do not have so we can remove them from the main csv of classifications.
# Phillip Waul
# 2/13/24
import pandas as pd

file_path = "pieces_of_datasets/plant_images_cleaned.csv"
links_frame = pd.read_csv(file_path, dtype=str, low_memory=False)

f = open("pieces_of_datasets/extra_images.txt", "r")
f_clean = open("pieces_of_datasets/extra_images_cleaned.txt", "a")

iList = []
count = 0

for line in f:
    count = count + 1
    line_cleaned = line[:-1]
    if line_cleaned not in iList:
        isInFile = False
        for links_index, image_id_row in links_frame.iterrows():
            if line_cleaned in image_id_row["image_name"]:
                isInFile = True
                break
        if isInFile == False:
            iList.append(line_cleaned)
            print(line_cleaned)
            f_clean.write(line)
        else:
            print("image verified" + str(count))


f_clean.close()
f.close()
            
        