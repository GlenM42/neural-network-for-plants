#Phillip Waul
#2/7/2024
import pandas as pd

#Open our csv and use it as a data frame in pandas
file_path = "plant_images.csv"
df = pd.read_csv(file_path, dtype=str, low_memory=False)

#I think the name "image_name" makes a lot more sense than "metadata" at this point so I'm changing that in particular.
#This makes a copy with the new name which will be our new dataframe
new_df = df.rename(columns={"metadata" : "image_name"})
#Go through each row and chop off the first 6 and last two characters. 
rowIndex = 0
for row in new_df["locations"]:
    new_df.loc[rowIndex, "locations"] = row[6:-2]
    rowIndex +=1

#Test to make sure the ends are correct. 
#for row in new_df["locations"]:
#    print(row)
#    if (row[-4:] != "jpeg" and row[-4:] != ".png") or row[:4] != "http":
#        print("The last one wasn't right")
#        quit()

new_df.to_csv('plant_images_cleaned.csv', index=False)