#Phillip Waul
#2/7/2024
import pandas as pd
import requests 
from PIL import Image 

#Open our csv and use it as a data frame in pandas
file_path = "pieces_of_datasets/plant_images_cleaned.csv"
df = pd.read_csv(file_path, dtype=str, low_memory=False)

for index, row in df.iterrows():
    #get link from csv row by row and save images into a folder using the image_name as the file name."
    #get file
    url = row["locations"]
    data = requests.get(url).content
    #create new file with the name of the image in the working row
    image_path = "./images/" + row["image_name"]
    f = open(image_path,'wb') 
    #write data to file and finish before looping to next row.
    f.write(data)
    f.close()

