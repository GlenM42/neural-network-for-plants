#Go through the list of iamge names we have and compare it to the list of images we have downloaded and make a file with the images that are not downloaded.
#output_data has all of the image names

#Phillip Waul
#2/7/2024
import pandas as pd


def Clean_filename(filename):
    #Function for removing everything after a non alphanumeric character (except underscores and dashes) in the filename
    for char in filename:
        if char.isalnum() == False and char != "_" and char != "-":
            return filename[:filename.index(char)]
    return filename




#Open our csv and use it as a data frame in pandas
file_path = "output_data.csv"
name_frame = pd.read_csv(file_path, dtype=str, low_memory=False)
file_path = "plant_images_cleaned.csv"
links_frame = pd.read_csv(file_path, dtype=str, low_memory=False)

count = 0
f = open("extra_images.txt", "a")

#Not the most efficient way to do this, but I just wanted to make sure it would work
#Loops through every image in one csv, then loops through the entire second csv and makes sure the image is in both.
for names_index, name_row in name_frame.iterrows():
    isInFile = False
    for links_index, image_id_row in links_frame.iterrows():
        filename = name_row["Filename"]
        #Here I'm just chopping off the end of the file name such that we're more likely to see it is an actually different image.
        filename_clean = Clean_filename(filename)
        #checks if the names are the same
        if filename_clean in image_id_row["image_name"]:
            #If they are the same, we flag that it is in the second file and move on to the next image.
            isInFile = True
            break
    if isInFile == False:
        print(name_row["Filename"])
        count += 1
        f.write(filename + "\n")
    else: 
        print("Image Verified: " + str(names_index) + "/" + str(len(name_frame.index)-1))

print(count)
#f.write(str(count))
f.close()


