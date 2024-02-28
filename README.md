## neural-network-for-plants

### Plan
1. Some of the useless columns got deleted. That way the user_data.csv went from being 500 MB to <100MB.
2. parser.py was created as a file to do some initial analysis on the data. It showed:
   - there are ~6 columns worth analyzing
   - we have to go about each of them one by one
   - the first one, 'user_name' was analyzed

### What we've done
1. Parsing Data
   - We had to maniplulate the csv files of our given data to isolate data that was relevant to us and download pictures for the dataset. 
      - The python files we used for this are in the parsers folder
      - The csv files are in the pieces_of_datasets folder
   - parser01.py and parser02.py 
      - parser01.py takes user_data_cleaned.csv and outputs output_data.csv
      - user_data_cleaned came from manually deleting columns from the given  unfolding-of-microplant-mysteries-classifications 3.20.2023.csv. All we had left over were the columns: user_name,workflow_name,subject_data,question,answer,another_answer
      - output_data.csv is then put through parser02.py and is modified to be what it is in this repository: There are 2 columns: Filename is the name of each image, and Answer is the highest reported answer for that image.
   - linkCleaner.py
      - plant_images.xlsx was saved as plant_images.csv
         - plant_images.xlsx was a given file
         - this file has a lot of extra characters around the links
      - this file simply removes the extra characters from the links in the original file so it is more useable later
      - saves output as plant_images_cleaned.csv
   - image_downloader.py
      - this script simply goes through each image in pant_images_cleaned.csv and downloads them to a folder.
   - extra_images_parser.py and findMissingImages.py
      - plant_images_cleaned had a different amount of unique images as output_data.csv
      - findMissingImages.py creates the text file extra_images.txt that finds all the images that are in output_data.csv and not in plant_images_cleaned.csv
      - this process included duplicates, so extra_images_parser.py created extra_images_cleaned.txt to list the unique images that were not in both files. There were only a few, so these images were manually deleted.
   - image_sorter.py
      - images needed to be put into files depending on their classification for ease of use by first_nn.py - the neural network itself. 
2. Neural Network
   - first_nn.py
      - This is our first attempt at a bare bones CNN network! Because of that, we can 
      - More data processing!
         - The first section of code walks through all the images and makes sure they are not corrupted or otherwise unuseable. 
         - all images that were not useable were manually moved into to images_not_used folder
      - We split the data into a testing set, validation set, and training set
      - All that's left is to do some formatting and the neural network runs! It learns the training set well, but is not good at generalizing it to new images. 

### Some usefull links for future

Article on how to do the CNN with Keras (they use CIFAR-10 dataset, the handwritten numbers)
https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/?utm_source=blog&utm_source=learn-image-classification-cnn-convolutional-neural-networks-5-datasets#one

How to create a dataset of images with Keras
https://keras.io/api/data_loading/image/


