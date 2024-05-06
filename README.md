## neural-network-for-plants

### Plan
1. Some of the useless columns got deleted. That way the user_data.csv went from being 500 MB to <100MB.
2. A whole folder of python parses has been created to analyze the dataset.
3. All the results were transformed into the .csv files on which we ran the NN.
4. With different version of NN we couldn`t break the ceiling of ~65% validation accuracy on the dataset. We consider it soo small.
5. Might include unit testing to see if there were any mistakes on the previous steps.
6. Might restructure the NN from guessing all four choices (female, male, both, sterile) to just binary (sterile/not-sterile). 

### What we've done
1. Parsing Data
   - We had to maniplulate the csv files of our given data to isolate data that was relevant to us and download pictures for the dataset. 
      - The python files we used for this are in the parsers folder
      - The csv files are in the pieces_of_datasets folder
   - **parser01.py** and **parser02.py**
      - parser01.py takes __user_data_cleaned.csv__ and outputs __output_data.csv__
      - __user_data_cleaned__ came from manually deleting columns from the given __unfolding-of-microplant-mysteries-classifications 3.20.2023.csv__. All we had left over were the columns: user_name,workflow_name,subject_data,question,answer,another_answer
      - __output_data.csv__ is then put through parser02.py and is modified to be what it is in this repository: There are 2 columns: Filename is the name of each image, and Answer is the highest reported answer for that image.
   - **reproductive_parser.py**
      - This one is working with Reproductive.csv. From now on, the project got focused on the Reproductive classification.
      - It creaates the most clear dataset as a pandas dataframe with columns 'Filename' (name of the image), 'Most_Frequent_Answer' (self-explanatory, right?), 'Answer_Probabilities' (creates a python dictionary with the probabilities for all the answers), 'Answers' (creates a Python list with all the answes from all the users), and 'locations' (exact links to the images).
      - Based on this, it creates two .csv files:
         - __reproductive_almost_rtu.csv__ (rtu stands for ready-to-use) which includes all of the columns mentioned above and image_id. This version was intended to be shown for everybody as the most complete piece of data we could get.
         - __reproductive_rtu.csv__ gets rid of the 'Answers' and 'image_id' columns. This version was used in the later processes. 
   - **linkCleaner.py**
      - __plant_images.xlsx__ was saved as __plant_images.csv__
         - plant_images.xlsx was a given file
         - this file has a lot of extra characters around the links
      - this file simply removes the extra characters from the links in the original file so it is more useable later
      - saves output as plant_images_cleaned.csv
   - **image_downloader.py**
      - this script simply goes through each image in __pant_images_cleaned.csv__ and downloads them to a folder.
   - **extra_images_parser.py** and **findMissingImages.py**
      - __plant_images_cleaned__ had a different amount of unique images as __output_data.csv__
      - **findMissingImages.py** creates the text file extra_images.txt that finds all the images that are in __output_data.csv__ and not in __plant_images_cleaned.csv__
      - this process included duplicates, so extra_images_parser.py created __extra_images_cleaned.txt__ to list the unique images that were not in both files. There were only a few, so these images were manually deleted.
   - **image_sorter.py**
      - images needed to be put into files depending on their classification for ease of use by first_nn.py - the neural network itself. 
2. Neural Network
   - **nn_version_0.1.py**
      - This is our first attempt at a bare bones CNN network! Because of that, we can 
      - More data processing!
         - The first section of code walks through all the images and makes sure they are not corrupted or otherwise unuseable. 
         - all images that were not useable were manually moved into to images_not_used folder
      - We split the data into a testing set, validation set, and training set
      - All that's left is to do some formatting and the neural network runs! It learns the training set well, but is not good at generalizing it to new images. The approximate numbers are around 90% for the training dataset, and 40% for the validation.
      - Also, we have to note that the loss parameter drops too quickly in the learning process: on the first iteration the model has very high loss, and on the second and forward it makes it vary small. What it means is that the neural network is applying very heavy weights when changing its behavior. 
   - **nn_version_0.2.py**
      - Adds a function to check_and_resave images that will prompt a question in the command line to do just that.
      - Since we classified the problem as overfitting, we decided to implement the following practicies to reduce it:
         - Data Augmentation. That module takes our images and randomly rotates them, making the learning more impactful. The value of parameter is 0.2.
         - Dropout Layer. That layer is making the neural network disregard some portion of the nodes it has created in the learning process. The idea is that it will disregard the modules that "memorized the training dataset too good". The value of a parameter is 0.5.
         - Regularizer layer. This layer adresses the heavy weights problem: it punishes neural network for rapidly changing the answers, as we want the process to go more smoothly. The value of a paramter is 0.001.
      - After applying all of the mentioned techniques and increasing the training-to-validation ratio to 4:1, this version produces results with ~60% validation accuracy.
   - **nn_version_0.3.py**
      - The problem with a previous version lies with the parameters -- we do not know what the optimal values are. This version makes the NN try all the possible values for all three methods (data augmentation, dropout, and regularizer) using the Hyperband, and selects the best parameters based on the validation accuracy.
      - The results can be seen on the pic below:
      - <img width="508" alt="pic" src="https://github.com/GlenM42/neural-network-for-plants/assets/149723560/1daeb508-a0ec-47d1-a2e3-3232a7a1e342">
      - In the process of observing these trials, we had to notice it consistently got 65.35%. At some point we even begun to question if it would ever break this ceiling.
      - As you can tell by the image, it did. The best accuracy we have gotten was 65.84%, which is not a significant improvement. Based on this, we conclude that __it is not possible to get better results on this dataset__. Most likely, it is __too small__ for the NN to train on. 


### Some usefull links for future

Article on how to do the CNN with Keras (they use CIFAR-10 dataset, the handwritten numbers)
https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/?utm_source=blog&utm_source=learn-image-classification-cnn-convolutional-neural-networks-5-datasets#one

How to create a dataset of images with Keras
https://keras.io/api/data_loading/image/


The second branch in this repository also contains some extra files. I made an attempt to use a custom type of network that uses harmonics to do a convolutional neural network. This was problematic when it came to our custom dataset that used code from different versions of tensorflow. I was not able to get these to work correctly. It could be an interesting project for the future!




