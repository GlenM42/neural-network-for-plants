## neural-network-for-plants

### Plan
1. Some of the useless columns got deleted. That way the user_data.csv went from being 500 MB to <100MB.
2. parser.py was created as a file to do some initial analysis on the data. It showed:
   - there are ~6 columns worth analyzing
   - we have to go about each of them one by one
   - the first one, 'user_name' was analyzed

### Activity Log
Working on the Reproductive dataset 

### Some usefull links for future

Article on how to do the CNN with Keras (they use CIFAR-10 dataset, the handwritten numbers)
https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/?utm_source=blog&utm_source=learn-image-classification-cnn-convolutional-neural-networks-5-datasets#one

How to create a dataset of images with Keras
https://keras.io/api/data_loading/image/
