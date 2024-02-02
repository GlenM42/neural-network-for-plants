## neural-network-for-plants

### Plan
1. Some of the useless columns got deleted. That way the user_data.csv went from being 500 MB to <100MB.
2. parser.py was created as a file to do some initial analysis on the data. It showed:
   - there are ~6 columns worth analyzing
   - we have to go about each of them one by one
   - the first one, 'user_name' was analyzed

### Activity Log
Glen: I think we should analyze each of those columns one by one. In the process of doing so, we will
simplify and shring dataset to a substantial degree. I was able to do it for 'user_name' column. 
For example, from the start we got 1767 distinct users. After we delete 'not-logged-in', we are left
with 1009. The graph of the distribution shows that SO MANY of them are dudes who made 1 guess. 
After considering only 80% of the distribution, we are left with 199 users. 

### Some usefull links for future

Article on how to do the CNN with Keras (they use CIFAR-10 dataset, the handwritten numbers)
https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/?utm_source=blog&utm_source=learn-image-classification-cnn-convolutional-neural-networks-5-datasets#one
