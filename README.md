# VLFeat SVM-Face-Recognition-Algorithm
This purpose of this algorithm is to use the VLfeat library to distinguish between an image with a face and an image without a face using the built in Support Vecotr Machine Algorithm in the VLFet Library.

## Introduction

The VLFeat library is a popular computer vision library specializing in image understanding and local features extraction and matching. Algorithms include Fisher Vector, VLAD, SIFT, MSER, k-means, hierarchical k-means, agglomerative information bottleneck, SLIC superpixels, quick shift superpixels, large scale SVM training, and many others.

![Capture](https://user-images.githubusercontent.com/39222728/57195365-8f476180-6f1f-11e9-9202-72ce0b49d109.JPG)

## Datasets

For this project we will be working with two datasets:

1. image_notefaces 
2. cropped_training_images_faces

Both of which we can find in our repository above.

Firstly, we need to geenrate a bit of code to convert all of the images that are note faces, into grayscale and crop them to a 36x36 pixel size.

Below we can see the snippet of code as an example of how to do that

![Capture](https://user-images.githubusercontent.com/39222728/57195392-d6cded80-6f1f-11e9-9747-44ea22c7ff93.JPG)

After running this code, we create a directory or folder with all of the cropped and grayscale version of the images that are not faces

![Capture](https://user-images.githubusercontent.com/39222728/57195752-33330c00-6f24-11e9-8b0a-3c03274a5d5f.JPG)

Next, we need to split our training set into 2 catagories: a training set and a validation set. Our split will be 80% training and 20% validation. Below we we the section of code that lets us do this.

![Capture](https://user-images.githubusercontent.com/39222728/57195804-a9377300-6f24-11e9-99c3-2aa3e88a0459.JPG)

Note how we relabel the image_not faces in the folder previously created into training and validation images. Our final folder containing all of the images that are not faces will then look like this:

![Capture](https://user-images.githubusercontent.com/39222728/57195835-177c3580-6f25-11e9-8b93-7212856e558f.JPG)

Once we've established our training and validation sets, we can being training our dataset and algorithm using the built-in SVM machine learning algorithm in the VLFeat library we are using.

Firstly, we look at the HOG features for the images that are faces. To do that we need to create two directories for the images, the images that are a postive match, and the images that are a negative match. Below we can see how to do that

![Capture](https://user-images.githubusercontent.com/39222728/57195969-113a8900-6f26-11e9-8743-3696ed5208b2.JPG)

Note that we can change the variable cellSize to change how our algoirithm classifies an image, this paramter can be useful when it comes to improving the accuracy of our aglorithm

To get all the postive image HOG features we use the snippet of code below:

![Capture](https://user-images.githubusercontent.com/39222728/57196003-78583d80-6f26-11e9-8ef5-46f48c167486.JPG)


![Capture](https://user-images.githubusercontent.com/39222728/57195922-b30da600-6f25-11e9-88e8-953eedf5a7e4.JPG)
