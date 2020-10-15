#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks
# 
# ## Project: Write an Algorithm for a Dog Identification App 
# 
# ---
# 
# In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.
# 
# The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.
# 
# 
# 
# ---
# ### Why We're Here 
# 
# In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 
# 
# ![Sample Dog Output](images/sample_dog_output.png)
# 
# In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!
# 
# ### The Road Ahead
# 
# We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.
# 
# * [Step 0](#step0): Import Datasets
# * [Step 1](#step1): Detect Humans
# * [Step 2](#step2): Detect Dogs
# * [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
# * [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
# * [Step 5](#step5): Write your Algorithm
# * [Step 6](#step6): Test Your Algorithm
# 
# ---
# <a id='step0'></a>
# ## Step 0: Import Datasets
# 
# Make sure that you've downloaded the required human and dog datasets:
# * Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dogImages`. 
# 
# * Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  
# 
# *Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*
# 
# In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.

# In[1]:


import numpy as np
from glob import glob


# In[2]:


# load filenames for human and dog images
human_files = np.array(glob("lfw/*/*"))
dog_files = np.array(glob("dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))


# <a id='step1'></a>
# ## Step 1: Detect Humans
# 
# In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  
# 
# OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.

# In[3]:


import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  
# 
# In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.
# 
# ### Write a Human Face Detector
# 
# We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.

# In[5]:


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# ### (IMPLEMENTATION) Assess the Human Face Detector
# 
# __Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
# - What percentage of the first 100 images in `human_files` have a detected human face?  
# - What percentage of the first 100 images in `dog_files` have a detected human face? 
# 
# Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

# __Answer:__ 
# (You can print out your results and/or write your percentages in this cell)

# In[6]:


from tqdm import tqdm


# In[7]:


human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
human_img_detected = [face_detector(f) for f in tqdm(human_files_short)]
dog_img_detected = [face_detector(f) for f in tqdm(dog_files_short)]

print(f'{np.mean(human_img_detected)*100}% of human images with a detected face')
print(f'{np.mean(dog_img_detected)*100}% of dog images with a detected face')


# We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

# In[8]:


### (Optional) 
### TODO: Test performance of another face detection algorithm.
### Feel free to use as many code cells as needed.


# ---
# <a id='step2'></a>
# ## Step 2: Detect Dogs
# 
# In this section, we use a [pre-trained model](https://pytorch.org/docs/stable/torchvision/models.html) to detect dogs in images.  
# 
# ### Obtain Pre-trained VGG-16 Model
# 
# The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  

# In[9]:


import torch
import torchvision.models as models


# In[10]:


# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# set VGG16 model to eval mode
VGG16.eval()

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


# Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

# ### (IMPLEMENTATION) Making Predictions with a Pre-trained Model
# 
# In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.
# 
# Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).

# In[11]:


from PIL import Image
import torchvision.transforms as transforms


# In[12]:


# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    img = Image.open(img_path)
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = img_transform(img).unsqueeze(0)
    if use_cuda:
        img = img.cuda()
    pred = VGG16(img)[0].argmax()
    
    return pred.item()  # predicted class index


# ### (IMPLEMENTATION) Write a Dog Detector
# 
# While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).
# 
# Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

# In[13]:


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    dog_detected = (151 <= VGG16_predict(img_path) <= 268)
    return dog_detected  # true/false


# ### (IMPLEMENTATION) Assess the Dog Detector
# 
# __Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
# - What percentage of the images in `human_files_short` have a detected dog?  
# - What percentage of the images in `dog_files_short` have a detected dog?

# __Answer:__ 
# 

# In[14]:


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
human_img_detected = [dog_detector(f) for f in tqdm(human_files_short)]
dog_img_detected = [dog_detector(f) for f in tqdm(dog_files_short)]

print(f'{np.mean(human_img_detected)*100}% of human images with a detected dog')
print(f'{np.mean(dog_img_detected)*100}% of dog images with a detected dog')


# We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](https://pytorch.org/hub/pytorch_vision_inception_v3/), [ResNet-50](https://pytorch.org/hub/pytorch_vision_resnet/), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

# In[15]:


### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.


# In[16]:


resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
if use_cuda:
    resnet50 = resnet50.cuda()

def resnet50_predict(img_path):
    img = Image.open(img_path).convert('RGB')
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = img_transform(img).unsqueeze(0)
    if use_cuda:
        img = img.cuda()
    pred = resnet50(img)[0].argmax()
    return pred.item()

def dog_detector_resnet50(img_path):
    dog_detected = (151 <= resnet50_predict(img_path) <= 268)
    return dog_detected

human_img_detected = [dog_detector_resnet50(f) for f in tqdm(human_files_short)]
dog_img_detected = [dog_detector_resnet50(f) for f in tqdm(dog_files_short)]

print('ResNet50:')
print(f'{np.mean(human_img_detected)*100}% of human images with a detected dog')
print(f'{np.mean(dog_img_detected)*100}% of dog images with a detected dog')


# ---
# <a id='step3'></a>
# ## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
# 
# Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.
# 
# We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  
# 
# Brittany | Welsh Springer Spaniel
# - | - 
# <img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">
# 
# It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  
# 
# Curly-Coated Retriever | American Water Spaniel
# - | -
# <img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">
# 
# 
# Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  
# 
# Yellow Labrador | Chocolate Labrador | Black Labrador
# - | -
# <img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">
# 
# We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  
# 
# Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!
# 
# ### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
# 
# Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!

# In[17]:


import os
from torchvision import datasets


# In[18]:


### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
data_dir = './dogImages'

train_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(data_dir+'/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir+'/valid', transform=test_transforms)
test_data  = datasets.ImageFolder(data_dir+'/test',  transform=test_transforms)

n_classes = len(train_data.classes)

batch_size = 20
# Setting shuffle=True can improve the performance a lot!
loaders_scratch = {
    'train': torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),
    'valid': torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True),
    'test':  torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=True)
}

print(f'  Train: {len(train_data)}')
print(f'  Valid: {len(valid_data)}')
print(f'   Test: {len(test_data)}')
print(f'Classes: {n_classes}')


# **Question 3:** Describe your chosen procedure for preprocessing the data. 
# - How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
# - Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?
# 

# **Answer**:
# 
# - I resize the images by:
#     - reshaping the raw (test) image to be a 256x256 square
#         - reason: reduce information loss in the following cropping step
#     - cropping out a 224x224 part from the above reshaped image
#         - reason: standard input image size referring to pre-trained models in pytorch
#         - reason: also, 224 is an even number, making it easier to work with max pooling (kernel: 2, stride: 2)
# - Yes. I agumented the train set in order to achieve better out-of-sample performance, by applying:
#     - random rotation
#     - random cropping
#     - random horizontal flip
# - Validation and test sets are applied with the same transformation to make sure that they come from the same distribution.

# ### (IMPLEMENTATION) Model Architecture
# 
# Create a CNN to classify dog breed.  Use the template in the code cell below.

# In[19]:


import torch.nn as nn
import torch.nn.functional as F


# In[20]:


# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128,  out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=7*7*256, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=n_classes)
        self.dropout = nn.Dropout(p=0.3)
        ## Weights Initialization
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')
    
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.bn5(x)
        b, _, _, _ = x.size()
        x = x.view(b, -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch = model_scratch.cuda()


# In[21]:


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n_in = m.in_features
        m.weight.data.normal_(mean=0, std=1/np.sqrt(n_in))
        m.bias.data.fill_(0)

model_scratch.apply(weights_init_normal)


# __Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

# __Answer:__ 
# 
# | Step | Architecture Details |
# |------|---------------------------|
# |raw image|(3 x 224 x 224)|
# |conv1|`Conv2d(3, 16, kernel_size3, stride=1, padding=1)`|
# |max pool|`MaxPool2d(kernel_size=2, stride=2)`|
# |conv2|`Conv2d(16, 32, kernel_size=3, stride=1, padding=1)`|
# |max pool|`MaxPool2d(kernel_size=2, stride=2)`|
# |conv3|`Conv2d(32, 64, kernel_size=3, stride=1, padding=1)`|
# |max pool|`MaxPool2d(kernel_size=2, stride=2)`|
# |conv4|`Conv2d(64, 128, kernel_size=3, stride=1, padding=1)`|
# |max pool|`MaxPool2d(kernel_size=2, stride=2)`|
# |conv5|`Conv2d(128, 256, kernel_size=3, stride=1, padding=1)`|
# |max pool|`MaxPool2d(kernel_size=2, stride=2)`|
# |fc1|`Linear(in_features=12544, out_features=2048)`|
# |fc2|`Linear(in_features=2048, out_features=133)`|
# 
# 
# - Details of my Neural Network architecture can be found above.
# - A few side notes:
#     - In this NN, `ReLU` activation function is always applied.
#     - Weights initialization (normally distributed) is implemented to provide a good starting point for optimization.
#     - Batch normalization is included to help speeding up the optimization algorithm.
#     - Dropout is used to mitigate the problem of overfitting to the training data.
# - Regarding the design of the overall architecture,
#     - In the convolution part, I follow the general rule of "increasing depth and reducing width/height" while moving deeper into the NN.
#     - Larger depth helps us extract more features from raw inputs, and smaller width/height reduces the number of params to be estimated.
#     - `16, 32, ..., 256` these numbers are chosen so that the whole network can be trained on my local gpu. Ideally, with adequate computing power, we could mimic the structure of VGG16 (or other established networks) and include more convolution layers.
#     - In the fully connected layer part, I used two linear layers to balance between "the no. of params need to be estimated" and "the loss of information". For the number of neurons to be included in the second fc layer, `512` and `2048` are tested and `2048` achieves better results.

# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
# 
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.

# In[22]:


import torch.optim as optim


# In[23]:


### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

# specify optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=1e-3, momentum=0.9)


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.

# In[24]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[25]:


# the following import is required for training to be robust to truncated images
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model


# In[26]:


get_ipython().run_cell_magic('time', '', "# train the model\nmodel_scratch = train(10, loaders_scratch, model_scratch, optimizer_scratch, \n                      criterion_scratch, use_cuda, 'model_scratch.pt')")


# In[27]:


# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))


# ### (IMPLEMENTATION) Test the Model
# 
# Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.

# In[28]:


def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


# In[29]:


# call test function
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)


# ---
# <a id='step4'></a>
# ## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
# 
# You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.
# 
# ### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
# 
# Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 
# 
# If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.

# In[30]:


## TODO: Specify data loaders
loaders_transfer = {
    'train': torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),
    'valid': torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True),
    'test':  torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=True)
}


# ### (IMPLEMENTATION) Model Architecture
# 
# Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.

# In[31]:


import torchvision.models as models
import torch.nn as nn


# In[32]:


## TODO: Specify model architecture 
model_transfer = models.vgg16(pretrained=True)

# freeze training for all "features" layers
for param in model_transfer.features.parameters():
    param.requires_grad = False

n_in = model_transfer.classifier[6].in_features
model_transfer.classifier[6] = nn.Linear(in_features=n_in, out_features=n_classes)
print(model_transfer)

if use_cuda:
    model_transfer = model_transfer.cuda()


# __Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

# __Answer:__ 
# 
# - When designing the architecture, I follow guidelines in the previous transfer learning lab of this course.
# - VGG16 is pre-trained and provides us a good feature extractor. Besides, the dataset (ImageNet) VGG16 is pre-trained on is similar to the dog breed dataset we have at hand. Therefore, it is reasonable to freeze the `features` part of VGG16 and update (re-train) the final layer in order to accommodate for our new classification task (with 133 classes).

# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
# 
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.

# In[33]:


criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=1e-3)


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.

# In[34]:


get_ipython().run_cell_magic('time', '', "# train the model\nn_epochs = 10\nmodel_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer,\n                       criterion_transfer, use_cuda, 'model_transfer.pt')")


# In[35]:


# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


# ### (IMPLEMENTATION) Test the Model
# 
# Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.

# In[36]:


test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)


# ### (IMPLEMENTATION) Predict Dog Breed with the Model
# 
# Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  

# In[37]:


### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in train_data.classes]

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    img = Image.open(img_path).convert('RGB')
    in_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = in_transform(img)[:3,:,:].unsqueeze(0)
    if use_cuda:
        img = img.cuda()
    model_transfer.eval()
    output = model_transfer(img)
    idx = output.argmax().item()
    return class_names[idx]


# In[38]:


img_path = './images/Brittany_02625.jpg'
predict_breed_transfer(img_path)


# ---
# <a id='step5'></a>
# ## Step 5: Write your Algorithm
# 
# Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
# 
# - if a __dog__ is detected in the image, return the predicted breed.
# - if a __human__ is detected in the image, return the resembling dog breed.
# - if __neither__ is detected in the image, provide output that indicates an error.
# 
# You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  
# 
# Some sample output for our algorithm is provided below, but feel free to design your own user experience!
# 
# ![Sample Human Output](images/sample_human_output.png)
# 
# 
# ### (IMPLEMENTATION) Write your Algorithm

# In[39]:


### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
def imshow(img_path):
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show()


def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    # model predictions
    human_detected = face_detector(img_path)
    dog_detected = dog_detector(img_path)
    breed = predict_breed_transfer(img_path)
    # show image
    imshow(img_path)
    if human_detected:
        print('Hello, human!')
        print('You look like a ...')
        print(breed)
    elif dog_detected:
        print('Hello, dog!')
        print('Your breed seems to be ...')
        print(breed)
    else:
        print('An "alien" detected!!!')
        print('Neither a dog nor a human!!!')


# ---
# <a id='step6'></a>
# ## Step 6: Test Your Algorithm
# 
# In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?
# 
# ### (IMPLEMENTATION) Test Your Algorithm on Sample Images!
# 
# Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  
# 
# __Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

# __Answer:__ (Three possible points for improvement)
# 
# - The model performs very well (better than I expected). This project showcases the power of transfer learning.
# - Talking about 3 possible improvements, I would say:
#     1. we could use weights from VGG16 as a starting point and try to re-train the whole network (given enough computing power)
#     2. we could try more advanced optimization algorithms in order to achieve faster convergence
#     3. we could include another fully connected layer in the classifier part to mitigate the sharp drop of the number of neurons in adjacent layers

# In[40]:


## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

## suggested code, below
test_images = np.array(glob('images_test/*'))
for img_path in test_images:
    run_app(img_path)

