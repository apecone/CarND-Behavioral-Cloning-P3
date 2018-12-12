# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-nvidia.png "Model Visualization"
[image2]: ./examples/RGB_image.jpg "Original Image"
[image3]: ./examples/YUV_image.png "YUV Image"
[image4]: ./examples/YUV_image_flipped.png "Flipped YUV Image"
[image5]: ./examples/YUV_image_flipped_blurred.png "Flipped & Blurred YUV Image - Left"
[image6]: ./examples/YUV_image_flipped_blurred_cropped.png "Flipped, Blurred & Cropped YUV Image"
[image7]: ./examples/YUV_image_flipped_blurred_cropped_resized.png "Flipped, Blurred, Cropped, & Resized YUV Image (Final Result)"
[image8]: ./examples/Final_result_left_flpped.png "Final Result - Left & Flipped"
[image9]: ./examples/Final_result_right_flipped.png "Final Result - Right & Flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

My project does not include the following files:
* data.tgz containing all data which was used for training (~442 MB)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture I used is a convolution neural network first utilized by Nvidia for self-driving car applications.  The network consists of 9 layers, including normalization layer, 5 convolutional layers, and 3 fully connected layers.  The image is preprocessed and converted into YUV planes, which are then passed through the network.

The model includes ELU layers to introduce nonlinearity (model.py lines 120, 122, 124, 127, 129, 134, 136, and 138), and the data is normalized in the model using a Keras lambda layer (model.py line 117). 

| ![alt text][image1] |
|:--:|
| *Nvidia CNN Architecture* |

#### 2. Attempts to reduce overfitting in the model

The model contains l2 kernel regularization in order to reduce overfitting (model.py lines 119, 121, 123, 126, 128, 133, 135, and 137). 

The model was trained and validated on different data sets to ensure that the model was not overfitting during training (code line 188-194, 211, and 214). The model was later tested via two methods. First, I tested it on a holdout set for which none of the hyperparameters had been tuned. Second, by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with an initial learning rate of 0.0001, beta_1 of 0.9, and beta_2 of 0.999, and 0.0 decay (model.py line 207).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first see what researchers in the field had been doing for these kinds of problems.  After looking at several different architectures, I landed on the Nvidia architecture since it was simple to implement and well suited for my particular self-driving car application.

I knew it would be a good architecture because it had convolutional layers which allowed for the model to be of a reasonable size as compared to a fully connected network.  Furthermore, the ELU layers allowed for the network to learn interesting non-linear functions.  Because the model had fewer parameters, it would be easier and quicker to train with the limited amount of data at my disposal.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.  As I was training the model, my ModelCheckpoint saved all models which saw validation-set improvements. Although I had success very early, I decided to gather more training data to help my model generalize better.

In order to create more data for training my model, I first decided to implement a DataGenerator class which inherits the properties of the Keras Sequence class (model.py lines 17-98).  The reason why I wanted to implement this class is because it allows me to leverage some nice functionalities such as multiprocessing (model.py lines 217); therefore, I could preprocess images very quickly and theoretically speed up my training.  In my data generator, I use left and right images with steering wheel corrections.  I also create flipped images with steering corrections.  The blurring was implemented to help the model to generalize to poor seeing conditions such as rain, fog, snow, etc.  And, finally, to ensure I have enough data for my model, I gathered my own data and allowed my generator class to do the rest (model.py lines 179-180).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 112-142) was the Nvidia self-driving car CNN architecture which consists of 9 layers, including normalization layer, 5 convolutional layers, and 3 fully connected layers.

The model includes ELU layers to introduce nonlinearity (model.py lines 120, 122, 124, 127, 129, 134, 136, and 138), and the data is normalized in the model using a Keras lambda layer (model.py line 117). 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

| ![alt text][image1] |
|:--:|
| *Nvidia CNN Architecture* |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two reverse laps on track one using center lane driving. Because I was driving pretty fast, I had to record several times due to mistakes during recording :D 

I also recorded a little bit on track 2 to help my model generalize better.

Here is an example image of center lane driving:

| ![alt text][image2] |
|:--:|
| *Original RGB Image* |

I never conducted any recovery routes because my augmented left and right data simulated these recovery routes for me.  Here are some examples of my preprocessed center image, along with it's left and right counterparts.

| ![alt text][image7] |
|:--:|
| *Center Image (Final Result -- YUV, Cropped, Resized, Blurred)* |

| ![alt text][image8] |
|:--:|
| *Right Image (Final Result -- YUV, Cropped, Resized, Blurred)* |

| ![alt text][image9] |
|:--:|
| *Left Image (Final Result -- YUV, Cropped, Resized, Blurred)* |

I finally randomly shuffled the data set and put 9% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 81 as evidenced by my saved model with the best validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
