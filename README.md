### Garbage-Classifier-using-CNN-with-PyTorch

<Note: Run this project in Google Collaboratory since it uses google drive>

**DATASET LINK**: https://drive.google.com/drive/folders/1GDSE1CVpuOUZXx9LyQ2kCfESix40-1Ul?usp=sharing

We gathered a dataset containing close to 3000 total images of plastic, metal and paper waste. To use it, you need to create a shortcut of the "Final_Dataset" folder to "My Drive". Upon running the code, you will be asked to choose your google account whose Drive will be mounted.

**Data Pre-Processing:**

This code is performing the following pre-processing steps on the image dataset:

* Reading images using cv2.imread() method
* Converting the color format from BGR to RGB using cv2.COLOR_BGR2RGB method
* Resizing the images to a fixed size (IMG_HEIGHT, IMG_WIDTH) using cv2.resize() method with interpolation=cv2.INTER_AREA
* Converting the images to a numpy array using np.array() method
* Converting the data type of images to float64 using image.astype('float64') method
* Normalizing the pixel values in the range [0, 1] by dividing each pixel value by 255
* Storing the images as numpy arrays in img_data_array
* Reshape image to (200,200,3)
* Storing the class labels in class_name

**Model Training & Testing**

This code segment defines and sets up a Convolutional Neural Network (CNN) model for image classification. Here are the steps it performs:

* The code defines a class called `CNNNet` that inherits from the `nn.Module` class, which is the base class for all neural network modules in PyTorch.
* Inside the `CNNNet` class, the model architecture is defined in the `__init__` method. The architecture consists of two main parts: `cnn_layers` and `linear_layers`.
* The `cnn_layers` are defined using the `nn.Sequential` container, which allows stacking multiple layers sequentially. The layers in `cnn_layers` are as follows:
   - `nn.Conv2d`: A 2D convolutional layer with 3 input channels, 16 output channels, a kernel size of (5, 5), a stride of (2, 2), and padding of (2, 2).
   - `nn.ReLU`: Activation function ReLU (Rectified Linear Unit) is applied element-wise to introduce non-linearity.
   - `nn.MaxPool2d`: A 2D max pooling layer with a kernel size of 2 and stride 2, which reduces the spatial dimensions of the input by taking the maximum value in each pooling region.
   - Another `nn.Conv2d` layer with 16 input channels (output from the previous layer), 3 output channels, a kernel size of (50, 50), and a stride of (1, 1).
   - Another `nn.MaxPool2d` layer with a kernel size of 1 and stride 1, effectively not reducing the spatial dimensions further.
* The `linear_layers` are defined using another `nn.Sequential` container, containing a single `nn.Linear` layer. This layer takes the output from the previous layers and performs a linear transformation.
* The `forward` method of the `CNNNet` class defines the forward pass of the model. It takes an input tensor `x` and passes it through the `cnn_layers` sequentially. Then, the output is reshaped using `x.view(x.size(0), -1)` to flatten the tensor, preserving the batch size but collapsing the spatial dimensions. Finally, the flattened tensor is passed through the `linear_layers` to produce the output.
* After defining the model architecture, an instance of the `CNNNet` class is created and assigned to the `model` variable.
* The code defines an optimizer using stochastic gradient descent (SGD) with a learning rate of 0.0001. The optimizer is initialized with the parameters of the `model` using `model.parameters()`.
* A loss function is defined using cross-entropy loss (`nn.CrossEntropyLoss`). This loss function is commonly used for multi-class classification problems.
* The model is then trained and tested for the given architecture, epochs = 100
