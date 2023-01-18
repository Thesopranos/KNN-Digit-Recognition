# Digit Recognition using OpenCV
This script uses the OpenCV library in Python to recognize digits drawn on a canvas. The model is trained using the K-Nearest Neighbors algorithm.

# Requirements
Python 3
OpenCV 4
Numpy  

![Screenshot_1](https://user-images.githubusercontent.com/68299931/213156550-8a76c941-96e7-4507-870a-6886381d4fcb.png)
left window = predict  
mid window = image after dilation  
right widow = paint  
# Usage

1. Run the script using the command python scriptname.py  
2. Draw a digit on the canvas by clicking and dragging the mouse.  
3. Press 'q' to quit the program or 'c' to clear the canvas.  
4. The prediction of the digit will be displayed on the window named "result"  

# Data
The dataset used for training the model is the digits.png dataset. It contains 5000 images of size 20x20 pixels of handwritten digits.

# Accuracy
The accuracy of the model is calculated as the ratio of the number of correctly predicted digits to the total number of digits in the test set. The accuracy of the model is around 90%.
