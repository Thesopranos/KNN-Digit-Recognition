# -*- coding: utf-8 -*-
import cv2
import numpy as np

# read image and convert it to grayscale
img = cv2.imread("digits.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# split the image into cells (50x100 cells)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)

# train and test sets
train = x[:, :50].reshape(-1,400).astype(np.float32)
test = x[:, 50:100].reshape(-1,400).astype(np.float32)

# assign labels for the train and test sets
k = np.arange(10)
train_responses = np.repeat(k,250).reshape(-1,1)
test_responses = np.repeat(k,250).reshape(-1,1)

# train the k-nearest neighbor model
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_responses)

# predict the digits using the trained model
ret, results, neighbours, distance = knn.findNearest(test, 3)

# calculate the accuracy of the model
matches = test_responses == results
correct = np.count_nonzero(matches)
accuracy = (correct/results.size)*100

# function to predict the digit drawn on the screen
def test(img):
    img = cv2.medianBlur(img, 21) # blur the image
    img = cv2.dilate(img, np.ones((15,15),np.uint8)) # dilate the image
    cv2.imshow("img", img)
    img = cv2.resize(img, (20,20)).reshape(-1,400).astype(np.float32)
    ret, _, _, _ = knn.findNearest(img, 5)
    cv2.putText(img2, str(int(ret)), (100,300), font, 10, 255, 4, cv2.LINE_AA)
    return ret

# variables for the drawing function
drawing = False
last_position = None
first_dot = False

# font for displaying the prediction
font = cv2.FONT_HERSHEY_SIMPLEX

# create an empty image
img = np.ones((400,400), np.uint8)

# function to draw a round line
def roundline(start, end):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        cv2.circle(img,(x, y),10,255,-1)
        
# function to handle mouse events for drawing
def draw(event, x, y, flags, param):
    global drawing, last_position
    if event == cv2.EVENT_LBUTTONDOWN:
        last_position = x,y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roundline((x,y), last_position)
            last_position = x,y
        else:
            pass
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# create a window for the drawing canvas
cv2.namedWindow("paint")      
cv2.setMouseCallback("paint", draw)

# loop to continuously display the drawing canvas and the prediction
while True:
    img2 = np.ones((400,400),np.uint8)
    key = cv2.waitKey(33) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        img[:,:] = 0
    test(img)
    cv2.imshow("paint",img)
    cv2.imshow("result",img2)
    
cv2.destroyAllWindows()
