import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import f1_score 
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
plate_cascade = cv2.CascadeClassifier('india.xml')
def detect_plate(img, text=''):
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7)
    for (x,y,w,h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :]
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2,y), (x+w-3, y+h-5), (51,181,155), 3) 
    if text!='':
        plate_img = cv2.putText(plate_img, text, (x-w//2,y-h//2), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (51,181,155), 1, cv2.LINE_AA)
    return plate_img, plate
def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()
img = cv2.imread('audi.jpg')
display(img, 'input image')
output_img, plate = detect_plate(img)
display(output_img, 'detected license plate in the input image')
display(plate, 'extracted license plate from the image')
im=plate
inverted_img=cv2.bitwise_not(im)
cv2.imshow('inverted image', inverted_img)
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image=grayscale(im)
cv2.imshow('grayscale image', gray_image)
thresh, im_bw = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('bw image', im_bw)
def noise_removal(image):
    import numpy as np
    kernel=np.ones((1, 1), np.uint8)
    image=cv2.dilate(image, kernel, iterations=1)
    kernel=np.ones((1, 1), np.uint8)
    image=cv2.erode(image, kernel, iterations=1)
    image=cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image=cv2.medianBlur(image, 1)
    return image
no_noise=noise_removal(im_bw)
cv2.imshow('No noise image', no_noise)
import pytesseract
ocr_result=pytesseract.image_to_string(no_noise)
print(ocr_result)
cv2.waitKey(0)
cv2.destroyAllWindows()