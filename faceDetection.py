"""First we need to import cv2 library to process image"""

import cv2

"""Using cv2 we are processing the xml file which huge contains data of faces"""

face_cascade=cv2.CascadeClassifier(r"C:\Users\2116060\OneDrive - Cognizant\Documents\classandobjects\classespython\Lib\site-packages\six-1.16.0.dist-info\haarcascade_frontalface_default.xml")

"""processing the image and converting it into numpy array it will be in rbg"""

img=cv2.imread(r"images_sample.jpg")

"""processing the image and converting it into numpy array and it will be in grey scale"""

grey_img=cv2.imread(r"images_sample.jpg",0)

"""The face_cascade data is conpared with the image and detecting the faces.
The scale factor 1.05 means for every search it will reduce the size into half and it will search"""

faces=face_cascade.detectMultiScale(grey_img,scaleFactor=1.05,minNeighbors=1)

"""x and y is the axis and w ,h is the weight and height
(0,255,0) is the color of the rectange and 5 is the intensity of the rectange
"""

for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)


"""The image size is reduced by 1/3 percent"""
resize=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

"""The image is shown using imshow"""

cv2.imshow("grey_img",resize)

print(faces)

"""The waitkey and destroyAllWindows help to pause the image in the screen if we press any key then the image will close"""

cv2.waitKey(0)
cv2.destroyAllWindows()
