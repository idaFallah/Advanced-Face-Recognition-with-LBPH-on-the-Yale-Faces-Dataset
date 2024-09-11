
# import libs
from PIL import Image
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/content/drive')

import zipfile
path = '/content/drive/MyDrive/Computer Vision/Datasets/yalefaces.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

import os
print(os.listdir('/content/yalefaces/train'))

#preprocessing

def get_image_data():
  paths = [os.path.join('/content/yalefaces/train', f) for f in os.listdir('/content/yalefaces/train')]
  #print(path)
  faces = []
  ids = []
  for path in paths:
    #print(path)
    image = Image.open(path).convert('L') # L here means grayscale
    #print(type(image))
    image_np = np.array(image, 'uint8')
    #print(type(image_np))
    #id = os.path.split(path) # index 0 was the path and index 1 was the id
    id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    #print(id)
    ids.append(id)
    faces.append(image_np)

  return np.array(ids), faces

ids, faces = get_image_data()

ids

len(ids)

faces

len(faces)

faces[0], faces[0].shape  # one single image and it's size

243 * 320  # number of pixels in each image

# training the model = LBPH classifier, the default version of this classifer is 8*8, yani grids the pic into 64 grids

# lbph params by default = radius=1 , neighbors=8, grid_x=8 & y=8, threshold (or confidence)=1.79...

lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius=4, neighbors=14, grid_x=9, grid_y=9)
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')  # stores the histograms for each image

# recognizing faces
lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('/content/lbph_classifier.yml')

test_image = '/content/yalefaces/train/subject13.noglasses.gif'

image = Image.open(test_image).convert('L')
image_np = np.array(image, 'uint8')   #uint8 means pixels are in int format
image_np

image_np.shape

prediction = lbph_classifier.predict(image_np)
prediction # index 0 is the id, index 1 is the confidence

prediction[0]

expected_output = int(os.path.split(test_image)[1].split('.')[0].replace('subject', ''))
expected_output

cv2.putText(image_np, 'Pred: '+ str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.putText(image_np, 'Exp: '+ str(expected_output), (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

cv2_imshow(image_np)

# evaluating the model : lbph classifier

paths = [os.path.join('/content/yalefaces/test', f) for f in os.listdir('/content/yalefaces/test')]
# joining the path of each file to the end of test file path to create their own path & store it
# f here means file
predictions = []
expected_outputs = []
for path in paths:
  #print(path)
  image = Image.open(path).convert('L')
  image_np = np.array(image, 'uint8')
  prediction, _ = lbph_face_classifier.predict(image_np)
  expected_output = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

  predictions.append(prediction)
  expected_outputs.append(expected_output)

type(predictions)

predictions = np.array(predictions)
expected_outputs = np.array(expected_outputs)

type(predictions)

predictions

expected_outputs

from sklearn.metrics import accuracy_score
accuracy_score(expected_outputs, predictions)

len(predictions) # since the length is small 0.67% is  not a good acc score

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected_outputs, predictions)
cm

import seaborn
seaborn.heatmap(cm, annot=True);  # ; to hide the excess message

! nvcc --version



















