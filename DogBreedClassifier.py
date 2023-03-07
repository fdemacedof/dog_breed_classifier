### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

import keras
from keras.preprocessing import image                  
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tqdm import tqdm
import cv2                
import matplotlib.pyplot as plt 
import numpy as np
from glob import glob

class DogBreedClassifier():
    '''
    Detect and classify dog breeds from images.
    
    Attributes
    ----------
    model: keras.models.Sequential
        CNN model for classifying dog breeds - uses ResNet50 bottleneck features.
    
    Methods
    -------
    classify(image_path)
        if a dog is detected in the image, return the predicted breed;
        if a human is detected in the image, return the resembling dog breed;
        if neither is detected in the image, provide output that indicates an error.
    
    '''    
    
    def __init__(self):
        self.model = keras.models.load_model('saved_models/DogResnet50_model')
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        self.ResNet50_model = ResNet50(weights='imagenet')
        self.dog_names = dog_names = [item[20:-1] for item in sorted(glob("../../../data/dog_images/train/*/"))]
        
    def __str__(self):
        return f'{self.model}'

    def __path_to_tensor(self, img_path):
        '''
        Takes a string-valued file path to a color image as input and returns a 4D tensor.
        
        INPUT:
        img_path (str) path to image
        
        RETURNS:
        tensor (numpy.ndarray) 4D image tensor
        '''
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        tensor = np.expand_dims(x, axis=0) 
        return tensor

    def __extract_Resnet50(self, tensor):
        '''
        Takes a tensor and extract Resnet50 bottleneck features.
        
        INPUT:
        tensor (numpy.ndarray) 4D image tensor.
        
        RETURNS:
        (numpy.ndarray) array of extracted bottleneck features for Resnet50        
        '''
        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
        
    def __face_detector(self, img_path):
        '''
        Takes a path to image file, loads the image and returns True if any human face is detected.
        
        INPUT:
        img_path (str) path to image file.
        
        RETURNS:
        (bool) True if any face is detected, False otherwise.        
        '''
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)        
        return len(faces) > 0
    
    def __ResNet50_predict_labels(self, img_path):
        '''
        Takes a path to image file, preprocesses input, uses Resnet50 to classify the image and returns Resnet50 label with maximum probability.
        
        INPUT:
        img_path (str) path to image file.
        
        RETURNS:
        (int) label with maximum probability as classified by Resnet50_model.predict().        
        '''
        # returns prediction vector for image located at img_path
        img = preprocess_input(self.__path_to_tensor(img_path))
        return np.argmax(self.ResNet50_model.predict(img))

    def __dog_detector(self, img_path):
        '''
        Takes path to image file and returns True if a dog is detected.
        
        INPUT:
        img_path (str) path to image file.
        
        RETURNS:
        (bool) True if any dog is detected and False otherwise.
        '''
        prediction = self.__ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))
    
    
    def __DogResNet50_predict_breed(self, img_path):
        '''
        Takes path to image with a dog and human face and determines dog breed.
        
        INPUT:
        img_path (str) path to image file.
        
        RETURNS:
        dod_breed (str) name of a dog breed.
        '''
        # extract bottleneck features
        bottleneck_feature = self.__extract_Resnet50(self.__path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self.model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        dog_breed = self.dog_names[np.argmax(predicted_vector)][15:].lower().replace("_"," ")
        return dog_breed

    def classify(self, img_path):
        '''
        Takes path to image file, determines if there is a dog or human face, prints the image and the dog breed (or dog breed thar most resembles human face).
        
        INPUT:
        img_path (str) path to image file.
        '''
        # print picture
        # print image
        
        img = cv2.imread(img_path)
        # convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # display the image
        plt.imshow(cv_rgb)
        plt.show()
        
        # checks if there is a dog or human face in the image
        dog_detected = self.__dog_detector(img_path)
        face_detected = self.__face_detector(img_path)
        if (dog_detected or face_detected) == 0:
            return "ERROR: no dog nor human face found - aborting..." 
        if dog_detected:
            print("found dog in picture!")        
        if face_detected:
            print("found human face in picture!")       
        
        breed = self.__DogResNet50_predict_breed(img_path)
        
        if dog_detected:
            print(f"dog in image is a {breed}")
        if face_detected:
            print(f"human in picture resembles a {breed}")