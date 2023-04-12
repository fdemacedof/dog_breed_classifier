# Dog Breed Classifier

As part of the Udacity Data Scietist Nanodegree’s final project, I have developed an algorithm using Convolutional Neural Networks (CNN) to classify dogs according to their breed. The algorithm also works on human faces — telling the which dog breed the human most resembles.

Writing a program to identify dog breeds from images might be tricky — even humans might have a hard time telling the breed of a dog! In order to do it, we have to find ways to extract relevante features from the image — and use those features to predict the most probable breed for the dog.

I have followed the steps bellow:

- Use Haar feature-based cascade classifiers to detect human faces in images;
- Use ResNet50 to detect dogs in images;
- Build a CNN from scratch;
- Build a CNN with transfer learning, using ResNet50 to classify dog breeds;
- Implement an algorithm that classifies dogs breeds from images.

I have written an [article on medium](https://medium.com/@fdemacedof/identifying-dog-breeds-with-convolutional-neural-networks-cnn-67363018957a) explaining the whole process step by step. And [another article](https://medium.com/@fdemacedof/udacity-nanodegree-capstone-project-dog-breed-classifier-bfe24f582c80) on the technical aspects of the project.

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `data/`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `data/`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `bottleneck_features/`.

# DogBreedClassifier Class

You can find the final algorith, as a Class, in the DogBreedClassifier.py file. Use the _classify(image_path)_ method to use it:
### Using DogBreedClassifier

```	
from DogBreedClassifier import DogBreedClassifier

classifier = DogBreedClassifier()
classifier.classify("path/to/image")
```	
