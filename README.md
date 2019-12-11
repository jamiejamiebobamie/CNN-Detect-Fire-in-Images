# Fire-Detection-Image-Dataset

This is a repository for my Data Science 2.2: Introduction to Neural Networks
class final.

It makes use of images and starter code provided by the professor:
    rar_image_files
    image_data_prep.ipynb

The professor provided a roadmap for students to follow:
    In-class_guidelines.jpg

The main work for the project is contained in:
    fire_image_classification.py

This file contains lines of code that train and pickle a Convolutional Neural
Network to classify the images as containing a fire (a label of 1) or no fire
(a label of 0).

The trained model has an accuracy of 91%.

Please refer to the comments in fire_image_classification.py for an in-depth
breakdown of my model and process.

NOTE: The fire_image_classification.py lacks a confusion matrix, but uses data
augmentation to balance the dataset.
