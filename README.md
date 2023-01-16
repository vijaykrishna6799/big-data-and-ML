# big-data-and-ML
projects implemented as a part of big data and machine learning course spring 2022

for detailed explaination go to main_document.pdf

## Languages used :
- Python


Images are taken from the FIDS30 Dataset : https://www.vicos.si/resources/fids30/

- Fruit images are taken for analysis, by using cv2 package in python we read the images.
- Extracting the Gray scale of the images as well as the RGB channels of each image.

- Images are resized based on the requirement (resized so that the pixels are divisible by 9 and height of the image = 261 pixels)
- resized images are then used to generate the feature space (creating the block feature vectors) and saving as a .csv file
- Each image is converted to 81 features and we add a label as the 82nd feature.
- for all the images selected we also generate the sliding block features for the height of 261
- For all the block and sliding block vectors we create Statistical analysis (histograms, mean plots, scatter plots etc..)
- we merge all the 3 images datasets to create a FEATURE SPACE (both for the block feature vectors as well as the sliding block feature vectors)


- For the feature spaces created we do the LASSO and RANDOM FOREST regression
- classification is done by splitting the train and test dataset to 78:22 respectively
- by using the lasso regression and random forest regression we create a predicted label for the dataset and then implemented the confusion matrix on the predicted label and the actual label.
- for the merged datasets we extract the Quantitative measures and the confusion matrix to estimate the model performance.
- for the 81 features we generated, we add the actual label as the 82nd feature and predicted label is added as the 83rd label.
- we compare the performances of both lasso regression and the random forest classifier for both overlapping and non-overlapping datasets for all the merged datasets.
- we implement the random forest classifier on the huge datasets (say dataset of 100 images) using a big data system like DataBricks.
