# Tomato_leaf_disease_classification
I've developed this deep learning project using TensorFlow and PyQt5 to detect various kinds to diseases of tomato plants from images of tomato leaves

![](https://github.com/sumanta1997/Tomato_leaf_disease_classification/blob/master/tomato.gif)

Tomato is the most popular crop in the world and in every kitchen, 

It is found in different forms irrespective of the cuisine. 

After potato and sweet potato, it is the crop which is cultivated worldwide. 

However, the quality and quantity of tomato crop goes down due to the various kinds of diseases.



It can detect various diseases such as  Bacterial Spot ,Early Blight ,Late Blight .Yellow Leaf Curl Virus , Mosaic Virus from images of tomato leaves
 
I've downloaded the tomato leaf disease classfication dataset from Kaggle and divided it into three parts Training Set,Validation Set and Test Set.

I've used Tensorflow 2.8.0 as the Deep learning framework

Here transfer learning approach has been applied and final layer of EfficientNetV2B0 has been trained for 40 epochs


To run this application, following libraries must be installed on the system along with Python 3: 

1.PyQt5 pip install command: pip install PyQt5

2.TensorFlow

pip install command : pip install tensorflow

3.Numpy

pip install command: pip install numpy

Matplotlib(optional)
pip install command: pip install matplotlib

After installing libraries, users have to follow these steps:

1.Open command prompt or bash shell and browse to the directory where final.py is located using cd command

2.then run the final.py by using the following command "python tomato_disease.py"

3.Click on browse button

4.Select and image

5.Click on classify button and result will pop up in bottom textedit window

6.If you want to train the model, click on train model
