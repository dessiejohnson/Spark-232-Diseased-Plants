# Spark-232-Diseased-Plants
**Abstract**

As our climate changes drastically, the vitality of crops becomes unstable. With disease spreading and ravaging the world’s crops, there has to be a way to identify diseased plants at a large scale. Machine learning methods have been used to classify images of handwritten digits, faces, and X-rays. We can apply similar techniques to images of plants as well. In this project, binary image classification of diseased versus healthy plants and multiclass image classification of different types of plants are performed. The data used is a 19.47 GB dataset from Kaggle containing images of 13 diseased and healthy plants. A distributed processing method is used due to the dataset's large size and the need to preprocess and work with thousands of images. This project will help deepen the understanding of how machine learning algorithms can play an important role in disease detection and potentially be vital to humanity's survival. 

Link to Dataset: https://www.kaggle.com/datasets/samareshkumar/multipleplantdiseases


**Pre-Processing The Data**

From the exploratory analysis, we found that there is no missing data in our dataset; therefore, no preprocessing needs to be done in that regard. 
However, there are significant data imbalances that we observed in the initial visualizations. With regard to our multiclass classification problem, tomato images dominate the dataset, but there are not many chili or cauliflower images at all in comparison. To address this imbalance, we can undersample our majority class, tomato. To achieve this we can randomly sample tomato images and delete them so that they do not completely dominate our dataset. Another option is to oversample our minority classes to increase their size relative to the larger classes. We can perform these operations in Spark using .sample(), .filter(), and .union(). There is also a greater quantity of diseased plant images than healthy plant images for our binary classification problem. We can apply similar methods of undersampling the majority class (diseased plants) and oversampling the minority class (healthy plants). In addition, the images in the dataset have different sizes.

All images must be resized to 224x224 dimensions before training. We might create a UDF to access the images, resize them, and store the resized versions as bytes in the dataframe using .withColumn(). We also must split the data into train and test sets. We can do this using .randomSplit() in Spark. We should perform this before oversampling the majority and undersampling the minority because we only want to apply those to the training set and leave the test set as is.


**First Distributed Model**



**Spark UI Verification**
![Executors](images/spark_ui_executors.h)

