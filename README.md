# DSC232R Spark Group Project: Diseased Plants
## Introduction
As our climate changes drastically, the vitality of crops becomes unstable. With disease spreading and ravaging the world’s crops, there has to be a way to identify diseased plants at a large scale. Machine learning methods have been used to classify images of handwritten digits, faces, and X-rays. We can apply similar techniques to images of plants as well. In this project, binary image classification of diseased versus healthy plants and multiclass image classification of different types of plants are performed. The data used is a 19.47 GB dataset from Kaggle containing images of 13 diseased and healthy plants. A distributed computing framework is used due to the dataset's large size and the need to preprocess and work with thousands of images. The dataset size makes traditional single-machine processing inefficient and difficult to scale. Spark enables distributed storage and parallel computation, allowing efficient preprocessing, scalable machine learning workflows, fault tolerance, and significantly reduced execution time.This project will help deepen the understanding of how machine learning algorithms can play an important role in disease detection and potentially be vital to humanity's survival.

Link to dataset: https://www.kaggle.com/datasets/samareshkumar/multipleplantdiseases

## Methods
### Data Exploration

### Pre-Processing

To begin preprocessing the data for model training, we created a Spark dataframe with four columns: file_path, class_label, plant, and disease. To create the latter 3 columns, we split the file_path string at different parts to isolate what we needed. We created a new column called "disease_clean" to serve as the label for our binary classification problem addressed by Model 1. This new column strictly labeled an image as either healthy or diseased since the dataset originally included the specific disease type. The "plant" column served as the label for our multiclass classification problem addressed by Model 2. 

No null values were present in the dataset, so we moved on to splitting the data into training, validation, and test sets. There were significant data imbalances; images of diseased plants vastly outnumbered images of healthy plants for our binary classification problem. Images of tomatoes were extremely overrepresented while other plants like chilis were severely underrepresented. To address this, we used stratified random sampling to ensure that the smaller classes were still represented in our training dataset.





### Model 1

### Model 2


## Results


## Discussion


## Conclusion


## Statement of Collaboration
| Name            | Title                          | Contribution                               |
|-----------------|--------------------------------|--------------------------------------------|
| Desirè Johnson  | Project Manager, Writer, Coder | Abstract, data exploration, final report   |
| Angela Fan      | Writer, Coder                  | Abstract, pre-processing, final report     |
| Megan Felix     | Coder, Writer                  | Model 1, final report                      |
| Ashley Chong    | Coder, Writer                  | Model 2, final report                      |

