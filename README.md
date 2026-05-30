# DSC232R Spark Group Project: Diseased Plants
## Introduction
As our climate changes drastically, the vitality of crops becomes unstable. With disease spreading and ravaging the world’s crops, there has to be a way to identify diseased plants at a large scale. Machine learning methods have been used to classify images of handwritten digits, faces, and X-rays. We can apply similar techniques to images of plants as well. In this project, binary image classification of diseased versus healthy plants and multiclass image classification of different types of plants are performed. The data used is a 19.47 GB dataset from Kaggle containing images of 13 diseased and healthy plants. A distributed computing framework is used due to the dataset's large size and the need to preprocess and work with thousands of images. The dataset size makes traditional single-machine processing inefficient and difficult to scale. Spark enables distributed storage and parallel computation, allowing efficient preprocessing, scalable machine learning workflows, fault tolerance, and significantly reduced execution time.This project will help deepen the understanding of how machine learning algorithms can play an important role in disease detection and potentially be vital to humanity's survival.

Link to dataset: https://www.kaggle.com/datasets/samareshkumar/multipleplantdiseases

## Methods
### Data Exploration

### Pre-Processing

To begin preprocessing the data for model training, we created a Spark dataframe with four columns: file_path, class_label, plant, and disease. To create the class_label, plant, and disease columns, we split the file_path string at different parts to isolate what we needed. 

```python
metadata = spark.read.format("binaryFile") \
    .option("recursiveFileLookup", "true") \
    .load(data_path) \
    .withColumn("file_path", F.col("path")) \
    .withColumn("class_label", F.element_at(F.split("path", "/"), -2)) \
    .withColumn("plant", split.getItem(0)) \
    .withColumn("disease", split.getItem(1)) \
    .select("file_path", "class_label", "plant", "disease")
```

We created a new column called "disease_clean" to serve as the label for our binary classification problem addressed by Model 1. This new column strictly labeled an image as either healthy or diseased since the dataset originally included the specific disease type. The "plant" column served as the label for our multiclass classification problem addressed by Model 2.

```python
metadata = metadata.withColumn(
    "disease_clean",
    F.when(F.lower(F.col("disease")).rlike("healthy|fresh_leaf"), "healthy")
     .otherwise(F.col("disease"))
).withColumn("health", \
 F.when(F.lower(F.col("disease_clean"))== "healthy", "healthy").otherwise("diseased"))\
.select("file_path", "plant", "health").cache()
```

No null values were present in the dataset, so we moved on to splitting the data into training, validation, and test sets for each model. There were significant data imbalances; images of diseased plants vastly outnumbered images of healthy plants. Images of tomatoes were overrepresented while other plants like chilis and cauliflowers were severely underrepresented. To address this, we used stratified random sampling to ensure that the smaller classes were represented in our training dataset.

```python
w_d = Window.partitionBy("health").orderBy(F.rand(seed=1))
metadata_d = metadata.withColumn("row_num", F.row_number().over(w_d))\
            .withColumn("count", F.count("*").over(Window.partitionBy("health")))
metadata_d = metadata_d.withColumn("split", F.when(F.col("row_num") <= 0.7* F.col("count"), "train")\
            .when(F.col("row_num") <= 0.85 * F.col("count"), "val").otherwise("test"))
train_d = metadata_d.filter(F.col("split") == "train").select("file_path", "health")
val_d = metadata_d.filter(F.col("split") == "val").select("file_path", "health")
test_d = metadata_d.filter(F.col("split") == "test").select("file_path", "health")
```

Next, we label encoded. Because of the data imbalance, we also added a new column to the training datasets only. For the binary classification problem, we added a “weight” column that weighted the minority class, healthy plant images, more than the majority class. For the multiclass classification problem, we added a “weight” column that weighted the minority classes like chili, cauliflower, and cabbage more strongly. These new columns would be used when training the models.

```python
class_counts_d = train_d.groupBy("label").count()

total_d = train_d.count()
num_classes_d = class_counts_d.count()

weights_d = class_counts_d.withColumn(
    "weight",
    F.lit(train_d.count()) / (F.lit(class_counts_d.count()) * F.col("count"))
).select("label", "weight")

train_d = train_d.join(weights_d, on="label", how="left")
 ```

Finally, we preprocessed the images. We brought in the image data and joined them to our train, validation, and test sets. We used a pandas UDF to ensure all images were the same size and correct format for the models by resizing, normalizing, and flattening the images into vectors. 

```python
@pandas_udf(ArrayType(FloatType()))
def decode_image(contents: pd.Series) -> pd.Series:
    result = []

    for content in contents:
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            # small because im having memory/speed problems
            img = img.resize((32,32))
            arr = np.asarray(img, dtype = np.float32) / 255.0
            result.append(arr.reshape(-1).tolist())
        except Exception:
            result.append(None)

    return pd.Series(result)
 ```

### Model 1

### Model 2


## Results


## Discussion

For preprocessing, we waited to bring in the image data until the final step before starting the model building. This way, we could do transformations on the data like stratified random sampling and splitting into the train, validation, and test sets more efficiently. The majority of the size of the dataset came from the images, so until we needed to use the data for the model building, we left it out.


## Conclusion
We learned that big data processing requires time. Even simply loading in the data required much more time than expected. Because of this, we had to think and act more carefully when it came to checking our work and tuning the models. Attempting to count the total number of null values to check our work, which would be a simple task in normal data preprocessing, might cause an out-of-memory error or timeout. Adjusting model parameters and retraining could be painstakingly slow. So, we had to think carefully about what we wanted to run.

We would like to explore different ways of preprocessing the image data, like using pretrained embeddings such as ResNet rather than our current approach. This may increase our model performance.  


## Statement of Collaboration
| Name            | Title                          | Contribution                               |
|-----------------|--------------------------------|--------------------------------------------|
| Desirè Johnson  | Project Manager, Writer, Coder | Abstract, data exploration, final report   |
| Angela Fan      | Writer, Coder                  | Abstract, pre-processing, final report     |
| Megan Felix     | Coder, Writer                  | Model 1, final report                      |
| Ashley Chong    | Coder, Writer                  | Model 2, final report                      |

