Multi-class Classifier in PyCaret for Multi-Class Classification - Base problem category as per Ready Tensor specifications.

* pycaret
* sklearn
* python
* pandas
* numpy
* scikit-optimize
* FastAPI
* nginx
* uvicorn
* docker
* multi-class classification

This is an AutoML Multi-class Classifier that uses pycaret to automatically select the best model(s) for the given dataset(s)

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as:
- dna_splice_junction
- gesture_phase
- ipums_census_small
- landsat_satellite
- page_blocks
- primary_tumor
- soybean_disease
- spotify_genre
- steel_plate_fault
- vehicle_silhouettes

This Multi-class Classifier is written using Python as its programming language. SciKitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. FastAPI + Nginx + uvicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.





