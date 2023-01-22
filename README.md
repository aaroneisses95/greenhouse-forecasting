# Assignment: Air-temperature model for Greenhouse climate

## 1. Introduction
Plant development and growth is a result of the direct environment of the plant. The environment consists of parameters such as air temperature, humidity, CO2 concentration and access to water and nutrients. These parameters are influenced by conditions outside, as well as control mechanisms inside the greenhouse. To achieve the right environment, the grower manipulates mechanical components of the greenhouse based on forecasted external weather conditions. These components include but are not limited to: screens, ventilation (through opening windows) and heating systems. 

This project contains the code to train a time-series model that can predict the air temperature (_T<sub>air</sub>_) in the greenhouse based on the above-described features. The following README.md describes the repo structure, how the model works, how to run it and how to test it on new data.

## 2. Repo structure

In this section there is more information about the important files and folders within this repo. 

- `data`: Folder of the input data for the model. Can be found through this link (https://drive.google.com/drive/folders/1USB4dOK9UodNt1mPiLf8-jE0TkuS94-f?usp=sharing)
- `notebooks/exploration.ipynb`: Notebook that I used for exploring the data, feature selection, feature engineering and finding a suitable model.
- `notebooks/runner.ipynb`: Notebook that can be used to load the trained model and test your new data.
- `predictor`: Python package that contains the ingestion and preprocessing functionality 
- `MLproject`: File that is used by MLflow to define the structure and configuration of a machine learning project.
- `requirements_prod.txt`: The txt file with the requirements that need to be installed to train and log the model. 
- `train.py`: The train end-point for the MLflow run. 

## 3. The model
In this section the details of the model are explained. To get a more complete image, it could be helpful to run the `exploration.ipynb` notebook in the `notebooks` folder. 


### 3.1 Data selection
In the scope of this project only the weather and the greenhouse climate dataset are used. These seemed to contain the most important features, but perhaps there were also other valuable features in the other data sets. 

### 3.2 Feature selection
To train a good model it is important to come to a good selection of features. Even though this will probably differ between the datasets, in the scope of this project the Automatoes dataset has been used to come to a feature selection strategy.

#### 3.2.1 Missing values
As a first step, features are removed that have lots of missing values. In the exploration it became clear that the features `int_blue_vip`, `int_farred_vip`, `int_red_vip`, `int_white_vip` and `t_vent_sp` missed a significant amount of values. These features are therefore removed from the dataset.

#### 3.2.2 Correlation with target
For the second step, features are been removed that have little correlation to the target. In the exploration it became clear that the features `PARout`, `Rhout`, `Tout`, `HumDef`, `Tot_PAR`, `t_heat_sp` and `t_heat_vip` had a significant amount of correlation (absolute value > 0.5). Therefore these features are the ones that remain.

#### 3.2.3 Correlation among features
In the third step, features are removed that have strong correlation to others. In the exploration it showed that there are two sets of features that have a strong correlation to the other:
- `Iglob`, `PARout` & `Tot_PAR`
- `t_heat_sp` & `t_heat_vip`

In this case the features `PARout`, `Tot_PAR` and `t_heat_vip` have been dropped.

#### 3.2.4 After feature selection

After the feature selection step, the following features are still remaining:
- `Iglob`
- `Rhout`
- `Tout`
- `HumDef`
- `t_heat_sp`

### 3.3 Feature engineering
To improve the model it might be a good idea to add features that seem valuable. In the scope of this project, only features have been added related to time. In this case, only three features are added:
- `t`: The period number. The first data point starts with t=-3984 and increases every row by 1 until you are at the end current point in time (t=0) 
- `hour_of_day`: Hour of the day (int)
- `month`: Month of the year (int)

After the feature engineering step, the following features are still remaining:
- `Iglob`
- `Rhout`
- `Tout`
- `HumDef`
- `t_heat_sp`
- `t`
- `hour_of_day`
- `month`

### 3.3 Model selection
To select a good model, several different models have been tried out with different hyperparameters.

#### 3.3.1 Train and test data
Since our model needs to predict up to 24 hours in advance (`t<=24`) and more recent data is very relevant, the training data contains the whole dataset except for the last day and the test data is the last day of the dataset. 

#### 3.3.2 RepeatingBasisFunction
Included in the models is the RepeatingBasisFunction method from the library called [Scikit-Lego](https://scikit-lego.netlify.app/index.html). This is a transformer that can be used for features with some form of circularity. It deals with the problem that 23:00 is as close to 00:00 as 01:00 even though numerically their distance is different. This is why it is also included for the `hour_of_day` feature.

#### 3.3.3 TimeSeriesSplit
Included in the models is the TimeSeriesSplit method. This is used because in a time-series model, you cannot use the normal cross-validation strategy that is used in GridSearchCV. It is important that you consider the chronological order of the data. For more information you can go to this [link](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

#### 3.3.4 Choosing the model
Two models have been tried out:

- Linear regression
- Random Forest

Based on the metrics, the Random Forest model seemed to be working well so this has been chosen.


## 4. How to run the model
To execute the code the following steps need to be followed.

### 4.1 Setting up the repo
When you run an MLflow project, a new folder with be created called `mlruns` with information about that run. For that reason we want to create a new directory (for example `source_project`) and clone our repo in there:

```
# Make a new directory
mkdir source_project

# Go into that directory
cd source_project

# Clone this repo
git clone https://github.com/aaroneisses95/source-assignment-data-science.git
```

You should have the following structure:
- ../source_project
    - ../source_project/source-assignment-data-science


### 4.2 Setting up a new virtual environment
In the next step we will create a new virtual environment and install the correct packages. In this example I will use an PyEnv environment called `source_env` but you can of course use something else:

```
# Create a new virtual environment with python version 3.8.10
pyenv virtualenv 3.8.10 source_env

# Activate your virtual environment
pyenv shell source_env
```

Now we will go into the repo and install the correct packages:

```
# Enter the repo
cd source-assignment-data-science

# Use the `make install` command to install the necessary packages
make install
```

### 4.3 Training the model
Now that we've installed the correct dependencies, we can run our model using MLflow. To that we first have to go back to the parent directory of our repo (`../source_project`):

```
# Go one directory back
cd ..
```

When we've done that, we can run our model with the following command:

```
# Run the model
mlflow run source-assignment-data-science --env-manager=local --entry-point train --param-list team=Automatoes
```

This commands triggers an MLflow run with the following parameters:
- `env-manager=local`: We will use our current environment. This shouldn't be changed.
- `entry-point=train`: We will trigger our train end-point. This shouldn't be changed.
- `team=Automatoes`: We will use the Automatoes dataset to train our model. This can be changed to the other team names that are in the `data` folder of the repo. It is case-sensitive so be aware. 

If all went well, you will see in your terminal some output which at the end should have something similar to this:
```
2023/01/22 16:46:53 INFO mlflow.projects: === Run (ID '8c7de92524144e51afe27811bc16e0f7') succeeded ===
```
Between the brackets you will see the `experiment_id` (here '8c7de92524144e51afe27811bc16e0f7') which we will need to see the artifacts of the experiment and run new test data on our model (see 4.4).

This will also have created the earlier-mentioned `mlruns` folder. In this folder you will find all the information about the different experiments that have been run. 

To go to the data on our example experiment, do the following:

```
# Go to the directory of the experiment
cd mlruns/0/8c7de92524144e51afe27811bc16e0f7/artifacts
```

In this directory you will find:
- A .txt file containing the metrics (MAE, MSE, RMSE, MAPE, R2) of the train data. 
- A .txt file containing the metrics (MAE, MSE, RMSE, MAPE, R2) of the test data.
- A plot of the last month 2020/05/01 - 2020/05/30 with both the train data, test data and the predicted data. 
- A directory `model` containing, among other files, the pickle file of the model.

### 4.4 Test new data
To test new data on our model we can use the `runner.ipynb` notebook in the `notebooks` folder of the repo. This will load the pickle file of the model that is stored in `../mlruns/0/<experiment_id>/artifacts/model`.

Add the new test data (both Weather and GreenhouseClimate) to the `notebooks/input` folder. There are already two dummy data csv's (`dummy_greenhouse.csv` and `dummy_weather.csv`) which can also be used as an example. Since our model needs to predict up to 24 hours in advance (`t<=24`), with 1 hour timesteps, we expect that the new data sets will not have more than 24 rows and the values in the `time` column should be integers between 0 and 24.

Before we can run the notebook, the user should fill in the following at section `2. User input` of the runner notebook:
- `experiment_id`: The id of the MLflow experiment
- `file_name_greenhouse`: The name of the file with the greenhouse climate data (.csv should be included)
- `file_name_weather`: The name of the file with the weather data (.csv should be included)

If this is done, the whole notebook can be run and you can see your results at the end!


## 5. Next steps
In this section I want to give some next steps that could improve the model and/or repo:
- `Include other data sets`: For the model we only use the weather and greenhouse climate data sets. Perhaps there is also valuable information in the other data sets
- `Automatic feature selection`: Currently the features that are selected are based on the "Automatoes" team, but those are not necessarily as relevant for the other teams/datasets. To automize this would be a good step. 
- `Add logging`: Right now, the repo does not contain any logging which makes debugging quite hard. This should be added therefore. 

