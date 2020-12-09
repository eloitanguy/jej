# JEJ

Machine Learning project for retweet estimation: Ecole polytechnique INF554 Data Challenge 2020

  * [Presentation](#presentation)
  * [Repository overview](#repository-overview)
  * [Getting started](#getting-started)
  * [Reproducing results](#reproducing-results)
  * [Training PyTorch models](#training-pytorch-models)
  * [Training XGBoost models](#training-xgboost-models)
  * [Outputting Kaggle-format predictions](#outputting-kaggle-format-predictions)

## Presentation

Given a tweet text and numeric data associated to the user, our objective is to estimate the number of retweets the tweet gets. 

The data is available [on Kaggle](https://www.kaggle.com/c/covid19-retweet-prediction-challenge-2020/data?select=data), downloading it into data/ is required for running this repository's code.

The repository is the outcome of many approaches to this problem, the final proposition being composed of two models:

* An auto-encoding recurrent neural network that uses the text data and numeric data in order to estimate the natural logarithm of the retweets
* A boosted random forest model ([XGBoost](https://arxiv.org/pdf/1603.02754.pdf)) that uses the RNN's prediction and the same numeric data in order to estimate the log-RT.

## Repository overview

* *models.py* defines the PyTorch models.
* *dataset.py* defines the torch dataset objects for training.
* *train.py* is used for training the PyTorch models.
* *config.py* defines all the hyperparameters of this project.
* *xgboost_evaluation.py* is for preparing the data for XGBoost and for training it.
* *export_evaluation.py* is for exporting a prediction text file for a Kaggle test.

The data folder stores all the data for training the models:

* You should put `evaluation.csv` and `train.csv` there (without renaming them).
* The vocabulary and XGBoost data will be generated in that folder.

Two other folders will be automatically created: checkpoints will store all the model checkpoints during and after training, and tb will store the tensorboard file if you wish to follow along the training execution.

## Getting started

Before executing the scripts, please go through the following steps:

* Place the `evaluation.csv` and `train.csv` files [from Kaggle](https://www.kaggle.com/c/covid19-retweet-prediction-challenge-2020/data?select=data) in data/
* If necessary, install the required modules using `pip install -r requirements.txt`

## Reproducing results

The configuration file has been prepared with the configuration of our best model. Please note that even with the deterministic parameter, the training will not be entirely reproductible. Furthermore, if your GPU has less than 8GB of memory, you should consider lowering the batch size in `config.py` at TRAIN_CONFIG.

After the [Getting started](#getting-started) section, the commands to execute are:

* Training the Auto-Encoder model, this will save the Auto-Encoder RNN model in `checkpoints/AE_reproduction/best.pth`.

    `python train.py --deterministic`
    
* Preparing the input data for training and testing the XGBoost model (three files for train, val and test saved in `data/`), then training the model (saving it in `checkpoints/XGB_reproduction/checkpoint.model`):

    `python xgboost_estimator.py --prepare`
    `python xgboost_estimator.py --train`

* Exporting the XGB model's estimation for the Kaggle test set in `checkpoints/XGB_reproduction/predictions.txt`:

    `python export_evaluation.py --xgb-name XGB_reproduction`

The complete reproduction takes 18 minutes on this machine, but may take significantly longer with less VRAM or less RAM (see the section below for comparing your setup to ours). In total the folder will take 870MB with the checkpoints saved.

The scores for this reproduction are expected to be:

* AE best validation MAE: 144.87 (against 144.14 for our original non-deterministic training).
* XGB validation MAE: 139.56 (against 139.4)
* Kaggle score: 149.60 (against 149.42)

Remark: our best Kaggle score uses the average output between this model and another (worse) XGB model with the same parameters that uses the output of an older RNN (with the exact same parameters but with a numeric data input size of 5 instead). This strategy improves the Kaggle score by 0.3, and is kept as a final answer for risk management.

## Training PyTorch models

Before starting training, configure your network by modifying RNN_CONFIG in `config.py` and adjust your training parameters by changing TRAIN_CONFIG.
Remark: if you change the Auto-Encoder vocabulary settings, you must run the command `python dataset.py` in order to prepare the vocabulary file.

When you are satisfied, make sure that you are using the right inference functions at the bottom of `train.py`, then you can simply run:

    python train.py

If you want your training to be deterministic, you can add the `--deterministic` option.

The models will be saved in checkpoints/[experiment_name]/ and you can follow the training live by executing:

    tensorboard --logdir tb

The best model in terms of validation loss will be kept in the dedicated checkpoint folder as well.

Our setup (for comparison):

* i7-8750H (6 cores, @4.1GHz)
* RTX 2080-MQ (8GB VRAM, used all of it here)
* 16GB RAM (used under 8GB here)

## Training XGBoost models

In the same manner as for PyTorch models, make sure to edit the config file to your liking before executing anything.
In particular, XGBoost models use the results (embeddings, hidden layers or output) of a PyTorch network, so make sure you have trained one first and specified its checkpoint at XGBOOST_CONFIG['embedder'].

Once you are ready, prepare the data files that XGBoost will use for training, validation and testing by using:

    python xgboost_estimator.py --prepare

You can then execute training with:

    python xgboost_estimator.py --train

If you change the XGBoost data, remember to execute the data preparation script before training!

## Outputting Kaggle-format predictions

If you want to prepare a prediction file from a PyTorch model, run:

    python export_evaluation.py --checkpoint [your_checkpoint_path.pth]

For outputting the predictions of an XGBoost model, use the name of its experiment (XGBOOST_CONFIG['experiment_name']):

    python export_evaluation.py --xgb-name [your_experiment_name]
