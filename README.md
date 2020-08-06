# tweet-sentiment

## Installation

To install with conda, clone this repo, cd into the main directory and do:

`conda env create -f environment.yml`

## Data

All databases and files used in the project can be found at:

https://drive.google.com/drive/folders/136tyBrJZew0tfJllaSIOc_YKanKt_A8a?usp=sharing

Please place all files inside the **/data** directory.

The main raw database is **tweets.csv**

## Notebooks

The main directory contains the following notebooks:

- **Full.ipynb** contains a reduced version of all the project. It requires the 
**tweets_reduced.csv** database
- **EDA.ipynb** contains exploratory analysis used for some charts in the presentation
- **Bots.ipynb** contains code to fine-tune a language generation model with GPT-2

To run the project in full, the notebooks should be run in the following order:

- **Topic Modeling.ipynb** Estimates Topic Models
- **Sentiment.ipynb** Estimates the sentiment model
- **Market.ipynb** Fits the market classifier to the data

