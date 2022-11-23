# Day After Tomorrow
## Team Andrew

A Repository for the ACDS mini project.

### Introduction

This project is a mock of a challenge released by FEMA(Federal Emergency Management Agency) in the US. The aim of the challenge was to improve their emergency protocals under hurricane threats, by providing a solution based on Machine Learning of forecasting the evolution of tropical cyclones in real-time. The dataset is from NASA satellite images of tropical storms around the Atlantic and East Pacific Oceans. A ML based Solution is provided in this project to generate future image predictions for a given strom.   
  
Fully connect LSTM, LSTM with PCA and ConvLSTM are used in the project to provide solution. The way of installation and how to used them to predict the storm are introduced below. 

### Installation guide
Firstly, clone the repository:
```
git@github.com:ese-msc-2021/acds-day-after-tomorrow-andrew.git
```
and `cd` into acds-day-after-tomorrow-andrew

Now create and activate the conda environment
```
conda env create -f environment.yml
conda activate Andrew
```

Install required packages
```
pip install -r requirements.txt
```

Activate the setup.py in order to create tools module with
```
pip install -e .
```

We can run pytests to see if everything is going alright
with the following command although there is an automated github workflow
```
pytest tools/tests/
```
### User instructions
#### For Notebooks:
The main notebooks is <b>FC_LSTM_PCA_Master.ipynb</b>. As the main submission, the model in it is fully connected LSTM with PCA. Please open these notebooks and run the notebooks from top to bottom.

The <b>secondary_notebook.ipynb</b> is the secondary submission of the project. It consists of data exploration, fully connect LSTM without PCA, and a ConvLSTM model. There are two ways to use this ConvLSTM model in this notebook. The first way is for each time of prediction, a dataset for a single storm is needed to train the model, then it can work on any storm dataset. The second ways is to train the model by 100 different storms to obtain a pretrained model, it can be used to predict any stroms without retraining.

#### For Folders:

Inside the <b>'.github/workflows'</b> folder there are four .yml files which consists of code that will automate PEP8 testing (both for .py and .ipynb notebooks), pytests and sphinx documentation everytime there is a a push to the repository.

Inside the <b>'docs'</b> folder there is a document for how to use the Pytest cases inside the 'tools/tests' folder or open the html/index.html on local computer. 

Inside the <b>'tools'</b> folder there are the functions that were imported and used throughout our notebooks. They are under the names 'dataprocessing.py', 'fc_lstm.py', 'fc_lstm_pca.py', 'metric.py' and 'visualisation.py'. Also, inside the 'tools' folder there is also a folder named 'tests' which enables doing pytests on the functions within the 'tools' folder.

### How to run the notebooks

Before you get started please load in all the .tar.gz source and train folder into the current repository. Also to evaluate the actual SSIM and MSE metric at the end please upload 5 real images labelled as "Real1.jpg", "Real2.jpg", "Real3.jpg", Real4.jpg", "Real5.jpg" in current directory.  

We import all the required packages including the modules we have created in a folder called tools. (Note: importing in local machine and in colab is different because in google colab we cannot install our created module, but you can do this in a local environment. If you are running the notebook on colab, please mannually load all python scripts from 'tools' folder first. )

### Important Note
We mainly used Google Colab to write the notebook rather than in local machine due to memory issues.  

### Documentation
You can find the documentation for how to use the functions inside the 'tools' folder inside docs/andrew.pdf or open the html/index.html on your local computer. 
