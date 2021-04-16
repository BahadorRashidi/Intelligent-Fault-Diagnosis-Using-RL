# Intelligent Fault diagnosis using Reinforcement Learning

**Problem Statement:**
Perform condition classification using an intelligent agent that learns the classification similar to perception of a human

**Methodology:**
Extract latent features using stacked autoencoders and use deep Q-Network to train the agent

## Dataset
The fault data of rolling bearings used in our study are collected by Case Western Reserve University (CWRU). The dataset consists of
ball bearing test data for both normal and faulty bearings. The sampling frequency for the measured data is 48KHz that is comming from a measurement near the motor shaft in the experimental setup.

* There are four different conditions:
  * N: Normal
  * IF: Inner fault
  * OF: Outer fault
  * RF: Roller fault

* To validate the proposed method, the vibration data are divided into four sets (A, B, C, and D) according to the amount of loads induces to the shaft as an operating condition which will result in different vibration mode accroding to the dynamic of the shaft.

* Each set (e.g. A) contains 10 different classes, (1,2,...,10) according to the fault diameter and fault location.
* A, B, C contains all the classes and there are no classes that are unseen in these three datasets.
* In average, each class under a certain dataset contains 480,000 samples. In other words, under that class condition the lab equipment was run and average of 480,000 observation from vibration sensor was captured.
* Since the measurements are in time-domain, we need to convert them to frequecy domain to be able to extract frequency charactristics for any classification analysis. With this aim, we apply Fast fourieri transformation(FFT) on each 480,000 time-based samples. To achieve this we need to stride a window with a lenth W and steping size L. According to the literature, we consider L = W = 2400 samples. Accordingly, the result of FFT on each window with length of 2400 will be 2400. Hiowever due to the symetricity conditoin of FFT, we only consider the second half of the result which will be 1200 samples in frequency domain. After finishing this basic pre-processing analysis we will end up having an average of ~200 rows with 1200 columns (i.e. coulmns are the result of fft and will be treated as features). This matrix will be for each class under each dataset and can be used for the Benchmark analysis.
**Note:** To achieve this, a function is defined in the pre-processsing class

## Fault Detection with Q-learning
* To achieve, run QlearningWITHbayesian.py. This file contains following steps:
* 1、Call Crossprocessing.py, to make a preprocessing on dataset for ten-fold cross validation
* 2、Call SAE.py, pretrain a SAE network and save the best parameter into encoder_para.pth.
* 3、Run Q-learning agent with Bayesian search. If you want to modify the search space of parameters, please modify para.json.
* 4、Finally we can get a best parameter combination for Q-learning based on the average accuracy on validation set. 

