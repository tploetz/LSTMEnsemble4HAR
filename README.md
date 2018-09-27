# LSTMEnsemble4HAR
sourcecode for IMWUT 2017 paper "Ensembles of Deep LSTM Learners for Activity Recognition using Wearables" (Guan, Ploetz)


In the code file, before running the training process, please select a specific dataset(default is Opptunity79).

The default configuration is using GPU with Cuda9.0 and tensorflow-gpu1.9, if cannot run please update your Cuda and tensorflow-gpu version.

If only using CPU, please use line 247 in singleLSTM.py to replace line 243, line 227 in fusionensemble.py to replace line 226.

In the training, first, run the singleLSTM.py Then run the fusionensemble.py and the result will be saved to a text file for each dataset.

A csv files will be created when running fusionensemble.py to collect the fused f1 value for top 1,5,10 and 20 in each trial to help you draw a figure or calculate t-test.

Python version: python3

Clone the repository as usual.

Manually download the dataset files in the required format for re-running the experiments here:
https://goo.gl/wgEuhu 
Then put the whole dataset file(named "Ensemble-datasets") in the downloded repository file(named "LSTMEnsemble4HAR").

If you use the code, please cite:
Yu Guan and Thomas Plötz. 2017. Ensembles of Deep LSTM Learners for Activity Recognition using Wearables. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. (IMWUT) 1, 2, Article 11 (June 2017), 28 pages. DOI: https://doi.org/10.1145/3090076

Acknowledgement: Yu Guan's postgraduate student Xinchao Cheng managed to write all the tensorflow code, based on Yu's original Theano+lasagne code 

Any questions about the code, please email: 229779981@qq.com (or chengxinchao0116@gmail.com)
