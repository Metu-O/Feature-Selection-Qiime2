# Feature-Selection-Qiime2

This repository describes the methods used to test different sci-kit learn feature selection methods as part of Qiime2 q2-classifier.

# Embedded Feature Selection

# 1. SelectFromModel Using Multinomial Naive Bayes Estimator

The sklearn multinomial naive bayes (MultinomialNB) classifier was used as a base estimator in the feature selection step of my classification pipeline. RFC has a feature_log_prob_ attribute after fitting, which was used as the importance_getter parameter. For more details about sklearn's MultinomialNB, click on this [link](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

# 2. SelectFromModel Using Random Forest Estimator

The sklearn random forest classifier (RFC) was used as a base estimator in the feature selection step of my classification pipeline. RFC has a feature_importances_ attribute after fitting, which was used as the importance_getter parameter. For more details on sklearn's SelectFromModel, click on this [link](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html). For more details about sklearn's RFC, click on this other [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

# 3. SelectFromModel Using Stochastic Gradient Descent Estimator

The sklearn stochastic gradient descent (SGD) classifier was used as a base estimator in the feature selection step of my classification pipeline. RFC has a coef_ attribute after fitting, which was used as the importance_getter parameter. For more details about sklearn's SGDClassifier, click on this [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).  

# Initializing 

1. Clone Feature-Selection-Qiime2

  ```
  git clone https://github.com/Metu-O/Feature-Selection-Qiime2
  ```
  
2. Change directory to the Feature-Selection-Qiime2 you just cloned. Change to your own directory e.g. /home/mosele/Feature-Selection-Qiime2

  ```
  cd Feature-Selection-Qiime2
  ```

3. Activate Qiime2 environment
   
   (see this [link](https://docs.qiime2.org/2022.8/install/) on how to install the QIIME 2 Core 2022.8 distribution)

4. Run codes 

   using command lines or bash script (BASH SCRIPT MUST BE EDITED TO FIT USER'S PATH) 

# Command lines

1. Naive_Bayes_Parameters.py contains code that runs the naive bayes classifier with no feature selection using qiime2 q2-classifier recommended parameters. Naive_Bayes_Parameters.py allows user to run codes with defaults or user input. I strongly advise using defaults except you are adept with directories. Run "help" to see usage and defaults .
```
python Naive_Bayes_Parameters.py -h
```
Example: 





Run script below to use defaults

```
python Naive_Bayes_Parameters.py 
```
Run script below with to change defaults with user input 
```
python Naive_Bayes_Parameters.py \
   -n 'reference database name'\
   -s 'path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files)'\
   -t 'path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files)'
```

Here is one example 
```
python Naive_Bayes_Parameters.py \
   -n 'greegenes'\
   -s '/home/mosele/99_gg_seq.qza'\
   -t '/home/mosele/99_otu_taxonomy_clean-Copy1.tsv.qza'
```

2. SelectFromModel_MultinomialNB.py contains code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the MultinomialNB estimator. Run "help" to see usage.
```
python SelectFromModel_MultinomialNB.py -h
```
Run script below to use defaults
```
python SelectFromModel_MultinomialNB.py 
```
Run script below with to change defaults with user input
```
python SelectFromModel_MultinomialNB.py \
   -n 'reference database name'\
   -s 'path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files)'\
   -t 'path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files)' 
```

3. SelectFromModel_RandomForest.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the RandomForestClassifier estimator. Run "help" to see usage.
```
python SelectFromModel_RandomForest.py -h
```
Run script below to use defaults
```
python SelectFromModel_RandomForest.py
```
Run script below with to change defaults with user input
```
python SelectFromModel_RandomForest.py \
   -n 'reference database name e.g greengenes or SILVA'\
   -s 'path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files)'\
   -t 'path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files)'
```

4. SelectFromModel_SDG.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the stochastic gradient descent (SDG) estimator. Run "help" to see usage.
```
python SelectFromModel_SDG.py -h
```
Run script below to use defaults
```
python SelectFromModel_SDG.py
```
Run script below with to change defaults with user input
```
python SelectFromModel_SDG.py \
   -n 'reference database name'\
   -s 'path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files)'\
   -t 'path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files)' 
```

Compare method accuracy by running Evaluate_method_accuracy.py

NOTE: Evaluate_method_accuracy.ipynb is the notebook version of Evaluate_method_accuracy.py, showing all plots and figures. 

Run "help" to see usage.
```
python Evaluate_Method_Accuracy.py -h
```
Run script below to use defaults
```
python Evaluate_Method_Accuracy.py
```
Run script below with to change defaults with user input
```
python Evaluate_Method_Accuracy.py \
   -p 'save plots in this directory 
```

Note: running these codes takes hours and may require a high computing processor. Do not wait around.

# Bash Script runs all codes 

Note: The bash script was personalized to my personal environments and filepaths. Edit before use.

# Conclusion

Feature-Selection-Qiime2 performed better than regular qiime2 more than 70% of the time. To find out more about the metrics used for comparison, read my thesis (will be cited once published). 

# Citation

1. Bokulich, N.A., Kaehler, B.D., Rideout, J.R. et al. Optimizing taxonomic classification of marker-gene amplicon sequences with QIIME 2’s q2-feature-classifier plugin. Microbiome 6, 90 (2018). https://doi.org/10.1186/s40168-018-0470-z

2. Osele, M. Machine Learning for Biological Data. ... (not yet available). 

3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
