An interpretable subpopulation-based modeling approach for diabetic kidney disease prediction
====================================

This is the working directory for building and validating an interpretable subpopulation-based modeling approach for Diabetic Kidney Disease (DKD) prediction based on electronic health records (EHRs) from University of Kansas Medical Center (KUMC).

by Bo Liu, Xinhou Hu, with Yong Hu and Mei Liu
[Big Data Decision Institute, Jinan University][BDDI]
[Medical Informatics Division, University of Kansas Medical Center][MI]

[BDDI]: https://bddi.jnu.edu.cn/
[MI]: http://informatics.kumc.edu/

Copyright (c) 2021 Jinan University  
Share and Enjoy according to the terms of the MIT Open Source License.

***

## Background

Diabetic kidney disease (DKD) is a complex and heterogeneous disease with numerous etiologic pathways. The current electronic medical record (EMR) based clinical risk prediction models often ignore special patient groups and cannot guarantee performance stability. The project was carried out with the following aims:

* **Aim 1 - Development of modeling approach**: An interpretable subpopulation-based modeling approach was developed and internally cross-validated using electronic medical record (EMR) data from the University of Kansas Medical Center’s (KUMC) de-identified clinical data repository called [HERON] (Health Enterprise Repository for Ontological Narration). 
      * Task 1.1: data extraction and quality check       
      * Task 1.2: exploratory data analysis (e.g. strategies for data cleaning and representation, feature engineering)     
      * Task 1.3: using an optimized decision tree to split the data into different subpopulations
      * Task 1.4: developing proposed models for each subpopulation through lightgbm
 
* **Aim 2 – Evaluation of model performance**: We implemented an automated package to develop models for all patients. Prediction performance was validated in general patients and subpopulations with global, subgroup model accordingly.
      * Task 2.1: testing of models in all test samples
      * Task 2.2: comparing subpopulation models with global model in each 
      * Task 2.3: ranking and comparing the predictors based on their importance and SHAP value in improving model performance

[HERON]:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3243191/

***

## Data Preprocessing

For each hospital admissions (encounters) in the data set, we extracted all demographic information, vitals data, medications, past medical diagnoses, and admission diagnosis from EMR. For test laboratory, we extracted a selected list of laboratory variables that may represent potential presence of a comorbidity correlated with DKD. ACR and eGFR were not included as predictors because they were used to determine the occurrence of DKD.

***
## Environmental Requirements
In order to run predictive models and generate final report, the following infrastructure requirement must be satisfied:

|python|lightgbm|sklearn|numpy|pandas|shap|scipy|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|3.7|2.1.2|0.24.1|1.21.0|1.2.5|0.39.0|1.7.0|

- **[Python]**: version >=3.7 is required.
- **[lightgbm]**: The version is 2.1.2
- **[sklearn]**: A widely used package for machine learning in python. The version is 0.24.1
- **[numpy]**: A python package used for data processing. The version is 1.21
- **[pandas]**: A python package used for process and save data. The version is 1.2.5
- **[shap]**: A python package used for compute shape value. the version is 0.39.0
- **[scipy]**: a python package used for Scientific computing.

[python]: https://www.python.org
[lightgbm]: https://lightgbm.readthedocs.io/en/latest/index.html
[sklearn]: https://scikit-learn.org/stable/
[numpy]: https://numpy.org/
[pandas]: https://pandas.pydata.org/
[shap]:https://shap.readthedocs.io/en/latest/index.html
[scipy]: https://scipy.org
***

## Model Validation
The following instructions are for generating final report from our study cohort.

### Part I: Data preparation
 1. Please make sure class label of patients (DKD or not DKD) are placed in the last column of the data sheet used for model training and testing.
 2. Edit the file path of input and output in our python code.

### Part II: Development of modeling approach 
1. Partitioning the subpopulations 
-  **Aim**: identify the subpopulations in general patients by using recursive decision tree method  
-  **experiment**: run the main "Experiment1.py" of building tree. The parameters of split are specified in this code(the core params of build). After this experiment, the tree model will be saved to the specified location.
    > eg: `python Experiment1.py 2017-train-data.pkl`
- **code**:The core code used for building tree model (The leaf node of the tree is the subgroup we need) is `subgroup_identify.py`, the core parameters of this class `SubgroupIdentifyTree` are as follows:
    - `tree_params`: A dictionary that holds the parameters for building a grouping decision tree
        - `depth`: the max depth of tree.
        - `min_split_size`: the min size of inner node. (only a node's sample size greater than it, then can continue to split)
        - `min_leaf_size`: the min size of leaf node. (if the sample size of subnode is less than it, break this split)
    - `root_model_param`: the parameters of build gbm model for base model. Actually, you can set this param to None, but then you should edit the code(location:SubgroupIdentifyTree.fit()), let the program automatically search for parameters.
    - `metric`: a string, value `ghhi`、`auc`, specifies the optimization objectives used in the decision tree growth process. You can set it to None, and must set a custom `metric_func`
    - `metric_func`: a function, this function will use to select candidate when tree is growing. you can set `metric` or `metric_func`, if both are set, `metric` takes precedence 
    - `candidates`: list, the candidates feature set, used to tree growth.

2. Developing models and comparing performance
-  **Aim**: develop predictive models for identified subpopulations and compare the performance with global model 
-  **experiment**: run "Experiment2.py" to search the best lightgbm parameter and build model for each subpopulation (leaf node of decision tree). Actually, Experiment 1 and Experiment 2 can be combined into one, but programming environment limits the running time of each submitted task, so it was divided into two experiments.
    > eg: `python Experiment2.py tree-model.pkl 2017-test-data.pkl`
    > - `tree-model.pkl`: The tree model saved in Experiment 1. After running experiment 1, the model is saved in the '/model' directory, so you only need to pass in the name.
    > - `2017-test-data.pkl`: test set name. By default, the program searches for test set data in the '/data' directory


[Model_development]: https://github.com/BDAII/SubgroupModeling

***
*updated 01/04/2022*
