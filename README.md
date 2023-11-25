# Coding Homeworks for the Berkeley CS189 F23 Class

1. Fully_Connected_Neural_Network: Added in missing forward and backward code to finish the fully connected linear network. Used the network to classify the Iris dataset. Added a second "Pima Indians Diabetes" dataset.

2. MLP_FashionMNIST_CNN_CIFAR10.ipynb: Wrote two Neural Nets in Pytorch. The first is a 4-level fully-connected network for the FashionMNIST dataset. Final accuracy is 81%. The second is a network with 6 CNN layers followed by 3 fully-connected linear layers. Final accuracy on the CIFAR-10 dataset is 82%.

3. Decision_tree: Wrote 4 variants of Decision Trees and test on 2 datasets: Titanic and Spam. 
    * a. Added in missing fit, predict, entropy and information gain code to finish a simple Decision Tree. No dependency beyond Numpy.
    * b. Used SKLearn's DecisionTreeClassifier to train a simple Decision Tree.
    * c. Used SKLearn's DecisionTreeClassifier to implement bagged trees that resample the training dataset. 
    * d. Used SKLearn's DecisionTreeClassifier to implement Random Forest, that both resample training dataset and randomly remove features. 

    Accuracy results on 5 runs:

    Titanic:

   |    |Simple Tree| SKLearn Tree| Bagged Trees| Random Forest
   |---|----|---|---|---|
   |Run|||                                                    
    |1|          79.0|         79.0|         79.0|          79.5
    |2|          81.0|         79.0|         81.5|          82.5
    |3|          77.0|         76.0|         75.0|          73.5
    |4|          84.0|         79.5|         83.5|          83.0
    |5|          81.5|         80.5|         81.5|          81.0
    |Avg|        80.5|         78.8|         80.1|          79.9
    
    Spam:

    |    |Simple Tree| SKLearn Tree| Bagged Trees| Random Forest
    |---|----|---|---|---|
    |Run|||                                                    
    |1|     79.8|    80.8|    81.1|     79.8
    |2|     80.4|    81.2|    81.0|     79.4
    |3|     80.1|    81.2|    81.4|     81.3
    |4|     81.7|    82.3|    82.4|     79.8
    |5|     78.7|    79.9|    79.7|     80.4
    |Avg|   80.2|    81.1|    81.1|     80.1

    
