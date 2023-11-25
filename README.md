# Coding Homeworks for the Berkeley CS189 F23 Class

1. Fully_Connected_Neural_Network: Added in missing forward and backward code to finish the fully connected linear network. Used the network to classify the Iris dataset. Added a second "Pima Indians Diabetes" dataset.

2. MLP_FashionMNIST_CNN_CIFAR10.ipynb: Wrote two Neural Nets in Pytorch. The first is a 4-level fully-connected network for the FashionMNIST dataset. Final accuracy is 81%. The second is a network with 6 CNN layers followed by 3 fully-connected linear layers. Final accuracy on the CIFAR-10 dataset is 82%.

3. Decision_tree: Wrote 4 variants of Decision Trees and test on 2 datasets: Titanic and Spam. 
    a. Added in missing fit, predict, entropy and information gain code to finish a simple Decision Tree. No dependency beyond Numpy.
    b. Used SKLearn's DecisionTreeClassifier to train a simple Decision Tree.
    c. Used SKLearn's DecisionTreeClassifier to implement bagged trees that resample the training dataset. 
    d. Used SKLearn's DecisionTreeClassifier to implement Random Forest, that both resample training dataset and randomly remove features. 

    Accuracy results on 5 runs:

    Titanic:

        Simple Tree SKLearn Tree Bagged Trees Random Forest
    Run                                                    
    1          79.0         79.0         79.0          79.5
    2          81.0         79.0         81.5          82.5
    3          77.0         76.0         75.0          73.5
    4          84.0         79.5         83.5          83.0
    5          81.5         80.5         81.5          81.0
    Avg        80.5         78.8         80.1          79.9
    
    Spam:

        Simple Tree SKLearn Tree Bagged Trees Random Forest
    Run                                                    
    1     79.806763    80.772947    81.062802     79.806763
    2     80.386473     81.15942    80.966184      79.42029
    3     80.096618     81.15942    81.449275     81.256039
    4      81.73913    82.318841    82.415459     79.806763
    5     78.743961    79.903382    79.710145     80.386473
    Avg   80.154589    81.062802    81.120773     80.135266

    