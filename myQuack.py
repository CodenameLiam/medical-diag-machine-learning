
'''

2020

Scaffolding code for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''

# Import numpy for array handling
import numpy as np

# Import matplotlib for plotting performance
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Import time for measuring performance
import time

# Import sklearn classifiers
from sklearn import tree, neighbors, svm, neural_network

# Import sklearn model and metric helpers
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(9959807, 'Liam', 'Percy')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
        - the first field is a ID number
        - the second field is a class label 'B' or 'M'
        - the remaining fields are real-valued

    Return two numpy arrays X and y where
        - X is two dimensional. X[i,:] is the ith example
        - y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
        X,y
    '''

    # Load diagnosis data. If the tumor is malignant, set the class label to 1, otherwise set it to 0
    y = (np.genfromtxt(dataset_path, delimiter=',', dtype=str, usecols=(1)) == 'M').astype('uint8')

    # Load features for each cell nucleus
    X = np.genfromtxt(dataset_path, delimiter=',', dtype='float16', usecols=range(2, 32))

    return X, y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Hyperparamter for decision tree classifier
MAX_DEPTH_START = 1
MAX_DEPTH_STOP = 200
MAX_DEPTH_NUM = 200
# Optimal Hyperparamter for decision tree classifier = 32

def build_DecisionTree_classifier(X_training, y_training):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    '''
    # Create the decision tree classifier using the sklearn library
    decisionTree_classifier = tree.DecisionTreeClassifier()
    # Define params using max depth hyperparamater
    params = {'max_depth': np.linspace(MAX_DEPTH_START, MAX_DEPTH_STOP, MAX_DEPTH_NUM, dtype=int)}
    # Define scoring metric to quantifying the quality of predictions
    scoring = {'Accuracy': 'balanced_accuracy'}

    # Estimate the best value of the hyperparameter using cross validated grid search
    clf = GridSearchCV(decisionTree_classifier, params, scoring = scoring, refit = 'Accuracy', return_train_score = True)
    # Train the model using the training data
    clf.fit(X_training, y_training)

    return clf 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Hyperparamter for nearest neighbours classifier
NUM_NEIGHBORS_START = 1
NUM_NEIGHBORS_STOP = 200
NUM_NEIGHBORS_NUM = 200
# Optimal Hyperparamter for nearest neighbours classifier = 12

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    '''
    # Create the nearest neighbours classifier using the sklearn library
    nearestNeighbours_classifier = neighbors.KNeighborsClassifier()
    # Define params using the number of neighbours hyperparamater
    params = {'n_neighbors': np.linspace(NUM_NEIGHBORS_START, NUM_NEIGHBORS_STOP, NUM_NEIGHBORS_NUM, dtype=int)}
    # Define scoring metric to quantifying the quality of predictions
    scoring = {'Accuracy': 'balanced_accuracy'}

    # Estimate the best value of the hyperparameter using cross validated grid search
    clf = GridSearchCV(nearestNeighbours_classifier, params, scoring = scoring, refit = 'Accuracy', return_train_score = True)
    # Train the model using the training data
    clf.fit(X_training, y_training)

    return clf 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Hyperparamter for support vector machine classifier
C_START = 0
C_STOP = 5
C_NUM = 200
# Optimal Hyperparamter for support vector machine classifier = 5231

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    '''
    # Create the support vector machine classifier using the sklearn library
    supportVectorMachine_classifier = svm.SVC()
    # Define params using the number of neighbours hyperparamater
    params = {'C': np.logspace(C_START, C_STOP, C_NUM)}
    # Define scoring metric to quantifying the quality of predictions
    scoring = {'Accuracy': 'balanced_accuracy'}

    # Estimate the best value of the hyperparameter using cross validated grid search
    clf = GridSearchCV(supportVectorMachine_classifier, params, scoring = scoring, refit = 'Accuracy', return_train_score = True)
    # Train the model using the training data
    clf.fit(X_training, y_training)

    return clf 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Hyperparamter for neural network classifier
# NEURONS_START = 100
# NERONS_END = 200
# NEURONS_NUM = 100
# Optimal Hyperparamter for neural network classifier = 146
OPTIMAL_NEURONS

def build_NeuralNetwork_classifier(X_training, y_training):
    '''
    Build a Neural Network classifier (with two dense hidden layers)
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    '''
    # Create the nerual network classifier using the sklearn library
    neuralNetwork_classifier = neural_network.MLPClassifier(max_iter=1000)
    # Define params using the number of neighbours hyperparamater
    # params = {'hidden_layer_sizes': np.linspace(NEURONS_START, NERONS_END, NEURONS_NUM, dtype=int)

    # This line should be run once the optimal neuons have been found to prevent lengthy run-times
    params = {'hidden_layer_sizes': (OPTIMAL_NEURONS,OPTIMAL_NEURONS)}
    # Define scoring metric to quantifying the quality of predictions
    scoring = {'Accuracy': 'balanced_accuracy'}

    # Estimate the best value of the hyperparameter using cross validated grid search
    clf = GridSearchCV(neuralNetwork_classifier, params, scoring = scoring, refit = 'Accuracy', return_train_score = True)
    # Train the model using the training data
    clf.fit(X_training, y_training)

    return clf 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def plot_hyperparamter_accuracy(clf, title, xLabel):
    '''
    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    Plots the score metric for a given classifier for each hyperparamter selection
    Can accept multiple score metrics

    @param
        clf: The classifier to plot
        title: The title of the graph
        xLabel: The hyperparameter label
    '''
    # Get the cross validated results from the classifier
    results = clf.cv_results_

    # Define the initial state of the plot
    plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=16)
    plt.xlabel(xLabel)
    plt.ylabel("Accuracy (%)")

    # Set axis limits
    ax = plt.gca()
    # ax.set_xlim(0, 200)
    # ax.set_xlim(0, 5e4)
    # ax.set_xlim(100, 200)
    ax.set_ylim(0.7, 1.05)

    # Define the X axis ticks and paramter to analyse
    param = 'param_' + list(clf.best_params_.keys())[0]
    X_axis = np.array(results[param].data, dtype=float)
    
    # For each score metric (and colour if multiple metrics are being used)
    for scorer, color in zip(sorted(clf.scoring), ['#E95678']):
        # Plot the training and test data for each hyperparameter
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.3 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        # Find the best score metric value
        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]
        best_X = X_axis[best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("({:.0f}, {:0.2f})".format(best_X, best_score),
                    (X_axis[best_index], best_score + 0.005))

    # Plot the graph
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def plot_accuracy_matrix(clf, name, labels, X_test, y_test):
    '''
    Plots a confusion matrix for a given classifier

    @params
        clf: The classifier to plot the confusion matrix for
        name: The name of the classifier
        labels: The class labels
        X_test: X testing data
        y_test: y testing data
    '''
    # Define the title and colours for the plot
    title = "Performance of " + name + " Classifier"
    colors = [(1, 1, 1), (0.914, 0.337, 0.47)] 
    # colors = [(1, 1, 1), (0.98, 0.717, 0.585)] yellow

    # Create a colour map for the probabilities
    cm = LinearSegmentedColormap.from_list("Performance", colors, N=100)

    # Plot the confusion matrix
    plot_confusion_matrix(classifier, X_test, y_test, cmap=cm, display_labels=labels, normalize='true')
    plt.title(title)
    plt.show()


def determine_testSplit(classifiers, X, y):

    # Define dictonary to store the avergae accuracy for each classifier
    # given a particular test split
    average_test_split = {}

    # For each test split...
    for split in np.around(np.arange(0.2, 0.45, 0.05), decimals=2):
        # Print the split ratio
        print("\n-------Split: {}-------".format(split))
        # Get training and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split)
        # Define a variable to store the total accuracy calcualted for the classifiers
        accuracy_total = 0

        # Find the average accuracy for all classifiers       
        for function, name, param in classifiers:
            # Train the classifier
            classifier = function(X_train, y_train)
            # Make predictions 
            prediction_test = classifier.predict(X_test)  

            # Determine accuracys          
            print(name, " Accuracy: {:0.2f}".format(accuracy_score(y_test, prediction_test)))      
            # Add it to the accuracy total   
            accuracy_total += accuracy_score(y_test, prediction_test)

        # Find the average
        accuracy_avg = accuracy_total/4
        # Add it to the dictionary
        average_test_split[split] = accuracy_avg

    # Find the optimal test split
    optimal_test_split = max(average_test_split, key=average_test_split.get)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Test data split value
# Controls the ratio between test and training data
TEST_SPLIT = 0.25

if __name__ == "__main__":

    # Write a main part that calls the different
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    # Specify seed to allow for repeatability
    np.random.seed(1)
    
    # Print the team, i.e. me :)
    print(my_team())

    # Start clock to allow for overall testing time to be computed
    start_time = time.perf_counter()
    
    # Prepare the dataset
    X, y = prepare_dataset('medical_records.data')

    # Define an array to store classifier functions, their names and their hyperparamter
    classifiers = [[build_DecisionTree_classifier, "Decision Tree", "Max Depth"],
                   [build_NearrestNeighbours_classifier, "Nearest Neighbour", "Number of Neighbours"],
                   [build_SupportVectorMachine_classifier, "Support Vector Machine", "Parameter C"],
                   [build_NeuralNetwork_classifier, "Neural Network", "Hidden Layer Size"]]

    # Define labels for classifiers used in report generation
    labels = ("Malignant", "Benign")

    # Function to find the optimal test split value, not needed after first use
    # determine_testSplit(classifiers, X, y)

    # Create initial training and testing data set based on the optimal value for test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SPLIT)

    # Test each classifier and output performance
    for function, name, paramter in classifiers:
        # Print heading
        print("################ " + name + " Classifier ################")

        # Train the classifier and print time taken to train
        classifier = function(X_train, y_train)
        print("\nTook %0.2f seconds to train " % (time.perf_counter() - start_time) + name + " classifier")
        # Reset
        start_time = time.perf_counter()

        # Output best parameters for each classifier
        print(name, "Best Parameters:", classifier.best_params_, "\n")

        # Generate classification report for training data
        prediction_training = classifier.predict(X_train)
        print(name, "Training Data Classification Report:")
        print(classification_report(y_train, prediction_training, target_names=labels))

        # Generate classification report for test data
        prediction_test = classifier.predict(X_test)
        print(name, "Test Data Classification Report:")
        print(classification_report(y_test, prediction_test, target_names=labels))

        # Plot hyperparamter accuracy (off unless required)
        # plot_hyperparamter_accuracy(classifier, name + " Accuracy", paramter)
        # Plot confusion matrix (off unless required)
        # plot_accuracy_matrix(classifier, name, labels, X_test, y_test)
    
 