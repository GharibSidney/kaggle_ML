from sklearn.model_selection import train_test_split
from data_processing import data_preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
#from imblearn.under_sampling import NearMiss

def data_splits(X, y):
    """
    Split the 'features' and 'labels' data into training and testing sets.
    Input(s): X: features (pd.DataFrame), y: labels (pd.DataFrame)
    Output(s): X_train, X_test, y_train, y_test
    """
    # Split the data into training and testing sets
    # I added the stratify=y to make sure that the distribution of the labels in the train and test sets are the same
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=0, stratify=y)

    return X_train, X_test, y_train, y_test

#def data_splits_equal_classes(X, y):

#    nm = NearMiss(version = 1 , n_neighbors = 10)

#    x_sm, y_sm= nm.fit_resample(X, y)

#    print(y_sm.shape , x_sm.shape)
#    X_train, X_test, y_train, y_test = train_test_split(x_sm,y_sm, test_size=0.2 , random_state=42)

#    return X_train, X_test, y_train, y_test


def train_model(model_name, X_train_scaled, y_train):
    '''
    inputs:
       - model_name: the name of learning algorithm to be trained
       - X_train: features training set
       - y_train: label training set
    output: cls: the trained model
    '''
    if model_name == 'Decision Tree':
        #   call classifier here
        cls = DecisionTreeClassifier() 
        
    elif model_name == 'Random Forest':
        #   call classifier here
        cls = RandomForestClassifier() 
    elif model_name == 'SVM':
        #   call classifier here
        cls = SVC()

    #   train the model
    cls.fit(X_train_scaled, y_train)

    return cls


def eval_model(trained_models, X_train, X_test, y_train, y_test):
    '''
    inputs:
       - trained_models: a dictionary of the trained models,
       - X_train: features training set
       - X_test: features test set
       - y_train: label training set
       - y_test: label test set
    outputs:
        - y_train_pred_dict: a dictionary of label predicted for train set of each model
        - y_test_pred_dict: a dictionary of label predicted for test set of each model
        - a dict of accuracy and f1_score of train and test sets for each model
    '''
    evaluation_results = {}
    y_train_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}
    y_test_pred_dict = {
        'Decision Tree': None,
        'Random Forest': None,
        'SVM': None}

    # Loop through each trained model
    for model_name, model in trained_models.items():
        # Predictions for training and testing sets
        y_train_pred = model.predict(X_train) #   predict y
        y_test_pred = model.predict(X_test) #   predict y

        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred) #   find accuracy


        test_accuracy = accuracy_score(y_test , y_test_pred) #  find accuracy

        # Calculate F1-score
        train_f1 = f1_score(y_train, y_train_pred) #   find f1_score
        test_f1 = f1_score(y_test, y_test_pred) #   find f1_score

        # Store predictions
        y_train_pred_dict[model_name] = y_train_pred #  
        y_test_pred_dict[model_name] = y_test_pred #  

        # Store the evaluation metrics
        evaluation_results[model_name] = {
            'Train Accuracy': train_accuracy, #   ,
            'Test Accuracy': test_accuracy,  #   ,
            'Train F1 Score': train_f1,  #   ,
            'Test F1 Score': test_f1, #  
        }

    # Return the evaluation results
    return y_train_pred_dict, y_test_pred_dict, evaluation_results


def report_model(y_train, y_test, y_train_pred_dict, y_test_pred_dict, trained_models):
    '''
    inputs:
        - y_train: label training set
        - y_test: label test set
        - y_train_pred_dict: a dictionary of label predicted for train set of each model, len(y_train_pred_dict.keys)=3
        - y_test_pred_dict: a dictionary of label predicted for test set of each model, len(y_train_pred_dict.keys)=3
    '''

    # Loop through each trained model
    for model_name, model in trained_models.items():
        print(f"\nModel: {model_name}")

        # Predictions for training and testing sets
        y_train_pred = y_train_pred_dict[model_name] #   compelete it
        y_test_pred = y_test_pred_dict[model_name] #   compelete it

        # Print classification report for training set
        print("\nTraining Set Classification Report:")
        #   write Classification Report train
        print(classification_report(y_train, y_train_pred))

        # Print confusion matrix for training set
        print("Training Set Confusion Matrix:")
        #   write Confusion Matrix train
        print(confusion_matrix(y_train, y_train_pred))

        # Print classification report for testing set
        print("\nTesting Set Classification Report:")
        #   write Classification Report test
        print(classification_report(y_test, y_test_pred))

        # Print confusion matrix for testing set
        print("Testing Set Confusion Matrix:")
        #   write Confusion Matrix test
        print(confusion_matrix(y_test, y_test_pred))





















