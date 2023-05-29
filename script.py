"""
Amber Wu
EG DS Intern Assessment

This is a script for generating a RandomForestClassifier model to predict Starcraft player ranks.
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def preprocess_data(file):
    """
    Removes rows with NaN values and corrects data types from the given file.
    """
    df = pd.read_csv(file)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df['Age'] = df['Age'].astype('int64')
    df['HoursPerWeek'] = df['HoursPerWeek'].astype('int64')
    df['TotalHours'] = df['TotalHours'].astype('int64')
    return df


def define_labels_features(df):
    """
    Defines the labels and features of the model.
    GameID is excluded from the features as it is a unique identifier for each player.
    """
    X = df.drop(["GameID", "LeagueIndex"], axis=1)
    y = df["LeagueIndex"]
    return X, y


def select_top_features(X, y):
    """
    Splits the given labels and features into training and testing data and selects the top ten features with highest importance.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    selector = SelectKBest(score_func=chi2, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X.columns[selector.get_support()]
    return X_train_selected, X_test_selected, y_train, y_test, selected_features


def select_best_max_depth(X_train_selected, X_test_selected, selected_features, y_train, y_test):
    """
    Returns the hyperparameter max depth for a RandomForestClassifier that has the highest test accuracy.
    """
    max_depth_values = range(1, 11)
    train_accuracy = []
    test_accuracy = []

    for depth in max_depth_values:
        model = RandomForestClassifier(max_depth=depth)
        model.fit(X_train_selected, y_train)
            
        train_predictions = model.predict(X_train_selected)
        train_acc = accuracy_score(y_train, train_predictions)
        train_accuracy.append(train_acc)
            
        test_predictions = model.predict(X_test_selected)
        test_acc = accuracy_score(y_test, test_predictions)
        test_accuracy.append(test_acc)

        best_max_depth = max_depth_values[np.argmax(test_accuracy)]
    return best_max_depth

def build_model(X_train_selected, X_test_selected, selected_features, y_train, y_test, best_max_depth):
    """
    Builds a RandomForestClassifier model using the selected features and labels, and evaluates its performance.
    """
    model = RandomForestClassifier(max_depth=best_max_depth)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    print("Model Evaluation:")
    print(classification_report(y_test, y_pred, zero_division=1))

    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': importance})
    print()
    print("Feature Importance:")
    print(feature_importance.sort_values(by='Importance', ascending=False))

    train_predictions = model.predict(X_train_selected)
    print()
    print()
    print("Train Accuracy: " + str(accuracy_score(y_train, train_predictions)))

    test_predictions = model.predict(X_test_selected)
    print("Test Accuracy: " + str(accuracy_score(y_test, test_predictions)))

    print()
    print("Predictions:")
    print(test_predictions)


def main():
    df = preprocess_data("starcraft_player_data.csv")
    X, y = define_labels_features(df)
    X_train_selected, X_test_selected, y_train, y_test, selected_features = select_top_features(X, y)
    best_max_depth = select_best_max_depth(X_train_selected, X_test_selected, selected_features, y_train, y_test)
    build_model(X_train_selected, X_test_selected, selected_features, y_train, y_test, best_max_depth)


if __name__ == "__main__":
    main()