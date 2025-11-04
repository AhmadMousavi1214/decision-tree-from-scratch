import pandas as pd
import numpy as np

class Node():
    def __init__(
        self,
        feature=None,
        children=None,
        feature_values=None,
        value=None
    ):
        self.feature = feature
        self.children = children
        self.feature_values = feature_values
        self.value = value

class DecisionTree():
    def __init__(
        self,
        root = None,
        mode="gain",
        max_Depth=float("inf"),
        min_Samples=2,
        pruning_threshold=None,
    ):
       self.root = root
       self.mode = mode
       self.max_Depth = max_Depth
       self.min_Samples = min_Samples
       self.pruning_threshold = pruning_threshold

    def _create_Tree(self, X, Y, depth=0):
        num_Samples = len(Y)
        
        # Check stopping conditions (Pre-Pruning)
        if num_Samples >= self.min_Samples and depth < self.max_Depth:
            best_Feature = self._get_best_Feature(X, Y)
            children = []
            # Check gain or gini!
            for (Xi, Yi) in best_Feature["feature_values"]:
                # TODO: Recursively create child nodes
                pass
            
        # TODO: Create leaf node with predicted value
        return Node()

    def _get_best_Feature(self, X, Y):
        best_feature_name = None
        
        if self.mode == "gain":
            max_info_gain = -np.inf
            
            for feature in X.columns:
                feature_info_gain = self._information_Gain(X[feature], X, Y)
                
                if feature_info_gain > max_info_gain :
                    best_feature_name = feature
                    max_info_gain = feature_info_gain
                
        elif self.mode == "gini":
           min_gini = np.inf
           
           for feature in X.columns:
               feature_gini = self._gini_Split(X[feature], X, Y)
               
               if feature_gini < min_gini:
                   best_feature_name = feature
                   min_gini = feature_gini
        
        feature_values = X[best_feature_name].unique().tolist()
           
        return Node(feature=best_feature_name, feature_values=feature_values)
        
    def _information_Gain(self, feature, X, Y):
        
        def entropy(y):
            classes ,counts = np.unique(y, return_counts=True)
            total_count = len(y)
            classes_count = len(classes)
            entropy  =0
            for i in range(classes_count):
                predict = counts[i] / total_count
                entropy -= predict*np.log2(predict)
            return entropy
        
        X_count = len(X)
        total_entropy = 0
        for value in feature.unique():
            mask = feature == value
            subset_X = X[mask]
            subset_Y = Y[mask]
            p_value = len(subset_X)/X_count
            total_entropy += p_value*entropy(subset_Y)
        
        return entropy(Y) -total_entropy 
    
    def _gini_Split(self, feature, X, Y):
        
        def gini(y):
            classes, counts = np.unique(y, return_counts=True)
            total_count = len(y)
            classes_count = len(classes)
            gini = 0
            for i in range(classes_count):
                gini += (counts[i]/total_count)**2
            return 1 - gini

        X_count = len(X)
        total_gini = 0
        for value in feature.unique():
            mask = feature == value
            subset_X = X[mask]
            subset_Y = Y[mask]
            p_value = len(subset_X)/X_count
            total_gini += p_value*gini(subset_Y)

        return total_gini

    def _calculate_Value(self, Y):
        # Where is it used and what does it do?
        return max(set(Y), key=list(Y).count)

    def fit(self, X_Train, Y_Train):
        self.root = self._create_Tree(X_Train, Y_Train)

    def predict(self, X):
        def _move_Tree(sample, root):
            # TODO: If leaf node return pred value, Or find recursively leaf node
            pass

        # TODO: Apply _move_Tree to each sample in X
        pass