# implementation based on
# https://github.com/yn-sun/e2epp/blob/master/build_predict_model.py
# and https://github.com/HandingWang/RF-CMOCO
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class CART:
    """ Classification and Regression Tree """
    def __init__(self, n_tree=1000):
        self.n_tree = n_tree
        self.name = 'carts'
        self.model = None

    @staticmethod
    def _make_decision_trees(train_data, train_label, n_tree):
        feature_record = []
        tree_record = []

        for i in range(n_tree):
            sample_idx = np.arange(train_data.shape[0])
            np.random.shuffle(sample_idx)
            train_data = train_data[sample_idx, :]
            train_label = train_label[sample_idx]

            feature_idx = np.arange(train_data.shape[1])
            np.random.shuffle(feature_idx)
            n_feature = np.random.randint(1, train_data.shape[1] + 1)
            selected_feature_ids = feature_idx[0:n_feature]
            feature_record.append(selected_feature_ids)

            dt = DecisionTreeRegressor()
            dt.fit(train_data[:, selected_feature_ids], train_label)
            tree_record.append(dt)

        return tree_record, feature_record

    def fit(self, train_data, train_label):
        self.model = self._make_decision_trees(train_data, train_label, self.n_tree)

    def predict(self, test_data):
        assert self.model is not None, "carts does not exist, call fit to obtain cart first"

        # redundant variable device
        trees, features = self.model[0], self.model[1]
        test_num, n_tree = len(test_data), len(trees)

        predict_labels = np.zeros((test_num, 1))
        for i in range(test_num):
            this_test_data = test_data[i, :]
            predict_this_list = np.zeros(n_tree)

            for j, (tree, feature) in enumerate(zip(trees, features)):
                predict_this_list[j] = tree.predict([this_test_data[feature]])[0]

            # find the top 100 prediction
            predict_this_list = np.sort(predict_this_list)
            predict_this_list = predict_this_list[::-1]
            this_predict = np.mean(predict_this_list)
            predict_labels[i, 0] = this_predict

        return predict_labels
    
    def get_feature_importance(self, n_features):
        """
        Calculate feature importance by aggregating across all trees.
        
        Args:
            n_features: Total number of features in the original dataset
            
        Returns:
            Dictionary with feature indices as keys and importance scores as values
        """
        assert self.model is not None, "carts does not exist, call fit first"
        
        trees, features = self.model[0], self.model[1]
        
        # Initialize importance accumulator
        feature_importance = np.zeros(n_features)
        feature_count = np.zeros(n_features)  # Track how many trees use each feature
        
        for tree, selected_features in zip(trees, features):
            # Get feature importance from this tree
            tree_importance = tree.feature_importances_
            
            # Map back to original feature indices
            for i, feat_idx in enumerate(selected_features):
                if i < len(tree_importance):
                    feature_importance[feat_idx] += tree_importance[i]
                    feature_count[feat_idx] += 1
        
        # Average importance across trees that used each feature
        avg_importance = np.zeros(n_features)
        for i in range(n_features):
            if feature_count[i] > 0:
                avg_importance[i] = feature_importance[i] / feature_count[i]
        
        # Normalize to sum to 1
        if avg_importance.sum() > 0:
            avg_importance = avg_importance / avg_importance.sum()
        
        return {
            'importance': avg_importance,
            'usage_count': feature_count,
            'usage_percentage': feature_count / len(trees) * 100
        }
    
    def print_feature_importance(self, n_features, feature_names=None, top_k=10):
        """
        Print feature importance analysis.
        
        Args:
            n_features: Total number of features
            feature_names: Optional list of feature names
            top_k: Number of top features to display
        """
        importance_data = self.get_feature_importance(n_features)
        importance = importance_data['importance']
        usage_pct = importance_data['usage_percentage']
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        
        print("\n" + "="*80)
        print("CART SURROGATE MODEL - FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        print(f"Number of trees: {self.n_tree}")
        print(f"Number of features: {n_features}")
        print(f"\nTop {top_k} Most Important Features:")
        print("-"*80)
        print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Usage %':<12}")
        print("-"*80)
        
        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            if feature_names and idx < len(feature_names):
                feat_name = feature_names[idx]
            else:
                feat_name = f"Feature_{idx}"
            print(f"{rank:<6} {feat_name:<30} {importance[idx]:<12.4f} {usage_pct[idx]:<12.1f}")
        
        print("-"*80)
        print(f"Total importance (top {top_k}): {importance[sorted_indices[:top_k]].sum():.4f}")
        print("="*80 + "\n")

