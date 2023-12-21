import json, os
from natsort import natsorted
import argparse
import pandas as pd
import numpy as np
import pickle
import argparse

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

pblock_page_dict = {
            'p4': ['4','5','6','7'], 
            'p8': ['8','9','10','11'],
            'p12': ['12','13','14','15'],
            'p16': ['16','17','18','19'],
            'p20': ['20','21','22','23'],

            'p2': ['2','3'], 
            'p4_p0': ['4','5'], 'p4_p1': ['6','7'],
            'p8_p0': ['8','9'], 'p8_p1': ['10','11'], 
            'p12_p0': ['12','13'], 'p12_p1': ['14','15'], 
            'p16_p0': ['16','17'], 'p16_p1': ['18','19'],
            'p20_p0': ['20','21'], 'p20_p1': ['22','23'],

            'p2_p0': ['2'], 'p2_p1': ['3'],
            'p4_p1_p0': ['6'], 'p4_p1_p1': ['7'],
            'p8_p0_p0': ['8'], 'p8_p0_p1': ['9'], 'p8_p1_p0': ['10'], 'p8_p1_p1': ['11'],
            'p12_p0_p0': ['12'], 'p12_p0_p1': ['13'], 'p12_p1_p0': ['14'], 'p12_p1_p1': ['15'],
            'p16_p0_p0': ['16'], 'p16_p0_p1': ['17'], 'p16_p1_p0': ['18'], 'p16_p1_p1': ['19'],
            'p20_p0_p0': ['20'], 'p20_p0_p1': ['21'], 'p20_p1_p0': ['22'], 'p20_p1_p1': ['23']
}

knn_param_grid = [
    {'leaf_size': [20, 30, 40], 
     'n_neighbors': [3,5,7,9],
    },
]

rfc_param_grid = [
    {'n_estimators': [50, 100, 200],
     'max_features': ['sqrt', None],
    },
]

mlp_param_grid = [
    {'hidden_layer_sizes': [(3,),(5,),(10,),(10,10,)],
     'learning_rate': ['constant', 'adaptive'],
    },
]

#####

# RECALL_THRESHOLD = 0.99
HARD_CONSTRAINT = 0.7


# Only for LUTs
class hardConstraintClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return (X['LUT ratio'] >= HARD_CONSTRAINT)
        # return (X['LUT ratio'] >= HARD_CONSTRAINT) | (X['path delay'] >= 1.0)

# If any of resource is over the constraint, predicts that it fails
class hardConstraintClassifierAll(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return (X['LUT ratio'] >= HARD_CONSTRAINT) | (X['BRAM ratio'] >= HARD_CONSTRAINT) | (X['DSP ratio'] >= HARD_CONSTRAINT)
        # return (X['LUT ratio'] >= HARD_CONSTRAINT) | (X['BRAM ratio'] >= HARD_CONSTRAINT) | (X['DSP ratio'] >= HARD_CONSTRAINT) | (X['path delay'] >= 1.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--frequency', help="default: freq=200", default=200)
    parser.add_argument('-t','--threshold', help="default: RECALL_THRESHOLD=0.95", type=float, default=0.95)

    args = parser.parse_args()
    FREQ = str(args.frequency)
    RECALL_THRESHOLD = args.threshold

    # Precision value when recall is at least RECALL_THRESHOLD
    def precision_score_at_95_recall(y_train_1, y_pred_prob):
        precisions, recalls, thresholds = precision_recall_curve(y_train_1, y_pred_prob)
        idx_for_95_recall = (recalls >= RECALL_THRESHOLD).argmin() - 1
        threshold_for_95_recall = thresholds[idx_for_95_recall]
        y_train_recall_95 = (y_pred_prob >= threshold_for_95_recall)
        y_train_recall_95 = (y_pred_prob >= threshold_for_95_recall)
        recall_at_95_recall = recall_score(y_train_1, y_train_recall_95)
        precision_at_95_recall = precision_score(y_train_1, y_train_recall_95)
        return precision_at_95_recall


    # for pblock in pblock_page_dict:
    recall_dict = {}
    precision_dict = {}
    # pblock = "p4_p0"

    # os.system("mkdir -p ./rpt_dir/impl_results/" + FREQ + "MHz/analysis/")
    for pblock in natsorted(pblock_page_dict):
        print(pblock)

        impl_results_df = pd.read_csv('./rpt_dir/impl_results/' + str(FREQ) + 'MHz/csv/' + pblock + '.csv',skipinitialspace=True)
        assert(len(list(impl_results_df)) == 9)

        # label is True if the result is Implementation failed
        # Don't include Timing violation!
        impl_results_df['label'] = (impl_results_df['label'] == 'Implementation failed')
        impl_results_df = impl_results_df.drop('path delay', axis=1)
        assert(len(list(impl_results_df)) == 8)

        X = impl_results_df[list(impl_results_df)[1:]]
        y = impl_results_df['label']

        X_train, X_test, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # min_max_scaler = MinMaxScaler(clip=True)
        standard_scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(standard_scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

        # Classifier candidates
        # knn_model = KNeighborsClassifier()
        rfc_model = RandomForestClassifier(random_state = 42)
        # mlp_model = MLPClassifier(random_state=42, max_iter=1000)

        custom_scorer = make_scorer(precision_score_at_95_recall,
                                    greater_is_better=True, needs_proba=True)
        # Grid seasrch
        # grid_search_knn = GridSearchCV(knn_model, knn_param_grid, cv=5, scoring=custom_scorer)
        grid_search_rfc = GridSearchCV(rfc_model, rfc_param_grid, cv=5, scoring=custom_scorer)
        # grid_search_mlp = GridSearchCV(mlp_model, mlp_param_grid, cv=5, scoring=custom_scorer)

        # grid_search_knn.fit(X_train_scaled, y_train_1)
        grid_search_rfc.fit(X_train, y_train_1) # scaling not necessary
        # grid_search_mlp.fit(X_train_scaled, y_train_1)

        # grid_search_list = [grid_search_knn, grid_search_rfc, grid_search_mlp]
        grid_search_list = [grid_search_rfc]

        max_val = -1
        best_model = None
        for grid_search in grid_search_list:
            if grid_search.best_score_ > max_val:
                max_val = grid_search.best_score_
                best_model = grid_search.best_estimator_

        # Best Classifier for this PR page
        if best_model.__class__.__name__ != 'RandomForestClassifier':
            X_train_selected = X_train_scaled
            X_test_scaled = pd.DataFrame(standard_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
            X_test_selected = X_test_scaled
        else:
            X_train_selected = X_train
            X_test_selected = X_test


        dir_name = str(RECALL_THRESHOLD).split('.')[1]
        os.system('mkdir -p ./rpt_dir/classifier/' + str(FREQ) + 'MHz/' + dir_name)
        # save the best model, and standard_scaler
        pickle.dump(best_model, open('./rpt_dir/classifier/' + str(FREQ) + 'MHz/' + dir_name + '/' + pblock + '.pickle', "wb"))
        pickle.dump(standard_scaler, open('./rpt_dir/classifier/' + str(FREQ) + 'MHz/' + dir_name + '/' + pblock + '_sc.pickle', "wb"))


        # best_model.fit(X_train, y_train_1)
        y_test_pred = best_model.predict(X_test_selected)

        # Default ver. without adjusting threshold
        recall_default = recall_score(y_test_1, y_test_pred)
        precision_default = precision_score(y_test_1, y_test_pred)

        # Recall >= 0.95 ver.
        y_test_pred_prob = best_model.predict_proba(X_test_selected)
        y_test_pred_prob = y_test_pred_prob[:,1] # predictions for label == 1
        precisions, recalls, thresholds = precision_recall_curve(y_test_1, y_test_pred_prob)
        idx_for_95_recall = (recalls >= RECALL_THRESHOLD).argmin() - 1
        threshold_for_95_recall = thresholds[idx_for_95_recall]
        y_test_recall_95 = (y_test_pred_prob >= threshold_for_95_recall)
        recall_at_95_recall = recall_score(y_test_1, y_test_recall_95)
        precision_at_95_recall = precision_score(y_test_1, y_test_recall_95)

        # Hard Constraint Classifier, for LUTs only, Note that here, it's NOT scaled training data
        hardClassifier = hardConstraintClassifier()
        y_test_pred_hard = hardClassifier.predict(X_test) # original X_test
        recall_hard = recall_score(y_test_1, y_test_pred_hard)
        precision_hard = precision_score(y_test_1, y_test_pred_hard)

        # Hard Constraint Classifier, for all LUTs, BRAMs, DSPs
        hardClassifierAll = hardConstraintClassifierAll()
        y_test_pred_hard_all = hardClassifierAll.predict(X_test) # original X_test
        recall_hard_all = recall_score(y_test_1, y_test_pred_hard_all)
        precision_hard_all = precision_score(y_test_1, y_test_pred_hard_all)

        recall_dict[pblock] = (recall_default, recall_at_95_recall, recall_hard, recall_hard_all, best_model.__class__.__name__)
        precision_dict[pblock] = (precision_default, precision_at_95_recall, precision_hard, precision_hard_all, best_model.__class__.__name__)

        # recall_dict[pblock] = (recall_hard, recall_hard_all)
        # precision_dict[pblock] = (precision_hard, precision_hard_all)


    filedata = 'pblock, default, thrshd adjusted, LUT hard, all hard, best model\n'
    file_name = 'recall_' + str(RECALL_THRESHOLD).split('.')[1]
    print("Recall results")
    for pblock in natsorted(pblock_page_dict):
        print(pblock, end=" ")
        print(*recall_dict[pblock])
        default = recall_dict[pblock][0]
        thrshd_adjusted = recall_dict[pblock][1]
        LUT_hard = recall_dict[pblock][2]
        all_hard = recall_dict[pblock][3]
        best_model = recall_dict[pblock][4]
        filedata = filedata + str(default) + ", " + str(thrshd_adjusted) + ", " + str(LUT_hard) + ", " + str(all_hard) + ", "\
                            + str(best_model) + "\n"
    with open("./rpt_dir/impl_results/" + FREQ + "MHz/analysis/" + file_name + ".csv", "w") as outfile:
        outfile.write(filedata)

    filedata = 'pblock, default, thrshd adjusted, LUT hard, all hard, best model\n'
    file_name = 'precision_' + str(RECALL_THRESHOLD).split('.')[1]
    print("Precision results")
    for pblock in natsorted(pblock_page_dict):
        print(pblock, end=" ")
        print(*precision_dict[pblock])
        default = precision_dict[pblock][0]
        thrshd_adjusted = precision_dict[pblock][1]
        LUT_hard = precision_dict[pblock][2]
        all_hard = precision_dict[pblock][3]
        best_model = precision_dict[pblock][4]
        filedata = filedata + str(default) + ", " + str(thrshd_adjusted) + ", " + str(LUT_hard) + ", " + str(all_hard) + ", "\
                            + str(best_model) + "\n"
    with open("./rpt_dir/impl_results/" + FREQ + "MHz/analysis/" + file_name + ".csv", "w") as outfile:
        outfile.write(filedata)