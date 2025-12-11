import sys

import os
import pickle
import datetime
import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

from ece.blackbox import BlackBox
from ece.random_ece import RandomECE
from ece.feature_ece import FeatureECE
from ece.neighbor_ece import NeighborECE
from ece.cluster_ece import KMeansECE
from ece.tree_ece import TreeECE
from ece.ensemble_ece import EnsembleECE
from ece.distr_ece import DistrECE
from ece.casebased_ece import CaseBasedECE

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset

# DPG imports
import sys as sys_module
sys_module.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DPG')))
from dpg.core import DecisionPredicateGraph
from dpg.sklearn_dpg import select_dataset, test_dpg
from metrics.graph import GraphMetrics
from metrics.nodes import NodeMetrics

# CounterFactual imports
sys_module.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from CounterFactualModel import CounterFactualModel
from ConstraintParser import ConstraintParser


def extract_dpg_constraints(X_train, y_train, feature_names, output_dir, model_type='RandomForestClassifier',
                           n_learners=5, purity_threshold=0.001, decimal_threshold=2, verbose=False):
    """
    Extract decision boundaries from training data using DPG.
    
    Args:
        X_train: Training feature data (n_samples, n_features)
        y_train: Training labels (n_samples,)
        feature_names: List of feature names
        output_dir: Directory to save metrics file
        model_type: Type of ensemble model ('RandomForestClassifier', 'ExtraTreesClassifier', etc.)
        n_learners: Number of learners in ensemble
        purity_threshold: Percentage variance threshold for path filtering
        decimal_threshold: Rounding precision for feature values
        verbose: Print debug information
        
    Returns:
        Path to generated metrics file
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train ensemble model on the provided data
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    
    model_classes = {
        'RandomForestClassifier': RandomForestClassifier,
        'ExtraTreesClassifier': ExtraTreesClassifier,
    }
    
    if model_type not in model_classes:
        model_type = 'RandomForestClassifier'
    
    model = model_classes[model_type](
        n_estimators=n_learners,
        random_state=42,
        n_jobs=-1
    )
    
    if verbose:
        print(f"[DPG] Training {model_type} with {n_learners} learners...")
    
    model.fit(X_train, y_train)
    
    # Initialize DPG with temporary config
    temp_config_path = os.path.join(output_dir, 'temp_config.yaml')
    import yaml
    config_data = {
        'dpg': {
            'default': {
                'perc_var': purity_threshold,
                'decimal_threshold': decimal_threshold,
                'n_jobs': -1
            },
            'visualization': {
                'graph_attrs': {'bgcolor': 'white', 'rankdir': 'R'},
                'node_attrs': {'shape': 'box', 'fillcolor': '#ffc3c3'},
                'class_node': {'fillcolor': '#a4c2f4', 'shape': 'box', 'style': 'rounded, filled'}
            }
        }
    }
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    try:
        # Initialize DPG
        target_names = [str(c) for c in np.unique(y_train)]
        dpg = DecisionPredicateGraph(
            model=model,
            feature_names=list(feature_names),
            target_names=target_names,
            config_file=temp_config_path
        )
        
        if verbose:
            print("[DPG] Extracting decision paths...")
        
        # Fit DPG to extract paths
        dot = dpg.fit(X_train)
        
        # Convert to NetworkX and extract metrics
        if verbose:
            print("[DPG] Computing graph metrics...")
        
        dpg_model, nodes_list = dpg.to_networkx(dot)
        
        if len(nodes_list) < 2:
            if verbose:
                print("[DPG] Warning: Insufficient nodes for analysis, returning empty constraints")
            return None
        
        # Extract graph metrics (which includes class boundaries)
        df_dpg = GraphMetrics.extract_graph_metrics(
            dpg_model, 
            nodes_list,
            target_names=target_names
        )
        
        # Save metrics to file
        metrics_filename = os.path.join(
            output_dir,
            f'{model_type}_l{n_learners}_pv{purity_threshold}_t{decimal_threshold}_dpg_metrics.txt'
        )
        
        if verbose:
            print(f"[DPG] Saving metrics to {metrics_filename}")
        
        with open(metrics_filename, 'w') as f:
            for key, value in df_dpg.items():
                f.write(f"{key}: {value}\n")
        
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        return metrics_filename
        
    except Exception as e:
        if verbose:
            print(f"[DPG] Error during DPG extraction: {str(e)}")
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return None


class DPGCounterFactualExplainer:
    """
    Wrapper class for DPG-based counterfactual generation.
    Implements the same interface as ECE explainers.
    """
    
    def __init__(self, variable_features, constraints_dict, dict_non_actionable=None, 
                 verbose=False, population_size=20, max_generations=60):
        """
        Initialize DPG Counterfactual Explainer.
        
        Args:
            variable_features: List of variable feature indices
            constraints_dict: Dictionary of constraints from ConstraintParser
            dict_non_actionable: Dict mapping features to actionability rules ('none', 'non_decreasing', etc.)
            verbose: Print debug information
            population_size: GA population size
            max_generations: GA max generations
        """
        self.variable_features = variable_features
        self.constraints_dict = constraints_dict
        self.dict_non_actionable = dict_non_actionable or {}
        self.verbose = verbose
        self.population_size = population_size
        self.max_generations = max_generations
        
        self.blackbox = None
        self.X_train = None
        self.cf_model = None
        self.feature_names = None
        
    def fit(self, blackbox, X_train, feature_names=None):
        """
        Fit the explainer with blackbox and training data.
        
        Args:
            blackbox: BlackBox wrapper object
            X_train: Training data
            feature_names: Optional list of feature names for constraint mapping
        """
        self.blackbox = blackbox
        self.X_train = X_train
        self.feature_names = feature_names
        
        # Initialize CounterFactualModel with constraints
        self.cf_model = CounterFactualModel(
            model=blackbox,
            constraints=self.constraints_dict,
            dict_non_actionable=self.dict_non_actionable,
            verbose=self.verbose
        )
        
        if self.verbose:
            print("[DPG-CF] Explainer fitted successfully")
        
    def get_counterfactuals(self, x, k=1, search_diversity=False, covertype=None, 
                           lambda_par=1.0, cf_rate=0.5, cf_rate_incr=0.1):
        """
        Generate k counterfactuals for instance x.
        
        Args:
            x: Instance to explain (numpy array)
            k: Number of counterfactuals to generate
            search_diversity: Ignored (kept for interface compatibility)
            covertype: Ignored (kept for interface compatibility)
            lambda_par: Ignored (kept for interface compatibility)
            cf_rate: Ignored (kept for interface compatibility)
            cf_rate_incr: Ignored (kept for interface compatibility)
            
        Returns:
            List of k counterfactual arrays
        """
        cf_list = []
        
        # Predict current class
        current_class = self.blackbox.predict(x.reshape(1, -1))[0]
        
        # Get target classes (opposite of current class)
        unique_classes = np.unique(self.X_train)
        target_classes = [c for c in unique_classes if c != current_class]
        
        if not target_classes:
            if self.verbose:
                print("[DPG-CF] No target class available")
            return []
        
        # Generate k counterfactuals with different seeds
        for i in range(k):
            for target_class in target_classes[:1]:  # Use first target class
                try:
                    # Convert x to dict if needed for CounterFactualModel
                    x_dict = {}
                    if self.feature_names:
                        for j, fname in enumerate(self.feature_names):
                            # Ensure values are floats
                            x_dict[fname] = float(x[j])
                    else:
                        for j in range(len(x)):
                            x_dict[str(j)] = float(x[j])
                    
                    # Generate counterfactual with different seed
                    np.random.seed(42 + i)  # Different seed for diversity
                    
                    cf_dict = self.cf_model.generate_counterfactual(
                        sample=x_dict,
                        target_class=int(target_class),
                        population_size=self.population_size,
                        generations=self.max_generations
                    )
                    
                    if cf_dict is not None:
                        # Convert dict back to numpy array, ensuring floats
                        if self.feature_names:
                            cf_array = np.array([float(cf_dict.get(fname, x[j])) 
                                               for j, fname in enumerate(self.feature_names)], dtype=np.float64)
                        else:
                            cf_array = np.array([float(cf_dict.get(str(j), x[j])) 
                                               for j in range(len(x))], dtype=np.float64)
                        cf_list.append(cf_array)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"[DPG-CF] Error generating CF {i}: {str(e)}")
        
        return cf_list[:k]  # Return exactly k counterfactuals


def experiment(cfe, bb, X_train, variable_features, metric, continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results, filename_cf, features_names,
               covertype, n_estimators, dpg_constraints_path=None, dpg_dict_non_actionable=None):

    time_start = datetime.datetime.now()
    max_samples_count = len(X_train) * 0.2
    if max_samples_count > 1000:
        max_samples = 1000 / len(X_train)
    else:
        max_samples = 0.2
    
    if cfe == 'dpg-cf':
        # Parse constraints from file
        if dpg_constraints_path is None or not os.path.exists(dpg_constraints_path):
            print(f'DPG constraints file not found: {dpg_constraints_path}')
            raise Exception
        
        parser = ConstraintParser(dpg_constraints_path)
        constraints_dict = parser.read_constraints_from_file()
        
        # Map feature names to dict for actionability constraints
        # Default: all features mutable ('none')
        dict_non_actionable = dpg_dict_non_actionable or {}
        
        exp = DPGCounterFactualExplainer(
            variable_features=variable_features,
            constraints_dict=constraints_dict,
            dict_non_actionable=dict_non_actionable,
            verbose=False,
            population_size=20,
            max_generations=60
        )
    elif cfe == 'sace-ens-d':
        exp = EnsembleECE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                          base_estimator='dist', n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                          n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-t':
        exp = EnsembleECE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                          base_estimator='tree', n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                          n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-f':
        exp = EnsembleECE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                          base_estimator='feat', n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                          n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-n':
        exp = EnsembleECE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                          base_estimator='neigh', n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                          n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-c':
        exp = EnsembleECE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                          base_estimator='cb', n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                          n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-l':
        exp = EnsembleECE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                          base_estimator='clus', n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                          n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-r':
        exp = EnsembleECE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                          base_estimator='rand', n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                          n_jobs=-1, verbose=0)
    elif cfe == 'sace-ens-h':
        exp = EnsembleECE(variable_features, weights=None, metric=metric,
                          feature_names=None, continuous_features=continuous_features,
                          categorical_features_lists=categorical_features_lists, normalize=False, pooler=None,
                          n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                          base_estimator='pippo',
                          estimators_params={
                               'dist': {'n_attempts': 10,
                                        'n_batch': 1000,
                                        'stopping_eps': 0.01,
                                        'kind': 'gaussian_matched',
                                        'tol': 0.01},
                               'tree': {'use_instance_weights': False,
                                        'kernel_width': None,
                                        'min_samples_leaf': 0.01,
                                        'max_depth': None,
                                        'closest_in_leaf': True},
                               'feat': {'nbr_intervals': 10,
                                        'nbr_features_to_test': 1,
                                        'tol': 0.01},
                               # 'neig': {'random_samples': 100},
                               # 'rand': {}
                               # 'cb': {},
                               # 'clus': {},
                           },
                          n_jobs=-1, verbose=0)
    else:
        print('unknown counterfactual explainer %s' % cfe)
        raise Exception

    exp.fit(bb, X_train)

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    if covertype is None:
        print(datetime.datetime.now(), dataset, black_box, cfe, 'nbr_estimators', n_estimators)
    else:
        print(datetime.datetime.now(), dataset, black_box, cfe, covertype, 'nbr_estimators', n_estimators)

    for test_id, i in enumerate(index_test_instances):

        if covertype is None:
            print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(index_test_instances),
                  '%.2f' % (test_id / len(index_test_instances)))
        else:
            print(datetime.datetime.now(), dataset, black_box, cfe, covertype, test_id, len(index_test_instances),
                  '%.2f' % (test_id / len(index_test_instances)))
        x = X_test[i]
        y_val = bb.predict(x.reshape(1, -1))[0]

        cf_list_all = list()

        x_eval_list = list()

        for k in [
            1,  # 2, 3, 4,
            5,  # 8,
            10,  # 12, 14,
            15,  # 16, 18, 20
        ]:

            time_start_i = datetime.datetime.now()

            cf_list = exp.get_counterfactuals(x, k=k, search_diversity=search_diversity,
                                              covertype=covertype,
                                              lambda_par=1.0, cf_rate=0.5, cf_rate_incr=0.1)

            time_test = (datetime.datetime.now() - time_start_i).total_seconds()

            x_eval = evaluate_cf_list(cf_list, x, bb, y_val, k, variable_features,
                                      continuous_features_all, categorical_features_all, X_train, X_test,
                                      ratio_cont, nbr_features)

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = cfe
            x_eval['idx'] = i
            x_eval['k'] = k
            x_eval['known_train'] = known_train
            x_eval['search_diversity'] = search_diversity
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            x_eval['metric'] = metric if isinstance(metric, str) else '.'.join(metric)
            x_eval['variable_features_flag'] = len(variable_features) > 0
            if cfe != 'dpg-cf':
                x_eval['n_estimators'] = n_estimators

            x_eval_list.append(x_eval)
            if len(cf_list):
                cf_list_all.append(cf_list[0])

        if len(cf_list_all) > 1:
            instability_si = np.mean(squareform(pdist(np.array(cf_list_all), metric='euclidean')))
        else:
            instability_si = 0.0

        for x_eval in x_eval_list:
            x_eval['instability_si'] = instability_si

        df = pd.DataFrame(data=x_eval_list)
        
        # Adjust column list based on method
        result_columns = columns.copy()
        if cfe != 'dpg-cf' and 'n_estimators' not in result_columns:
            result_columns = result_columns + ['n_estimators']
        elif cfe == 'dpg-cf' and 'n_estimators' in result_columns:
            result_columns = [c for c in result_columns if c != 'n_estimators']
        
        # Only keep columns that exist in df
        result_columns = [c for c in result_columns if c in df.columns]
        df = df[result_columns]

        if not os.path.isfile(filename_results):
            df.to_csv(filename_results, index=False)
        else:
            df.to_csv(filename_results, mode='a', index=False, header=False)


def main():

    nbr_test = 20
    dataset = 'titanic'
    black_box = 'RF'
    normalize = 'standard'

    np.random.seed(random_state)

    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in blackbox_list:
        print('unknown black box %s' % black_box)
        return -1

    print(datetime.datetime.now(), dataset, black_box)

    data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                               random_state=random_state, encode=None if black_box == 'LGBM' else 'onehot')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    class_values = data['class_values']
    if dataset == 'titanic':
        class_values = ['Not Survived', 'Survived']
    features_names = data['feature_names']
    variable_features = data['variable_features']
    variable_features_names = data['variable_features_names']
    continuous_features = data['continuous_features']
    continuous_features_all = data['continuous_features_all']
    categorical_features_lists = data['categorical_features_lists']
    categorical_features_lists_all = data['categorical_features_lists_all']
    categorical_features_all = data['categorical_features_all']
    continuous_features_names = data['continuous_features_names']
    categorical_features_names = data['categorical_features_names']
    scaler = data['scaler']
    nbr_features = data['n_cols']
    ratio_cont = data['n_cont_cols'] / nbr_features

    variable_cont_features_names = [c for c in variable_features_names if c in continuous_features_names]
    variable_cate_features_names = list(
        set([c.split('=')[0] for c in variable_features_names if c.split('=')[0] in categorical_features_names]))

    if black_box in ['DT', 'RF', 'SVM', 'NN', 'LGBM']:
        bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
    elif black_box in ['DNN']:
        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    bb = BlackBox(bb)

    known_train = True
    search_diversity = False
    metric = ('euclidean', 'jaccard')

    # Extract DPG constraints once for the dataset
    dpg_constraints_path = None
    dpg_output_dir = path_results + 'dpg_constraints/'
    os.makedirs(dpg_output_dir, exist_ok=True)
    
    print(datetime.datetime.now(), "Extracting DPG constraints...")
    dpg_constraints_path = extract_dpg_constraints(
        X_train=X_train,
        y_train=y_train,
        feature_names=features_names,
        output_dir=dpg_output_dir,
        model_type='RandomForestClassifier',
        n_learners=5,
        purity_threshold=0.001,
        decimal_threshold=2,
        verbose=True
    )
    
    if dpg_constraints_path is None:
        print("Warning: DPG constraint extraction failed, skipping dpg-cf method")
        dpg_enabled = False
    else:
        print(datetime.datetime.now(), f"DPG constraints saved to {dpg_constraints_path}")
        dpg_enabled = True

    # Define actionability constraints for DPG (feature_name -> 'none', 'non_decreasing', 'non_increasing', 'no_change')
    # For titanic dataset: all features are mutable ('none')
    dpg_dict_non_actionable = {}

    for cfe in [
        'sace-ens-h',
        'sace-ens-d',
        'sace-ens-t',
        'sace-ens-f',
        'sace-ens-n',
        'sace-ens-r',
        'dpg-cf' if dpg_enabled else None,
    ]:
        if cfe is None:
            continue

        if cfe == 'dpg-cf':
            # DPG doesn't use covertype parameter
            cfe_str = cfe
            filename_results = path_results + 'nbr_base_estimators_%s_%s_%s.csv' % (dataset, black_box, cfe_str)
            
            experiment(cfe, bb, X_train, variable_features, metric,
                       continuous_features, categorical_features_lists,
                       X_test, nbr_test, search_diversity, dataset, black_box, known_train,
                       continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
                       filename_results, filename_results, features_names, None, None,
                       dpg_constraints_path=dpg_constraints_path,
                       dpg_dict_non_actionable=dpg_dict_non_actionable)
        else:
            for covertype in ['majority',
                              # 'heuristic',
                              'naive',
                              'naive-sub',
                              'knn',
                              'knn-sub',
                              # 'knn-acc',
                              # 'knn-acc-sub'
                              ]:

                cfe_str = cfe + '_' + covertype
                filename_results = path_results + 'nbr_base_estimators_%s_%s_%s.csv' % (dataset, black_box, cfe_str)

                for nbr_estimators in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    experiment(cfe, bb, X_train, variable_features, metric,
                               continuous_features, categorical_features_lists,
                               X_test, nbr_test, search_diversity, dataset, black_box, known_train,
                               continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
                               filename_results, filename_results, features_names, covertype, nbr_estimators)


if __name__ == "__main__":
    main()
