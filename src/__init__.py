# src package — N-Queens (Task 1) + Kaggle Loan Classification (Task 2)

# === Task 1: N-Queens ===
try:
    from .nqueens import (
        generate_random_state,
        calculate_h,
        get_neighbors,
        hill_climbing_basic,
        hill_climbing_sideways,
        hill_climbing_stochastic,
        hill_climbing_random_restart,
        simulated_annealing,
        ga_fitness,
        ga_selection_roulette,
        ga_crossover,
        ga_mutate,
        genetic_algorithm,
        print_board,
        print_state_array,
        print_separator,
        print_h_history,
        visualize_search,
    )
except ImportError:
    pass  # nqueens module may not be available in all contexts

# === Task 2: Kaggle Loan Classification ===
# Import langsung dari sub-modules di notebook:
#   from src.data import load_train, load_test
#   from src.cleaning import DataCleaner
#   from src.preprocessing import Preprocessor
#   from src.algorithms.decision_tree import DecisionTreeCART
#   from src.algorithms.logistic_regression import LogisticRegressionScratch
#   from src.algorithms.svm import LinearSVMScratch
#   from src.optimizers import GradientDescent, Adam
#   from src.evaluation import cross_validate, macro_f1_score
#   from src.visualization import plot_tree_structure, ...
#   from src.predict import generate_submission
