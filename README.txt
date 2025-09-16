main.py loads the trained models from each folder along with the corresponding data files and performs predictions.

Excel files contain the database (Train, Validation, and Test datasets). Data file names correspond to the number of features and data generation methods. For example, Data_8F_D4 represents predictions using eight features and Augmented-Linear (D4).

Model folders store the best-performing models (DNN, LR, RF, SVR, XGB) and the feature scaling used in each case, following the same naming convention as the data files.