This project contains the trained models and database associated with the paper: Yuting Zhang, Héctor Marín-Moreno, Susan Gourvenec, Advancing Vs estimation from CPTu for engineering practice: A data-driven approach.


Files:

main.py loads the trained models from each folder along with the corresponding data files and performs predictions.

Excel files contain the database (Train, Validation, and Test datasets). Data file names correspond to the number of features and data generation methods. For example, Data_8F_D4 represents predictions using eight features and Augmented-Linear (D4).

Model folders store the best-performing models (DNN, LR, RF, SVR, XGB) and the feature scaling used in each case, following the same naming convention as the data files.


How to use:

Users need to modify two lines in the main.py that define the desired number of features and the database generation method in order to locate the target model. Next, the new data should be pasted into the corresponding data file, following the parameter names in the header line. The script will then generate an Excel file containing the original input features along with predictions from DNN, LR, SVR, RF, and XGB.
