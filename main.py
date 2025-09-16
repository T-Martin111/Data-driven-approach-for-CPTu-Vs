import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow import keras

# ========= CHANGE THESE TWO LINES TO CHOOSE DIFFERENT INPUT FEATURES AND DATABASE GENERATION METHODS=========
NUM_FEATURES = 3    # {3, 4, 6, 8}; "Three Features-3, Four Features-4, Six Features-6, Eight Features-8"
DATA_CODE    = "D1" # {"D1","D2","D3","D4","D5"}; "Point-Point-D1, Point-Average-D2, Augmented-Constant-D3, Augmented-Linear-D4, Augmented-Random-D5"
# ==========================================

FTAG        = f"{NUM_FEATURES}F"
MODEL_DIR   = Path(f"Model_{FTAG}_{DATA_CODE}")
DATAFILE    = f"Data_{DATA_CODE}.xlsx"
SHEET       = "Test"
OUTPUT_XLSX = f"Pred_{FTAG}_{DATA_CODE}.xlsx"
BATCHSIZE   = 64

def feature_indices(n: int):
    if n == 3: return [0, 1, 2]
    if n == 4: return [0, 1, 2, 3]
    if n == 6: return [0, 1, 2, 4, 5, 7]  # first 3 + 5th, 6th, 8th
    if n == 8: return list(range(8))
    raise ValueError("NUM_FEATURES must be one of {3, 4, 6, 8}")

def load_models(folder: Path):
    # DNN
    dnn_dir = folder / "best_DNN_model"
    models = {"DNN": keras.models.load_model(dnn_dir)}

    # LR, RF, SVR, XGB
    for name in ["LR", "RF", "SVR", "XGB"]:
        p = folder / f"best_{name}_model.pkl"
        models[name] = joblib.load(p)

    # Scaler
    scaler_path = folder / "feature_scaler.pkl"
    scaler = joblib.load(scaler_path)

    return models, scaler

def read_data(xlsx: str, sheet: str):
    df = pd.read_excel(xlsx, sheet_name=sheet)
    if df.shape[1] < 9:
        raise ValueError("Need ≥9 columns: first 8 inputs, 9th true output.")
    X_all = df.iloc[:, :8].copy()
    y  = df.iloc[:, 8].copy()
    return df, X_all, y

def predict(model, name: str, X: np.ndarray):
    if name == "DNN":
        return model.predict(X, batch_size=BATCHSIZE, verbose=0).reshape(-1)
    return np.asarray(model.predict(X)).reshape(-1)

def main():
    models, scaler = load_models(MODEL_DIR)
    raw_df, X_all, y_true = read_data(DATAFILE, SHEET)

    idx = feature_indices(NUM_FEATURES)
    X = X_all.iloc[:, idx].to_numpy(dtype=np.float32)

    # scale
    X = scaler.transform(X)

    out = raw_df.copy()
    for name, mdl in models.items():
        out[f"Prediction_{name}"] = predict(mdl, name, X)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as w:
        out.to_excel(w, sheet_name="predictions", index=False)

    print(f"Done: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()




# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# from tensorflow import keras
#
# # ========= CHANGE THESE TWO LINES =========
# NUM_FEATURES = 8    # {3,4,6,8}
# DATA_CODE    = "D3" # {"D1","D2","D3","D4","D5"}; "Point-Point-D1, Point-Average-D2, Augmented-Constant-D3, Augmented-Linear-D4, Augmented-Random-D5"
# # ==========================================
#
# FTAG        = f"{NUM_FEATURES}F"
# MODEL_DIR   = Path(f"Model_{FTAG}_{DATA_CODE}")
# DATAFILE    = f"Data_{FTAG}_{DATA_CODE}.xlsx"
# SHEET       = "Test"
# OUTPUT_XLSX = f"Pred_{FTAG}_{DATA_CODE}.xlsx"
# BATCHSIZE   = 64
#
# def feature_indices(n: int):
#     if n == 3: return [0, 1, 2]
#     if n == 4: return [0, 1, 2, 3]
#     if n == 6: return [0, 1, 2, 4, 5, 7]  # first 3 + 5th, 6th, 8th
#     if n == 8: return list(range(8))
#     raise ValueError("NUM_FEATURES must be one of {3, 4, 6, 8}")
#
# def load_models_strict(folder: Path):
#     if not folder.exists():
#         raise FileNotFoundError(f"Missing folder: {folder.resolve()}")
#
#     # DNN
#     dnn_dir = folder / "best_DNN_model"
#     if not dnn_dir.exists():
#         raise FileNotFoundError(f"Missing DNN SavedModel dir: {dnn_dir}")
#     models = {"DNN": keras.models.load_model(dnn_dir)}
#
#     # LR, RF, SVR, XGB
#     for name in ["LR", "RF", "SVR", "XGB"]:
#         p = folder / f"best_{name}_model.pkl"
#         if not p.exists():
#             raise FileNotFoundError(f"Missing model: {p}")
#         models[name] = joblib.load(p)
#
#     # Scaler
#     scaler_path = folder / "feature_scaler.pkl"
#     if not scaler_path.exists():
#         raise FileNotFoundError(f"Missing scaler: {scaler_path}")
#     scaler = joblib.load(scaler_path)
#
#     return models, scaler
#
# def read_data(xlsx: str, sheet: str):
#     df = pd.read_excel(xlsx, sheet_name=sheet)
#     if df.shape[1] < 9:
#         raise ValueError("Need ≥9 columns: first 8 inputs, 9th true output.")
#     X8 = df.iloc[:, :8].copy()
#     y  = df.iloc[:, 8].copy()
#     return df, X8, y
#
# def predict(model, name: str, X: np.ndarray):
#     if name == "DNN":
#         return model.predict(X, batch_size=BATCHSIZE, verbose=0).reshape(-1)
#     return np.asarray(model.predict(X)).reshape(-1)
#
# def main():
#     models, scaler = load_models_strict(MODEL_DIR)
#     raw_df, X_true, y_true = read_data(DATAFILE, SHEET)
#
#     idx = feature_indices(NUM_FEATURES)
#     X = X_true.iloc[:, idx].to_numpy(dtype=np.float32)
#
#     # scale
#     X = scaler.transform(X)
#
#     out = raw_df.copy()
#     for name, mdl in models.items():
#         out[f"Prediction_{name}"] = predict(mdl, name, X)
#
#     with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as w:
#         out.to_excel(w, sheet_name="predictions", index=False)
#
#     print(f"Done: {OUTPUT_XLSX}")
#
# if __name__ == "__main__":
#     main()
