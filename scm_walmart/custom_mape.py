import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
def custom_retail_mape(y_true, y_pred, threshold=10):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    mask = y_true_np >= threshold

    if np.sum(mask) == 0:
        return np.nan

    return mean_absolute_percentage_error(y_true_np[mask], y_pred_np[mask])