
import numpy as np

def rmse(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError
    elif y_true.size == 0 or y_pred.size == 0:
        raise ValueError
    rmse_res = np.sqrt(np.mean((y_true - y_pred)**2))
    return round(rmse_res,3)
