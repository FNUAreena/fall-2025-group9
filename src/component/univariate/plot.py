import matplotlib.pyplot as plt
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred, model_name, plot_path):
    """
    Plots actual vs. predicted values and saves the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5, label='Ideal (y = x)')
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Actual Production Cost")
    plt.ylabel("Predicted Production Cost")
    plt.title(f"{model_name} Predicted vs Actual")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
