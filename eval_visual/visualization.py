import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def plot_predictions(ax,
                     prediction_temporal,
                     futures_label,
                     line_alpha=0.7,
                     line_width=0.2,
                     edge_width=2,
                     circle_edge_width=0.5,
                     node_circle_size=0.3,
                     batch_num=0,
                     kde=False):
    r, b, l, d = prediction_temporal.shape
    if kde:
        # Create a mesh grid for the (l, c) coordinates
        data = prediction_temporal.sum(axis=0).squeeze().detach().cpu().numpy()

        # Flatten the (l, c) coordinates and data for KDE
        values = data.ravel()

        # Create a KDE using SciPy's multivariate_normal
        kde = multivariate_normal.pdf(values, mean=np.mean(values), cov=np.cov(values))

        # Reshape the KDE values back to (l, c)
        kde = kde.reshape((l, c))

        # Plot the surface
        l, c = np.meshgrid(np.arange(l), np.arange(c))
        ax.plot_surface(l, c, kde.T, cmap='viridis')

        # Customize the plot as needed
        ax.set_xlabel('L')
        ax.set_ylabel('C')
        ax.set_zlabel('KDE Value')
        ax.set_title('2D KDE in 3D Plot')