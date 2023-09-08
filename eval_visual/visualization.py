import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def plot_predictions(ax, fig,
                     prediction_prob,
                     futures_label,
                     kde=False):
    b, l, d = prediction_prob.shape
    if kde:
        # Create a mesh grid for the (l, c) coordinates
        data = prediction_prob.detach().cpu().numpy()

        # Flatten the (l, c) coordinates and data for KDE
        values = data.ravel()

        # Create a KDE using SciPy's multivariate_normal
        kde = multivariate_normal.pdf(values, mean=np.mean(values), cov=np.cov(values))

        # Reshape the KDE values back to (l, c)
        kde = kde.reshape((l, d))

        # Plot the surface
        l, d = np.meshgrid(np.arange(l), np.arange(d))
        ax.plot_surface(l, d, kde.T, cmap='viridis')

        # Customize the plot as needed
        ax.set_xlabel('L')
        ax.set_ylabel('F')
        ax.set_zlabel('KDE Value')
        ax.set_title('2D KDE in 3D Plot')
        ax.set_aspect('equal', adjustable='box')
        
    
    # Create a scatter plot with the 'Blues' colormap
    data = prediction_prob.detach().cpu().numpy()
    flat_data = data.flatten()
    scatter = ax.scatter(
        np.tile(np.arange(d), l),  # X-axis values
        np.repeat(np.arange(l), d),  # Y-axis values
        c=flat_data,  # Color based on data values
        cmap='Blues',  # Colormap ('Blues' for dark to bright blue)
        marker='s',  # Marker style (square)
        s=50,  # Marker size
    )

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('2D Probability Cloud')

    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, label='Probability')

    # Set the aspect ratio to equal for square cells
    ax.set_aspect('equal', adjustable='box')

