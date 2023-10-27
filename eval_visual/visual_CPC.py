import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def t_SNE_CPC(cpc_out, cpc_label):
    tsne = TSNE(n_components=3, random_state=0)
    cpc_embedded = tsne.fit_transform(cpc_out)
    
    # generate colors according to labels
    colormap = plt.cm.get_cmap('tab10', 30)
    colors = [colormap(i) for i in cpc_label]
    
    # Plot the embedded data points in 3D
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(cpc_embedded[:, 0], cpc_embedded[:, 1], cpc_embedded[:, 2], c=colors)
    plt.show()
    # save the figure
    plt.savefig('cpc_embedded.png')
    