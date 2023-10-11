import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def t_SNE_CPC(cpc_out, cpc_label):
    tsne = TSNE(n_components=2, random_state=0)
    cpc_embedded = tsne.fit_transform(cpc_out)
    
    # generate colors according to labels
    colormap = plt.cm.get_cmap('tab10', 30)
    colors = [colormap(i) for i in cpc_label]
    
    # Plot the embedded data points
    plt.figure(figsize=(10, 10))
    plt.scatter(cpc_embedded[:, 0], cpc_embedded[:, 1], c=cpc_label, cmap='Spectral')
    # save the figure
    plt.savefig('t-SNE_CPC.png')
    