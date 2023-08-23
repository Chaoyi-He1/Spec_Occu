import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns


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
        pass