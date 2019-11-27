import numpy as np


# We import sklearn.
import sklearn
from sklearn.manifold import TSNE


# We'll hack a bit with the t-SNE code in sklearn 0.15.2.

# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib


# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})



digits = load_digits()

# X = digits.data
# y = digits.target

X = np.load("56_train.npy")
Y = np.load("train_target.npy")

# We first reorder the data points according to the handwritten numbers.
X = np.vstack([X[Y==i] for i in range(10)])
y = np.hstack([Y[Y==i] for i in range(10)])


digits_proj = TSNE(random_state=RS).fit_transform(X)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

scatter(digits_proj, y)
plt.savefig('56_train.png', dpi=600)