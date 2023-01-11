import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
import os

""" plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
 """
data_path="./MopsiLocationsUntil2012-Finland.txt"
assert os.path.isfile(data_path), "File not found? An Exception"
df=pd.read_csv(data_path,sep = "\t",names=["Latitude", "Longitude"])
X = df.loc[:, ["Latitude", "Longitude"]]
X.head()

""" 
# Set up K-Means clustering with a fixed start and stop at 3 clusters
kmeans = KMeans(n_clusters=5, random_state=0).fit(x)

# Plot the data
sns.set_style("darkgrid")
plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_, cmap=plt.get_cmap("winter"))

# Save the axes limits of the current figure
x_axis = plt.gca().get_xlim()
y_axis = plt.gca().get_ylim()

# Draw cluster boundaries and centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='x')
vor = Voronoi(centers)
voronoi_plot_2d(vor, ax=plt.gca(), show_points=False, show_vertices=False)

# Resize figure as needed
plt.gca().set_xlim(x_axis)
plt.gca().set_ylim(y_axis)

# Remove ticks from the plot
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show() """