import pandas as pd
import numpy as np
import math
import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly
import plotly.graph_objs as go

df = pd.read_csv("cricket_4_unlabelled.csv", index_col = 0)
# cols = list(df.columns)
# for col in cols:
# 	df[col] = (df[col] - df[col].mean())/df[col].std(ddof = 0)
X = df.to_numpy()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0], X[:,1], X[:,2], c=X[:,2], cmap='Greens');
plt.show()

markersize = X[:,4]/12
markercolor = X[:,3]

print(markersize.shape)

#Make Plotly figure
fig1 = go.Scatter3d(x=X[:,0],
                    y=X[:,1],
                    z=X[:,2],
                    marker=dict(size=markersize,
                                color=markercolor,
                                opacity=0.9,
                                reversescale=True,
                                colorscale='Blues'),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="total_balls_faced"),
                                yaxis=dict( title="dot_balls_faced"),
                                zaxis=dict(title="number_of_fours")),)


plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("5D Plot.html"))

for k in range(2, 11, 1):
	kmeans = KMeans(n_clusters = k, random_state = 10).fit(X)
	print("For k = {}, silhouette_score = {}".format(k, silhouette_score(X, kmeans.labels_)))