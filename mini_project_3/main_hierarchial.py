# Ayan Chakraborty 18EC10075
# CS60050 ML MiniProject 3 Code: CS4
# Code for Hierarchial Algo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# returns the euclidean distance between 2 data samples
def euclidean_distance(v1, v2):
	return np.linalg.norm(v1 - v2)

# read the dataset excluding the 1st coloumn which is the index coloumn
df = pd.read_csv("cricket_4_unlabelled.csv", index_col = 0)

# Z-Score normalization of each coloumn
cols = list(df.columns)
for col in cols:
	df[col] = (df[col] - df[col].mean())/df[col].std(ddof = 0)

# Convert the dataset into numpy array
X = df.to_numpy()
scores = []

# Initializations
num_clusters = X.shape[0]
data_indices = [[] for temp in range(num_clusters)]
class_ids = np.zeros((1, X.shape[0]))

# Initially each data sample is a cluster
for i in range(X.shape[0]):
	data_indices[i].append(i)

# Do hierarchial merging until only single cluster left
while(num_clusters > 1):
	k = num_clusters

	# Initialize the proximity matrix
	proximity_mat = np.zeros((num_clusters, num_clusters))
	
	# All diagonal elements set to infinity (Just for ease of calculation so that the minimum distance returned will not be to its own center (i.e, 0))
	np.fill_diagonal(proximity_mat, np.inf)

	# Compute the Proximity matrix
	for i in range(num_clusters):
		for j in range((i+1), num_clusters):
			# ALl possible distances between any two points in the two clusters
			dists = [[euclidean_distance(X[index_1,:], X[index_2,:]) for index_2 in data_indices[j]] for index_1 in data_indices[i]]
			# Choose the minimum distance (Single Linkage)
			proximity_mat[i, j] = np.amin(dists)
			# Due to symmetry
			proximity_mat[j, i] = proximity_mat[i, j]

	# Co-ordinates of the point where minimum distance between two clusters
	index = np.unravel_index(proximity_mat.argmin(), proximity_mat.shape)
	# Index of 1st Cluster to be merged
	min_index = min(index[0], index[1])
	# Index of 2nd Cluster to be merged
	max_index = max(index[0], index[1])

	# Merging the two clusters
	for i in range(num_clusters):
		if i == min_index:
			# Append the 2nd cluster to the 1st Cluster
			data_indices[i] = data_indices[min_index] + data_indices[max_index]

	# Delete the 2nd cluster
	del data_indices[max_index]
	# Update the number of clusters
	num_clusters = num_clusters - 1

	# Compute the cluster index for each data sample (for ease of calcualation of Silhouette score)
	for i in range(num_clusters):
		for index in data_indices[i]:
			class_ids[0, index] = i

	# print(len(data_indices))

	# Recompute the proximity matrix to account for 1 less cluster
	proximity_mat = np.zeros((num_clusters, num_clusters))
	np.fill_diagonal(proximity_mat, np.inf)

	for i in range(num_clusters):
		for j in range((i+1), num_clusters):
			dists = [[euclidean_distance(X[index_1,:], X[index_2,:]) for index_2 in data_indices[j]] for index_1 in data_indices[i]]
			proximity_mat[i, j] = np.amin(dists)
			proximity_mat[j, i] = proximity_mat[i, j]

	print("Now computing Sihoutte coefficients")
	sil_coeffs = np.zeros((1, X.shape[0]))

	# Silgouette score calculation
	for i in range(X.shape[0]):
		# Cluster ID of current data sample
		class_id = int(class_ids[0, i])
		# Mean distance to all data samples in the same cluster
		intra_dist = np.mean(np.array([euclidean_distance(X[i,:], X[j,:]) for j in data_indices[class_id]]))
		# Choose the cluster whose sinngle linkage distance is minimum
		min_index = np.argmin(proximity_mat[class_id, :])
		# Calculate the mean distance to all points in that cluster
		inter_dist = np.mean(np.array([euclidean_distance(X[i,:], X[j,:]) for j in data_indices[min_index]]))
		# Calculate the silhouette score
		sil_coeffs[0,i] = (inter_dist - intra_dist)/(max(intra_dist, inter_dist))

	print("The average Sihoutte coefficient for k = {}, is: {}".format(k, np.mean(np.array(sil_coeffs))))
	# if(num_clusters <= 6):
	scores.append(np.mean(np.array(sil_coeffs)))


k = np.arange(2, X.shape[0]+1)

# To make from [2, X.shape[0]]
scores.reverse()
np.savetxt("data.csv", scores, delimiter = ",")

# plotting
plt.xlabel("Value of K (number of clusters)")
plt.ylabel("Silhoutte Coefficient Score")
plt.title("Variation of Silhoutte scores with K")
plt.plot(k,	scores)
plt.savefig("result_1.jpg")