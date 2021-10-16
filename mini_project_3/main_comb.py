# Ayan Chakraborty 18EC10075
# CS60050 ML MiniProject 3 Code: CS4
# Main Code combining KMeans and Hierarchial Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# returns the euclidean distance between 2 data samples
def euclidean_distance(v1, v2):
	return np.linalg.norm(v1 - v2)

# Needed for sortinng the final list of clusters by the 1st element of each cluster
def mySort(e):
	return e[0]

# read the dataset excluding the 1st coloumn which is the index coloumn
df = pd.read_csv("cricket_4_unlabelled.csv", index_col = 0)

# Z-Score normalization of each coloumn
cols = list(df.columns)
for col in cols:
	df[col] = (df[col] - df[col].mean())/df[col].std(ddof = 0)

# Convert to Numpy Array and Initializationss
X = df.to_numpy()
max_iterations = 20
data_indices_list = []
scores = np.zeros((1, 5))

# k_optimal = 6

# Same way as main_kmeans.py
# Please refer to main_kmeans.py for detailed comments
for k in np.arange(2, 7):
	# random_index = np.random.randint(0, X.shape[0], k_optimal)
	# centroids = [X[i] for i in random_index]
	centroids = []
	centroids.append(X[np.random.randint(0, X.shape[0])])
	for temp in range(k-1):
		probs = []
		cdf_mat = []
		cdf = 0
		for i in range(X.shape[0]):
			dists = [euclidean_distance(X[i], centroid) for centroid in centroids]
			min_dist = dists[np.argmin(np.array(dists))]
			prob_val = min_dist**2
			cdf = cdf + prob_val
			probs.append(prob_val)
			cdf_mat.append(cdf)

		den = np.sum(np.array(probs))
		probs = [prob_val/den for prob_val in probs]
		cdf_mat = [cdf/den for cdf in cdf_mat]

		val = np.random.uniform()
		for i in range(X.shape[0]):
			if cdf_mat[i] > val:
				index = i
				break

		centroids.append(X[index])

	print("Now entering iteration for k = {}".format(k))
	for iteration in range(max_iterations+1):
		actual_data = [[] for temp in range(k)]
		data_indices_kmeans = [[] for temp in range(k)]
		class_ids = np.zeros((1, X.shape[0]))
		for i in range(X.shape[0]):
			dists = [euclidean_distance(X[i], centroid) for centroid in centroids]
			index = np.argmin(np.array(dists))
			actual_data[index].append(X[i])
			data_indices_kmeans[index].append(i)
			class_ids[0, i] = index

		centroids = [np.mean(data, axis = 0) for data in actual_data]

	# Used for storing the clusters for each value of K
	data_indices_list.append(data_indices_kmeans)
	print("Now computing Silhoutte coefficients")
	sil_coeffs = np.zeros((1, X.shape[0]))
	for i in range(X.shape[0]):
		class_id = int(class_ids[0, i])
		intra_dist = np.mean(np.array([euclidean_distance(X[i,:], points) for points in actual_data[class_id]]))
		centroid_dist = [euclidean_distance(X[i,:], centroid) for centroid in centroids if centroid is not centroids[class_id]]
		min_index = np.argmin(np.array(centroid_dist))
		inter_dist = np.mean(np.array([euclidean_distance(X[i,:], points) for points in actual_data[min_index]]))
		sil_coeffs[0,i] = (inter_dist - intra_dist)/(max(intra_dist, inter_dist))

	score = np.mean(np.array(sil_coeffs))
	scores[0, k-2] = score
	print("The average Silhoutte coefficient for k = {} [Using K-Means], is: {}".format(k, score))

# Choose the K having the maximum silhouette score
k_optimal = np.argmax(scores) + 2
optimal_silhouette_score = scores[0, k_optimal - 2]
# Get the clusters for that value of K
data_indices_kmeans = data_indices_list[np.argmax(scores)]
print("The Silhouette scores for K = 2 to 6 [Using KMeans Algorithm] is: {}".format(scores))
print("The optimal value of K = {}, with average Silhouette score = {}".format(k_optimal, optimal_silhouette_score))

# Hierarchial Clustering for optimal value of K (same way as main_hierarchial.py) 
# Please refer to main_hierarchial.py for detailed comments
num_clusters = X.shape[0]
data_indices = [[] for temp in range(num_clusters)]
class_ids = np.zeros((1, X.shape[0]))

for i in range(X.shape[0]):
	data_indices[i].append(i)

print("Now Calculating using Hierarchial Clustering for optimal K = {}".format(k_optimal))
while(num_clusters > k_optimal):

	print("Now at number of clusters = {}".format(num_clusters))
	proximity_mat = np.zeros((num_clusters, num_clusters))
	np.fill_diagonal(proximity_mat, np.inf)

	for i in range(num_clusters):
		for j in range((i+1), num_clusters):
			dists = [[euclidean_distance(X[index_1,:], X[index_2,:]) for index_2 in data_indices[j]] for index_1 in data_indices[i]]
			proximity_mat[i, j] = np.amin(dists)
			proximity_mat[j, i] = proximity_mat[i, j]

	index = np.unravel_index(proximity_mat.argmin(), proximity_mat.shape)
	min_index = min(index[0], index[1])
	max_index = max(index[0], index[1])

	for i in range(num_clusters):
		if i == min_index:
			data_indices[i] = data_indices[min_index] + data_indices[max_index]

	del data_indices[max_index]
	num_clusters = num_clusters - 1

for i in range(num_clusters):
	for index in data_indices[i]:
		class_ids[0, index] = i

# print(len(data_indices))

proximity_mat = np.zeros((num_clusters, num_clusters))
np.fill_diagonal(proximity_mat, np.inf)

for i in range(num_clusters):
	for j in range((i+1), num_clusters):
		dists = [[euclidean_distance(X[index_1,:], X[index_2,:]) for index_2 in data_indices[j]] for index_1 in data_indices[i]]
		proximity_mat[i, j] = np.amin(dists)
		proximity_mat[j, i] = proximity_mat[i, j]

print("Now computing Sihoutte coefficients")
sil_coeffs = np.zeros((1, X.shape[0]))

for i in range(X.shape[0]):
	class_id = int(class_ids[0, i])
	intra_dist = np.mean(np.array([euclidean_distance(X[i,:], X[j,:]) for j in data_indices[class_id]]))
	min_index = np.argmin(proximity_mat[class_id, :])
	inter_dist = np.mean(np.array([euclidean_distance(X[i,:], X[j,:]) for j in data_indices[min_index]]))
	sil_coeffs[0,i] = (inter_dist - intra_dist)/(max(intra_dist, inter_dist))

print("The average Sihoutte coefficient for k = {} [Using Hierarchial Clustering], is: {}".format(k_optimal, np.mean(np.array(sil_coeffs))))

# Initializing the Jaccard Matrix
jaccard_mat = np.zeros((k_optimal, k_optimal))

# Sorting the clusters for storing as text file
for i in range(k_optimal):
	# Sort each cluster in increasing order
	data_indices[i].sort()
	data_indices_kmeans[i].sort()

# Sort the total list by the 1st element of each cluster
data_indices.sort(key = mySort)
data_indices_kmeans.sort(key = mySort)

# print(data_indices)
# print(data_indices_kmeans)

# Write the clusters to the text file
with open("kmeans.txt", 'w') as output:
    for row in data_indices_kmeans:
        output.write(','.join(str(i) for i in row))
        output.write('\n')

with open("agglomerative.txt", 'w') as output:
    for row in data_indices:
        output.write(','.join(str(i) for i in row))
        output.write('\n')

# Calculate the Jaccard Similarity matrix
for i in range(k_optimal):
	A = data_indices_kmeans[i]
	
	for j in range(k_optimal):
		# print("Class ID: {} for K_Means".format(i))
		# print("Class ID: {} for Hierarchial".format(j))

		B = data_indices[j]
		# Calculate the intersection elements
		common = [value for value in A if value in B]

		# Use the lengths of each cluster and intersectuon to calculate Jaccard Similarity
		jaccard_index = len(common)/(len(A) + len(B) - len(common))
		jaccard_mat[i, j] = jaccard_index
		# print(len(A))
		# print(len(B))

print("The Jaccard Matrix is:")
print(jaccard_mat)