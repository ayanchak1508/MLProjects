# Ayan Chakraborty 18EC10075
# CS60050 ML MiniProject 3 Code: CS4
# Code for KMeans Algo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# returns the cosine similarity value for 2 data samples
def cosine_similarity(v1, v2):
	return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

# returns the euclidean distance between 2 data samples
def euclidean_distance(v1, v2):
	return np.linalg.norm(v1 - v2)

# read the dataset excluding the 1st coloumn which is the index coloumn
df = pd.read_csv("cricket_4_unlabelled.csv", index_col = 0)
cols = list(df.columns)

# Z-Score normalization of each coloumn
for col in cols:
	df[col] = (df[col] - df[col].mean())/df[col].std(ddof = 0)

# Convert the dataset into numpy array
X = df.to_numpy()

# Number of iterations for each value of K
max_iterations = 20

# Number of times the KMeans Calculation will be repeated
avg_iterations = 10

# To store all the silhouette scores
scores = np.zeros((avg_iterations, 5))

# Main KMeans Algo
for avg_iter in range(avg_iterations):
	for k in [2, 3, 4, 5, 6]:

		# # random initialization of centroids
		# random_index = np.random.randint(0, X.shape[0], k)
		# centroids = [X[i] for i in random_index]	
		
		# K++ initialization of the centroids
		centroids = []

		# Choose the first centroid randomly
		centroids.append(X[np.random.randint(0, X.shape[0])])
		
		# Choose the remaining K-1 centroids
		for temp in range(k-1):
			probs = []
			cdf_mat = []
			cdf = 0

			# Iterate over all data samples
			for i in range(X.shape[0]):
				
				# Compute the distance the data sample to all of the previosuly generated centroids
				dists = [euclidean_distance(X[i], centroid) for centroid in centroids]
				
				# Calculayte the minimum distance
				min_dist = dists[np.argmin(np.array(dists))]
				
				# Probability and CDF calculation
				prob_val = min_dist**2
				cdf = cdf + prob_val
				probs.append(prob_val)
				cdf_mat.append(cdf)

			# Normalization of the probabilities and the CDF
			den = np.sum(np.array(probs))
			probs = [prob_val/den for prob_val in probs]
			cdf_mat = [cdf/den for cdf in cdf_mat]

			# Choose the next centroid whose CDF value is just greater than a random number between 0-1
			val = np.random.uniform()
			for i in range(X.shape[0]):
				if cdf_mat[i] > val:
					index = i
					break

			# Update the centroid list
			centroids.append(X[index])

		print("Now entering iteration for k = {}".format(k))
		
		# Calculating the K clusters
		for iteration in range(max_iterations+1):
			
			# Used for storing the clusters
			actual_data = [[] for temp in range(k)]
			class_ids = np.zeros((1, X.shape[0]))
			
			# Iterate over all data samples
			for i in range(X.shape[0]):
				# Calculate the distance from all centroids
				dists = [euclidean_distance(X[i], centroid) for centroid in centroids]
				# Choose the cluster with the minimum distance
				index = np.argmin(np.array(dists))
				# Append the data sample to that cluster
				actual_data[index].append(X[i])
				# Update the cluster ID of thqt data sample
				class_ids[0, i] = index

			# Update the centroid of each cluster by taking the mean of all samples in a cluster
			centroids = [np.mean(data, axis = 0) for data in actual_data]

		print("Now computing Sihoutte coefficients")
		sil_coeffs = np.zeros((1, X.shape[0]))
		for i in range(X.shape[0]):
			
			# Find which cluster the data sample belongs to
			class_id = int(class_ids[0, i])
			# Calculate the distance to all points in the same cluster
			intra_dist = np.mean(np.array([euclidean_distance(X[i,:], points) for points in actual_data[class_id]]))
			# Calculatye the distance of that point to all other centroids (except own centroid)
			centroid_dist = [euclidean_distance(X[i,:], centroid) for centroid in centroids if centroid is not centroids[class_id]]
			# Choose the closest centroid
			min_index = np.argmin(np.array(centroid_dist))
			# Calculate the mean distance to all points in that cluster
			inter_dist = np.mean(np.array([euclidean_distance(X[i,:], points) for points in actual_data[min_index]]))
			# Calculate the Silhouette coefficient
			sil_coeffs[0,i] = (inter_dist - intra_dist)/(max(intra_dist, inter_dist))

		score = np.mean(np.array(sil_coeffs))
		print("The average Silhoutte coefficient for k = {}, is: {}, at the {}th iteration".format(k, score, (avg_iter+1)))
		scores[avg_iter, k-2] = score

# Calculate the average Silhouette score across all runs
mean_scores = np.mean(scores, axis = 0)
# Choose the optimal K having highest silhouette score
optimal_k = np.argmax(mean_scores) + 2

k = np.arange(2, 7)

# Printing the necessary infomation
print("The mean scores across different values of K (from 2 to 6):")
print(mean_scores)

print("The optimial value of K is:{}".format(optimal_k))

# Plotting
plt.xlabel("Value of K (number of clusters)")
plt.ylabel("Average Silhoutte Coefficient Score across 10 runs")
plt.title("Variation of Silhoutte scores with K")
plt.plot(k,	mean_scores)
plt.savefig("result_1.jpg")