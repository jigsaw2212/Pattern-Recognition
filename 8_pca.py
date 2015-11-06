'''
Our desired outcome through PCA is to reduce the dimension of the dataset
with minimal loss of information.  This can be used to reduce the computational 
cost by reducing the number of dimensions of our feature space by 
extracting a subspace that describes our data best.

PCA ignores the class labels. 
 In PCA, we are interested to find the directions (components) that maximize 
 the variance in our dataset, where in MDA, we are additionally interested to
 find the directions that maximize the separation (or discrimination) between different classes

 Or, roughly speaking in PCA we are trying to find the axes with maximum variances 
 where the data is most spread (within a class, since PCA treats the whole data set 
 as one class)
'''


'''
Creating two 3X20 datasets - one dataset for each class w1 and w2
'''
import numpy as np

np.random.seed(1) # random seed for consistency

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

'''
Igniring class labels
'''

all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (3,40), "The matrix has not the dimensions 3x40"

'''
Computing d-dimesional mean vector
'''

mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print('Mean Vector:\n', mean_vector)

'''
Compute the covariance matrix
'''

cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
print('Covariance Matrix:\n', cov_mat)


'''
Compute the eigen values and eigen vectors
'''

scatter_matrix = np.zeros((3,3))
for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot(
        (all_samples[:,i].reshape(3,1) - mean_vector).T)
#print('Scatter Matrix:\n', scatter_matrix)

# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    #print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    #print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    #print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
    print(40 * '-')


'''
Sort eigen vectors by the decreasing eigen values
'''

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
             for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()


'''
Choosing k eigen vectors with largest eigen values
'''

matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),
                      eig_pairs[1][1].reshape(3,1)))
print('Matrix W :', matrix_w)


'''
Transform the samples to get the new subspace
'''

transformed = matrix_w.T.dot(all_samples)

'''
Plotting the transformed samples
'''

from matplotlib import pyplot as plt

plt.plot(transformed[0,0:20], transformed[1,0:20],
         'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40],
         '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()