'''
In case of LDA, rather than thus just finding axes(eigen vectors) that maximise the variance
of our data, we are additionally interested in the axes that maximise separation between multiple classes.

So the goal of LDA is to project a feature space(n- dimensional dataset) onto a smaller subspace 'k',
while maintaining the class discriminatory information.

Comparisons show that PCA outperforms LDA if the number of samples per class is relatively small.

LDA is supervised learning, while PCA is simple unsupervised learning technique.

Eigen values that have a similar magnitude are indicative of a good feature space, while the ones close to 0
are less informative. Eigenvalues are nothing but the magnitude associated with the eigen vectors.

Both Eigen values and Eigen Vectors are providing us with information about the distortion
of linear transformation. The Eigen Vectors are basically directions of this distortion and
the Eigen values are the scaling factor for the eigen vectors, describing the magnitude of distortion.
'''

#Data pre-processing step

feature_dict = {i:label for i,label in zip(
            range(4),
              ('sepal length in cm',
              'sepal width in cm',
              'petal length in cm',
              'petal width in cm', ))}


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end


#Now we want to label encode the classes in the dataset



X = df[[0,1,2,3]].values #Values of the first 4 columns

y = df['class label'].values #Values of the last column, that is class labels


enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

#print "Y", y

'''
Computing 4-dimensional mean vectors:-
'''
#np.set_printoptions(precision=4) is used to set precision in python

mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))


'''
Computing the within class covariance matrix
'''

S_W = np.zeros((4,4))
for cl,mv in zip(range(1,4), mean_vectors):
    class_sc_mat = np.zeros((4,4))                  # Covariance matrix for every class
    for row in X[y == cl]:
        row, mv = row.reshape(4,1), mv.reshape(4,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class Covariance matrices
print('within-class Co-variance Matrix:\n', S_W)
'''
Computing the betweem class covariance matrix
'''

overall_mean = np.mean(X, axis=0)

S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(4,1) # make column vector
    overall_mean = overall_mean.reshape(4,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('Between-class Co-variance Matrix:\n', S_B)


'''
Generating Eigenvalues and EigenVectors
'''

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))


'''
Sorting the eigen vectors
'''

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

'''
Selecting the eigen vectors with the largest 'k eigen values
'''

W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n', W.real)

'''
Transforming our data into a k-dimensional feature space
'''

X_lda = X.dot(W)

from matplotlib import pyplot as plt

def plot_step_lda():

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:,0][y == label],
                y=X_lda[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()


plot_step_lda()