from surprise import SVD, SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.cross_validation import train_test_split


# Load the movielens-100k dataset.
data = Dataset.load_builtin('ml-100k')

# SVD algorithm.
algo = SVD()
# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)# SVD++ - Singular Value Decomposition with Implicit Ratings
algo = SVD++() 
# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
algo = NMF()
# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

