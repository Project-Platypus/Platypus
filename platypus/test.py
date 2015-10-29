from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import time
start = time.time()
X = np.random.rand(100, 10000)
print X.shape
transformer = GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print X_new.shape
print time.time()-start