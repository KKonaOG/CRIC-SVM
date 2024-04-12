import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd

kradar_dataset = pd.read_csv('/mnt/KRadar/dataset_simplified/radar_quantified/kradar_processed.csv')
kradar_dataset = kradar_dataset.drop(columns=['Dataset', 'Frame Number', 'Max Intensity', 'Min Intensity', 'X Size', 'Y Size'])

X_train, X_test, y_train, y_test = train_test_split(kradar_dataset[['Avg Intensity', 'Total Size']], kradar_dataset['Label'], test_size=0.3)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a multi-class SVM classifier
# Possible changes: kernel (separate from Gp kernel), C parameter for regularization, gamma for , cache_size for speed that cost RAM
# for GP, focus on C and gamma

svm_model = SVC(kernel='rbf', C=400, gamma = 'scale', decision_function_shape='ovo')
svm_model.fit(X_train_scaled, y_train)

# # Create a mesh to plot the decision boundaries
# x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
# y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                      np.arange(y_min, y_max, 0.02))

# Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot the decision boundaries
# plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
# plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('SVM Multi-Class Classification')
# plt.show()

accuracy = svm_model.score(X_test_scaled, y_test)
print("Accuracy: " + str(accuracy))


