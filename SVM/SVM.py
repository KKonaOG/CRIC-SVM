import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from scipy.stats import norm

def SVM_model(C_val, gamma_val, X_train, y_train, X_test, y_test):
    # Train a multi-class SVM classifier
    # Possible changes: kernel (separate from Gp kernel), C parameter for regularization, gamma for , cache_size for speed that cost RAM
    # for GP, focus on C and gamma
    svm_model = SVC(kernel='rbf', C=C_val, gamma=gamma_val, decision_function_shape='ovo')
    svm_model.fit(X_train, y_train)

    # Create a mesh to plot the decision boundaries
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Get results
    y_predict = svm_model.predict(X_test)
    precision = precision_score(y_true=y_test, y_pred=y_predict, average='macro')
    accuracy = accuracy_score(y_true=y_test, y_pred=y_predict)
    recall = recall_score(y_true=y_test, y_pred=y_predict, average='macro')
    f1 = f1_score(y_true=y_test, y_pred=y_predict, average='macro')
    print("\nC: {}\nGamma: {}\nAccuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}".format(C_val, gamma_val, accuracy, recall, precision, f1))

    encoder = LabelEncoder()
    Z_encoded = encoder.fit_transform(Z.ravel())  # Flatten Z to fit
    Z_numeric = Z_encoded.reshape(Z.shape)  # Reshape back to the original shape of Z
    y_train_encoded = encoder.fit_transform(y_train)

    #Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true=y_test, y_pred=y_predict)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
    plt.savefig("confusion_matrix/confusion_matrix_C_" + str(C_val) + "_gamma_" + str(gamma_val) + ".png")

    # Plot the decision boundaries
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_numeric, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_encoded, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Avg Intensity')
    plt.ylabel('Total Size')
    plt.title('SVM Multi-Class Classification')
    plt.colorbar()
    plt.savefig("SVM/SVM_C_" + str(C_val) + "_gamma_" + str(gamma_val) + ".png")
    #plt.show()

    return precision, accuracy, recall, f1

# Determines the expected improvement across all values of X1 and X2
# relative to the best observed y value
def expected_improvement(X, gp_model, best_y):
    y_pred, y_std = gp_model.predict(X, return_std=True)
    z = (y_pred - best_y) / y_std
    ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei

# Make dataset
kradar_dataset = pd.read_csv('kradar_processed.csv')
kradar_dataset = kradar_dataset.drop(columns=['Dataset', 'Frame Number', 'Max Intensity', 'Min Intensity', 'X Size', 'Y Size'])

X_train, X_test, y_train, y_test = train_test_split(kradar_dataset[['Avg Intensity', 'Total Size']], kradar_dataset['Label'], test_size=0.3, random_state=6693)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bayesian optimization of hyperparameters
# C_axis and gamma_axis are all possible values that the two parameters can each take on
C_range = np.linspace(start=1e-3, stop=20, num=2000).reshape(-1, 1)#np.logspace(-2, 10, 13).reshape(-1, 1)
gamma_range = np.linspace(start=1e-3, stop=20, num=2000).reshape(-1, 1)#np.logspace(-9, 3, 13).reshape(-1, 1)
C, gamma = np.meshgrid(C_range, gamma_range)

# Observes an initial number of samples of the unknown objective function
# for hyperparameter performance
rng = np.random.RandomState(4)

# C and X2 are combined in a single matrix to be able to be inputted into GP
parameters_range = np.column_stack((C.reshape(-1), gamma.reshape(-1)))
training_indices = rng.choice(np.arange(len(parameters_range)), size=10, replace=False)
parameters_samples = parameters_range[training_indices]

# Obtains the kernel for the GP regression model
kernel = 1 * RBF(length_scale=2.0, length_scale_bounds=(1e-5, 1e5)) * DotProduct(sigma_0=2.0, sigma_0_bounds=(1e-5, 1e5))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Obtains accuracy of initial samples
precision_samples = []
accuracy_samples = []
recall_samples = []
f1_samples = []
for param in parameters_samples:
    C_sample = param[0]
    gamma_sample = param[1]
    # Black Box Function
    precision, accuracy, recall, f1 = SVM_model(C_sample, gamma_sample, X_train_scaled, y_train, X_test_scaled, y_test)
    precision_samples = np.append(precision_samples, precision)
    accuracy_samples = np.append(accuracy_samples, accuracy)
    recall_samples = np.append(recall_samples, recall)
    f1_samples = np.append(f1_samples, f1)

iterations_since_last_improvement = 0
while iterations_since_last_improvement < 10:

    # Fit the GP model to the samples
    parameters_samples = parameters_samples.reshape(-1,2)
    gaussian_process.fit(parameters_samples, f1_samples)

    # Obtains the mean of all surrogate function predictions and the
    # standard deviation, aka uncertainty
    f1_mean, f1_std = gaussian_process.predict(parameters_range, return_std=True)
    f1_pred = f1_mean.reshape(C.shape)

    # Determine the best point with the highest observed function value
    best_id_params = np.argmax(f1_samples)
    best_params = parameters_samples[best_id_params,:]
    best_f1 = f1_samples[best_id_params]

    # Calculates the EI of all potential parameter X1 and X2 value combinations
    exp_imp = expected_improvement(parameters_range, gaussian_process, best_f1)
    ei = exp_imp.reshape(C.shape)

    #if i < num_iterations - 1:
    # Select the next point with the highest EI
    new_params = parameters_range[np.argmax(exp_imp)]
    new_C = new_params[0]
    new_gamma = new_params[1]
    new_precision, new_accuracy, new_recall, new_f1 = SVM_model(new_C, new_gamma, X_train_scaled, y_train, X_test_scaled, y_test)
    parameters_samples = np.vstack([parameters_samples, new_params])
    precision_samples = np.append(precision_samples, new_precision)
    accuracy_samples = np.append(accuracy_samples, new_accuracy)
    recall_samples = np.append(recall_samples, new_recall)
    f1_samples = np.append(f1_samples, new_f1)

    if new_f1 > best_f1:
        iterations_since_last_improvement = 0
    else:
        iterations_since_last_improvement = iterations_since_last_improvement + 1

# Prints all observed samples
print("All Observed Samples: ")
for i in range(len(f1_samples)):
    print(parameters_samples[i], f1_samples[i])

# Selects best observed parameters
best_id_params = np.argmax(f1_samples)
best_params = parameters_samples[best_id_params,:]
best_f1 = f1_samples[best_id_params]
print("Best Observed Parameters:\nC:" + str(best_params[0]) + ", gamma: " + str(best_params[1]) + ", with F1 Score of " + str(best_f1))

# Creating figure
fig = plt.figure(figsize = (10, 8))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)

# Creating color map
my_cmap = plt.get_cmap('hsv')

# Creating plot
C_samples = parameters_samples[:,0]
gamma_samples = parameters_samples[:,1]
sctt = ax.scatter3D(C_samples, gamma_samples, f1_samples,
                    alpha = 0.8,
                    c = (C_samples + gamma_samples + f1_samples),
                    cmap = my_cmap,
                    marker ='^')
plt.title("Hyperparameters Tuning Effects on F1 Score")
ax.set_xlabel('C', fontweight ='bold')
ax.set_ylabel('gamma', fontweight ='bold')
ax.set_zlabel('F1 score', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
plt.savefig("Optimization.png")
