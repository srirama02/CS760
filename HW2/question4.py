import numpy as np
from scipy.interpolate import lagrange

a, b = 0, 2 * np.pi 
n = 100


x_train = np.linspace(a, b, n)
y_train = np.sin(x_train)

lagrange_poly = lagrange(x_train, y_train)

x_test = np.linspace(a, b, n)
y_test = np.sin(x_test)

train_predictions = lagrange_poly(x_train)
test_predictions = lagrange_poly(x_test)

train_mse = np.mean((y_train - train_predictions) ** 2)
test_mse = np.mean((y_test - test_predictions) ** 2)

train_log_mse = np.log(train_mse)
test_log_mse = np.log(test_mse)

print("Train Log MSE:", train_log_mse)
print("Test Log MSE:", test_log_mse)

print()
print("zero-mean guassian noise")
print()

a, b = 0, 2 * np.pi
n = 100
epsilon_stddevs = [0.01, 0.1, 0.5, 1.0]

for stddev in epsilon_stddevs:
    x_train_noisy = x_train + np.random.normal(0, stddev, n)
    y_train_noisy = np.sin(x_train_noisy)

    lagrange_poly = lagrange(x_train_noisy, y_train_noisy)

    x_test = np.linspace(a, b, n)
    y_test = np.sin(x_test)

    train_predictions = lagrange_poly(x_train_noisy)
    test_predictions = lagrange_poly(x_test)

    train_mse = np.mean((y_train_noisy - train_predictions) ** 2)
    test_mse = np.mean((y_test - test_predictions) ** 2)

    train_log_mse = np.log(train_mse)
    test_log_mse = np.log(test_mse)

    print(f"Train Log MSE (with noise stddev {stddev}):", train_log_mse)
    print(f"Test Log MSE (with noise stddev {stddev}):", test_log_mse)