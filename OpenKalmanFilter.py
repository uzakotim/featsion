import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Initialize state and covariance matrices
        self.state = np.zeros((state_dim, 1))
        self.covariance = np.eye(state_dim)

    def predict(self, dt):
        # State transition matrix
        F = np.eye(self.state_dim)
        for i in range(0, self.state_dim//2):
            F[i, i+self.state_dim//2] = dt

        # Process noise covariance matrix
        Q = np.zeros((self.state_dim, self.state_dim))
        for i in range(0, self.state_dim//2):
            Q[i, i] = dt**2 / 4
            Q[i+self.state_dim//2, i+self.state_dim//2] = dt

        # Perform prediction
        self.state = np.dot(F, self.state)
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

    def correct(self, measurement, measurement_noise):
        # Measurement matrix
        H = np.eye(self.measurement_dim, self.state_dim)

        # Measurement noise covariance matrix
        R = np.eye(self.measurement_dim) * measurement_noise**2

        # Kalman gain
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(np.dot(np.dot(H, self.covariance), H.T) + R))

        # Update state and covariance
        self.state = self.state + np.dot(K, (measurement - np.dot(H, self.state)))
        self.covariance = np.dot((np.eye(self.state_dim) - np.dot(K, H)), self.covariance)