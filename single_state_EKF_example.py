# EKF EXAMPLE: SINGLE STATE, NONLINEAR SYSTEM, NO CONTROL INPUTS
# model:
#  x(k+1) = sqrt(5 + x(k)) + w(k)
#  y(k) = x(k)^3 + v(k)


import numpy as np
import matplotlib.pyplot as plt

# Initialize simulation variables
SigmaW = 1  # Process noise covariance
SigmaV = 2  # Sensor noise covariance
maxIter = 40
xtrue = 2 + np.random.randn(1)  # Initialize true system initial state

xhat = 2  # Initialize Kalman filter initial estimate
SigmaX = 1  # Initialize Kalman filter covariance
u = 0  # Unknown initial driving input: assume zero
# Reserve storage for variables we might want to plot/evaluate
xstore = np.zeros((maxIter + 1, len(np.atleast_1d(xtrue))))
xstore[0, :] = xtrue
xhatstore = np.zeros((maxIter, len(np.atleast_1d(xhat))))
SigmaXstore = np.zeros((maxIter, len(np.atleast_1d(xhat)) ** 2))

for k in range(maxIter):
    # EKF Step 0: Compute Ahat, Bhat
    # Note: For this example, x(k+1) = sqrt(5+x(k)) + w(k)
    Ahat = 0.5 / np.sqrt(5 + xhat)
    Bhat = 1

    # EKF Step 1a: State estimate time update
    # Note: You need to insert your system's f(...) equation here
    xhat = np.sqrt(5 + xhat)

    # EKF Step 1b: Error covariance time update
    SigmaX = Ahat * SigmaX * Ahat.T + Bhat * SigmaW * Bhat

    # [Implied operation of system in background, with
    # input signal u, and output signal y]
    w = np.linalg.cholesky(np.atleast_2d(SigmaW)).T @ np.random.randn(1)
    v = np.linalg.cholesky(np.atleast_2d(SigmaV)).T @ np.random.randn(1)
    ytrue = xtrue**3 + v  # y is based on present x and u
    xtrue = np.sqrt(5 + xtrue) + w  # future x is based on present u

    # EKF Step 1c: Estimate system output
    # Note: You need to insert your system's h(...) equation here
    Chat = 3 * xhat**2
    Dhat = 1
    yhat = xhat**3

    # EKF Step 2a: Compute Kalman gain matrix
    SigmaY = Chat * SigmaX * Chat.T + Dhat * SigmaV * Dhat
    L = SigmaX * Chat.T / SigmaY

    # EKF Step 2b: State estimate measurement update
    xhat = xhat + L * (ytrue - yhat)
    xhat = np.maximum(-5, xhat)  # don't get square root of negative xhat!

    # EKF Step 2c: Error covariance measurement update
    SigmaX = SigmaX - L * SigmaY * L.T
    _, S, Vt = np.linalg.svd(np.atleast_2d(SigmaX))
    V = Vt.T
    HH = V @ np.diag(S) @ V.T
    SigmaX = (SigmaX + SigmaX.T + HH + HH.T) / 4  # Help to keep robust

    # [Store information for evaluation/plotting purposes]
    xstore[k + 1, :] = xtrue
    xhatstore[k, :] = np.atleast_1d(xhat)
    SigmaXstore[k, :] = np.atleast_2d(SigmaX).reshape(-1)

plt.figure(1)
plt.clf()
t = np.arange(maxIter)
plt.plot(t, xstore[:maxIter], 'k-', label='true')
plt.plot(t, xhatstore, 'b--', label='estimate')
plt.plot(t, xhatstore + 3 * np.sqrt(SigmaXstore), 'm-.', label='bounds')
plt.plot(t, xhatstore - 3 * np.sqrt(SigmaXstore), 'm-.')
plt.grid(True)
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('State')
plt.title('Extended Kalman filter in action')

plt.figure(2)
plt.clf()
plt.plot(t, xstore[:maxIter] - xhatstore, 'b-', label='Error')
plt.plot(t, 3 * np.sqrt(SigmaXstore), 'm--', label='bounds')
plt.plot(t, -3 * np.sqrt(SigmaXstore), 'm--')
plt.grid(True)
plt.legend()
plt.title('EKF Error with bounds')
plt.xlabel('Iteration')
plt.ylabel('Estimation error')

plt.show()