import numpy as np
import matplotlib.pyplot as plt

def kalman_xy(state, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):
    """
    Parameters:
    state: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise
    motion: external motion added to state vector state
    Q: motion noise (same shape as P)
    """
    return kalman(state, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))

def kalman(state, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    state: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*state)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector state
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*state
    H: measurement function: position = H*state

    Return: the updated and predicted new values for (state, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H
    '''
    # UPDATE state, P based on measurement m
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * state
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    state = state + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT state, P based on motion
    state = F*state + motion
    P = F*P*F.T + Q

    return state, P

def demo_kalman_xy():
    state = np.matrix('0. 0. 0. 0.').T
    P = np.matrix(np.eye(4))*1000 # initial uncertainty

    N = 20
    true_x = np.linspace(0.0, 10.0, N)
    true_y = true_x**2
    observed_x = true_x + 0.05*np.random.random(N)*true_x
    observed_y = true_y + 0.05*np.random.random(N)*true_y
    plt.plot(observed_x, observed_y, 'ro')
    result = []
    R = 0.01**2
    for meas in zip(observed_x, observed_y):
        print(meas)
        state, P = kalman_xy(state, P, meas, R)
        print(state)
        print("-----")
        print((state[:2]).tolist()[0][0])
        result.append((state[:2]).tolist())
    kalman_x, kalman_y = zip(*result)
    print("kalman_x: {}".format(kalman_x))
    print("kalman_y: {}".format(kalman_y))
    plt.plot(kalman_x, kalman_y, 'g-')
    plt.show()

if __name__ == '__main__':
    demo_kalman_xy()