import numpy as np
import pylab as plt

E = 1
B = 1
qm = 1

def lorentz_acceleration(velocity: np.ndarray) -> np.ndarray:
    acceleration = qm * np.array([velocity[1] * B, E - velocity[0] * B])
    return acceleration

def deriv(state: np.ndarray) -> np.ndarray:
    derivatives = np.concatenate([state[2:], lorentz_acceleration(state[2:])])
    return derivatives

def initialize_parameters(total_time: float = 50, num_time: int = 1_000):
    time_array = np.linspace(0, total_time, num_time)

    z0 = np.array([0, 0], dtype=float) # initial position
    v0 = np.array([0, 0], dtype=float) # initial velocity

    state0 = np.concatenate([z0, v0])

    history0 = np.zeros([len(state0), num_time])
    return state0, time_array, history0

def update_leapfrog(state: np.ndarray, interval: float) -> np.ndarray:
    En = qm * E * np.array([0, 1, 0], dtype=float)
    Bn = qm * B * np.array([0, 0, 1], dtype=float)

    v_old = np.concatenate([state[2:], [0]])
    A = Bn * interval / 2
    C = v_old + En * interval + np.cross(v_old, Bn) * interval / 2
    v_new = (- np.cross(A, C) + A * np.dot(A, C) + C) / (1 + np.dot(A, A))

    state[2:] = v_new[:2] # update velocity
    state[:2] += v_new[:2] * interval # update position

    return state

def evolution(state0, time_array, history, method='forward_euler'):
    # Create copies of the input arrays
    state0 = state0.copy()
    time_array = time_array.copy()
    history = history.copy()

    state = state0
    interval = time_array[1] - time_array[0]
    for tidx in range(len(time_array)):
        if method == 'forward_euler':
            state += deriv(state) * interval
        elif method == 'backwards_euler':
            state_prime = state + deriv(state) * interval
            state += deriv(state_prime) * interval
        elif method == 'midpoint':
            state_prime = state + deriv(state) * interval / 2
            state += (deriv(state) + deriv(state_prime)) * interval / 2
        elif method == 'leapfrog':
            state = update_leapfrog(state, interval)

        history[:, tidx] = state
    return history

state0, time_array, history0 = initialize_parameters(total_time = 20, num_time = 100)

methods = ['forward_euler', 'backwards_euler', 'midpoint', 'leapfrog']
# methods = ['leapfrog']
plt.figure(figsize=(5, 3))
for method in methods:
    history = evolution(state0, time_array, history0, method=method)
    plt.plot(history[0, :], history[1, :], label=method)

plt.axis('equal')
plt.legend()
plt.show()
