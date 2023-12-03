import random
from env import Env
import numpy as np
import matplotlib.pyplot as plt

def viterbi(observations: list[list[int]], epsilon: float, env) -> np.ndarray:
    """
    Params: 
    observations: a list of observations of size (T, 4) where T is the number of observations and
    1. observations[t][0] is the reading of the left sensor at timestep t
    2. observations[t][1] is the reading of the right sensor at timestep t
    3. observations[t][2] is the reading of the up sensor at timestep t
    4. observations[t][3] is the reading of the down sensor at timestep t
    epsilon: the probability of a single sensor failing

    Return: a list of predictions for the agent's true hidden states.
    The expected output is a numpy array of shape (T, 2) where 
    1. (predictions[t][0], predictions[t][1]) is the prediction for the state at timestep t
    """
    # TODO: implement the viterbi algorithm
    # pass
    # Each t-slice is a 2d array of 3-tuple (V_t, i_prev, j_prev)
    # (i_prev, j_prev) points to the state in time-slice t-1 that yields max V_t

    rows = env.rows
    cols = env.columns
    time_slices = []

    # t=0 time slice
    # prior_prob = 1.0 / (rows * cols)
    time_slice = [ [(1, i, j) for j in range(cols)] for i in range(rows)]
    time_slices.append(time_slice)

    for obs in observations[:]:
        prev_slice = time_slices[-1]
        curr_slice = [ [(0, 0, 0) for j in range(cols)] for i in range(rows)]
        for i in range(rows):
            for j in range(cols):
                neighbors = env.get_neighbors(i,j)
                num_disp = abs(len(neighbors) + int(sum(obs)) - 4)
                obs_prob = pow(epsilon, num_disp) * pow((1 - epsilon), 4 - num_disp)
                # print(f"{(i, j)} nbhr: {neighbors} obs: {obs}  obs_prob: {obs_prob}")
                # now we get the possible t-1 states that can get to (i,j) at time t
                max_V = None
                for nn in neighbors:
                    trans_prob = 1.0 / len(env.get_neighbors(nn[0], nn[1]))
                    prev_V = prev_slice[nn[0]][nn[1]][0]
                    curr_V = prev_V * trans_prob * obs_prob
                    if max_V is None or curr_V > max_V[0]:
                        max_V = (curr_V, nn[0], nn[1])
                curr_slice[i][j] = max_V
        time_slices.append(curr_slice)

    last_time_slice = time_slices[-1]
    max_V = 0
    pred = (0, 0)
    for i in range(rows):
        for j in range(cols):
            if last_time_slice[i][j][0] > max_V:
                max_V = last_time_slice[i][j][0]
                pred = (i, j)
    # print(f"last time slice: {last_time_slice}  pred: {pred}")
    predictions = [pred]
    for ts in time_slices[-1:1:-1]:
        print(f"time slice: {ts}  pred: {pred}")
        _, pred_i, pred_j = ts[pred[0]][pred[1]]
        pred = (pred_i, pred_j)
        print(f"  new pred: {pred}")
        predictions.insert(0, pred)
        # print(f"predictions: {predictions}")
    
    print(f"predictions: \n{predictions}")
    return predictions


if __name__ == '__main__':
    random.seed(12345)
    rows, cols = 16, 16 # dimensions of the environment
    openness = 0.3 # some hyperparameter defining how "open" an environment is
    traj_len = 100 # number of observations to collect, i.e., number of times to call env.step()
    num_traj = 100 # number of trajectories to run per epsilon

    env = Env(rows, cols, openness)
    env.plot_env() # the environment layout should be saved to env_layout.png

    plt.clf()
    """
    The following loop simulates num_traj trajectories for each value of epsilon.
    Since there are 6 values of epsilon being tried here, a total of 6 * num_traj
    trajectories are generated.
    
    For reference, the staff solution takes < 3 minutes to run.
    """
    for epsilon in [0.0, 0.05, 0.1, 0.2, 0.25, 0.5]:
        env.set_epsilon(epsilon)
        
        accuracies = []
        for _ in range(num_traj):
            env.init_env()

            observations = []
            for i in range(traj_len):
                obs = env.step()
                observations.append(obs)

            predictions = viterbi(observations, epsilon, env)

            accuracies.append(env.compute_accuracy(predictions))
        plt.plot(np.mean(accuracies, axis=0), label=f"epsilon={epsilon}")

    plt.xlabel("Number of observations")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("accuracies.png")