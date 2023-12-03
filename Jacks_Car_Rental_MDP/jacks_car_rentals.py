import numpy as np
from scipy.stats import poisson
from itertools import product
from functools import cache
import matplotlib.pyplot as plt

MAX_CARS_LOC_1 = 20 # maximum number of cars at location 1
MAX_CARS_LOC_2 = 20 # maximum number of cars at location 2
MAX_CARS_MOVED = 5 # maximum number of cars that can be moved in one night
FREE_MOVES = 1 # number of free moves from location 1 to 2 that Jack's employee can make
STORAGE_CAPACITY = 10 # maximum number of cars that can be stored at each location overnight for free

RENTAL_REWARD = 10  # reward for renting out a car
MOVE_COST = 2 # cost for moving a car overnight
STORAGE_COST = 4 # cost incurred for the second parking lot at a given location

LOC_1_RENTAL_LAMBDA = 3
LOC_1_RETURN_LAMBDA = 3
LOC_2_RENTAL_LAMBDA = 4
LOC_2_RETURN_LAMBDA = 2

GAMMA = 0.9 # discount factor

THRESHOLD = 1e-2 # the convergence threshold used for each of the algorithms above
# You may want to set the threshold to something high when debugging since it will allow your algorithm to terminate early.

@cache
def pmf(k: int, lam: float):
    """
    Computes P(X = k) where X ~ Poisson(lam).
    """
    return poisson.pmf(k, lam)

@cache
def sf(k: int, lam: float):
    """
    Computes P(X > k) where X ~ Poisson(lam).
    """
    return poisson.sf(k, lam)

@cache
def cdf(k: int, lam: float):
    """
    Computes P(X <= k) where X ~ Poisson(lam).
    """
    return poisson.cdf(k, lam)

@cache
def expectation(lam: float, end: int):
    """
    Suppose X ~ Poisson(lam).
    This function computes E[max(X, end)].
    """
    return sum(k * pmf(k, lam) for k in range(end + 1)) + (1 - cdf(end, lam)) * end

@cache
def transition_prob(state: tuple[int], action: int, next_state: tuple[int]) -> float:
    """
    Computes p(next_state | state, action), i.e., the probability of reaching 
    (next_state[0], next_state[1]) cars at location 1 and 2, respectively, at the end of the 
    next day given that we start at (state[0], state[1]) cars at the end of the current day and 
    move `action` cars overnight.

    Fill in the TODOs. You may find the functions above very helpful.

    Note: this function assumes that `state` and `next_state` are tuples instead of numpy arrays.
    This is because we are caching this function and numpy arrays are not hashable!
    """
    global LOC_1_RENTAL_LAMBDA, LOC_1_RETURN_LAMBDA, LOC_2_RENTAL_LAMBDA, LOC_2_RETURN_LAMBDA

    # Cars at each location at the end of the current day
    loc_1_curr, loc_2_curr = state
    # Cars at each location assuming `action` cars will be moved
    loc_1, loc_2 = loc_1_curr - action, loc_2_curr + action
    # Cars we want at each location at the end of the next day
    loc_1_new, loc_2_new = next_state

    if loc_1 < 0 or loc_2 < 0:
        return 0
    
    # Handle the probability of going from `loc_1`` to `loc_1_new`` first
    prob_loc_1 = 0

    # Split into 2 cases:
    # Case 1: we can only rent up to `loc_1`` cars the next day
    # To reach `loc_1_new`` cars by the end of the day, how many cars do we need to 
    # rent out at the bare minimum?
    min_cars_to_rent = max(loc_1 - loc_1_new, 0) # TODO
    for rented in range(min_cars_to_rent, loc_1 + 1):
        # Now, find out the number of cars we want to see returned to reach `loc_1_new`, 
        # given that we rented out `rented` cars:
        returned = loc_1_new - loc_1 + rented # TODO
        # We again have two cases:
        if loc_1_new < MAX_CARS_LOC_1:
            # Case 1.1: loc_1_new < MAX_CARS_LOC_1
            # Here, we need exactly `returned` cars to be returned. 
            # Compute the probability of seeing exactly `returned` cars returned at location 1.
            prob_returned = pmf(returned, LOC_1_RETURN_LAMBDA) # TODO
        else:
            # Case 1.2: loc_1_new == MAX_CARS_LOC_1
            # Here, we need at least `returned` cars to be returned.
            # The excess cars are removed from the system, as mentioned in the problem.
            # Compute the probability of seeing at least `returned` cars returned at location 1.
            prob_returned = sf(returned-1, LOC_1_RETURN_LAMBDA) # TODO
        # Compute the probability of seeing exactly `rented` cars being rented out.
        prob_rented = pmf(rented, LOC_1_RENTAL_LAMBDA) # TODO
        # Combine the two probabilities above. Note that the rentals and returns are independent.
        prob_loc_1 += prob_rented * prob_returned # TODO
    # Case 2: we get more rental requests than we have cars available at location 1.
    # Of course, we can only rent out `loc_1`` cars before running out but that doesn't affect
    # the number of requests that come in. 
    # Since we are renting out all of the cars available at location 1, we would need exactly 
    # `loc_1_new`` cars to be returned. 
    # Compute the probability of this event (more than `loc_1` rental requests and exactly `loc_1_new`
    # returns) happening.
    prob_loc_1 += sf(loc_1, LOC_1_RENTAL_LAMBDA) * pmf(loc_1_new, LOC_1_RETURN_LAMBDA) # TODO

    # We repeat the same process for location 2
    prob_loc_2 = 0

    # Split into 2 cases:
    # Case 1: we can only rent up to `loc_2`` cars the next day
    # To reach `loc_2_new`` cars by the end of the day, how many cars do we need to 
    # rent out at the bare minimum?
    min_cars_to_rent = max(loc_2 - loc_2_new, 0) # TODO
    for rented in range(min_cars_to_rent, loc_2 + 1):
        # Now, find out the number of cars we want to see returned to reach `loc_2_new`, 
        # given that we rented out `rented` cars:
        returned = loc_2_new - loc_2 + rented # TODO
        # We again have two cases:
        if loc_2_new < MAX_CARS_LOC_2:
            # Case 1.1: loc_2_new < MAX_CARS_LOC_2
            # Here, we need exactly `returned` cars to be returned. 
            # Compute the probability of seeing exactly `returned` cars returned at location 2.
            prob_returned = pmf(returned, LOC_2_RETURN_LAMBDA) # TODO
        else:
            # Case 1.2: loc_2_new == MAX_CARS_LOC_2
            # Here, we need at least `returned` cars to be returned.
            # The excess cars are removed from the system, as mentioned in the problem.
            # Compute the probability of seeing at least `returned` cars returned at location 2.
            prob_returned = sf(returned-1, LOC_2_RETURN_LAMBDA) # TODO
        # Compute the probability of seeing exactly `rented` cars being rented out.
        prob_rented = pmf(rented, LOC_2_RENTAL_LAMBDA) # TODO
        # Combine the two probabilities above. Note that the rentals and returns are independent.
        prob_loc_2 += prob_rented * prob_returned # TODO
    # Case 2: we get more rental requests than we have cars available at location 2.
    # Of course, we can only rent out `loc_2`` cars before running out but that doesn't affect
    # the number of requests that come in. 
    # Since we are renting out all of the cars available at location 2, we would need exactly 
    # `loc_2_new`` cars to be returned. 
    # Compute the probability of this event (more than `loc_2` rental requests and exactly `loc_2_new`
    # returns) happening.
    prob_loc_2 += sf(loc_2, LOC_2_RENTAL_LAMBDA) * pmf(loc_2_new, LOC_2_RETURN_LAMBDA) # TODO

    # We take the product of the two probabilities since each locations operates independently of the other.
    return prob_loc_1 * prob_loc_2

@cache
def expected_rewards(state: tuple[int], action: int, add_non_linearity: bool) -> float:
    """
    Helper function to compute E[r | state, action].
    Implement this function and use it as a subroutine for your q-function computation.

    `state`: a 2-tuple (loc_1_cars, loc_2_cars).
    `action`: a number between -MAX_CARS_MOVED and MAX_CARS_MOVED (inclusive) where a
    1. positive number means moving cars from location 1 to location 2
    2. negative number means moving cars from location 2 to location 1
    `add_non_linearity`: a boolean indicating whether we add the non-linearities from
    part (d) to the reward function;
    This function should have 2 possible return values depending on the value of `add_non_linearity`.

    Note: this function assumes that `state` is a tuple instead of a numpy array.
    This is because we are caching this function and numpy arrays are not hashable!
    """
    # TODO
    
    global LOC_1_RENTAL_LAMBDA, LOC_2_RENTAL_LAMBDA
     # Cars at each location at the end of the current day
    loc_1, loc_2 = state

    # reward = 0 if not enough cars to move 
    if action > 0 and loc_1 < action or action < 0 and loc_2 < abs(action):
        return 0
    
    expected_loc_1_rentals = expectation(LOC_1_RENTAL_LAMBDA, loc_1 - action)
    expected_loc_2_rentals = expectation(LOC_2_RENTAL_LAMBDA, loc_2 + action)
    
    rental_reward = RENTAL_REWARD * (expected_loc_1_rentals + expected_loc_2_rentals)

    if add_non_linearity and action >= FREE_MOVES:
        # employee move one car from Loc_1 to Loc_2 for free
        move_cost = MOVE_COST * (action - FREE_MOVES)
    else:
        move_cost = MOVE_COST * abs(action)
    
    award = rental_reward - move_cost

    # accont for addition overflow storage fee
    if add_non_linearity:
        if loc_1 - action > STORAGE_CAPACITY:
            award -= STORAGE_COST
        if loc_2 + action > STORAGE_CAPACITY:
            award -= STORAGE_COST
    
    return award
        


def q(state: np.ndarray, action: int, V: np.ndarray, add_non_linearity: bool) -> float:
    """
    Helper function to compute the q-value sum_{s', r} p(s', r | s, a) * (r + gamma * V(s')).
    Again, as before, we let s be `state` and a be `action`.
    Implement this function and use it as a subroutine for policy evaluation and improvement.
    """
    # TODO
    global MAX_CARS_LOC_1, MAX_CARS_LOC_2, GAMMA
    expected_V = 0
    for next_state in [(i, j) for i in range(MAX_CARS_LOC_1+1) for j in range(MAX_CARS_LOC_2+1)]:
        prob = transition_prob(state, action, next_state)
        # print(f"transition_prob: {prob}")
        expected_V += V[next_state] * prob
    expected_r = expected_rewards(state, action, add_non_linearity)
    return expected_r + GAMMA * expected_V

def policy_evaluation(policy: np.ndarray, V: np.ndarray, add_non_linearity: bool, threshold: float):
    """
    Runs policy evaluation on the given `policy` and updates `V` in-place.
    You should run the policy evaluation loop until the maximum change in `V` is less than `threshold`.
    """
    # TODO
    global MAX_CARS_LOC_1, MAX_CARS_LOC_2
    max_change = 10*threshold
    while max_change > threshold:
        change = 0
        for state in [(i, j) for i in range(MAX_CARS_LOC_1+1) for j in range(MAX_CARS_LOC_2+1)]:
            old_v = V[state]
            new_v = q(state, policy[state], V, add_non_linearity)
            change = max(change, abs(new_v - old_v))
            V[state] = new_v
            # print(f"V[{state}] = {new_v}")
        max_change = change
        # print(f"V: {V}")

def policy_improvement(policy: np.ndarray, V: np.ndarray, add_non_linearity: bool, threshold: float):
    """
    Runs policy improvement on the given `policy` and updates it in-place.
    As show in lecture, we update the policy greedily by choosing the action that maximizes the q-value.
    You should run the policy improvement loop until the policy is stable.
    Remember to call policy_evaluation() as a subroutine!

    For reference, the staff solution takes ~1.5 minutes to run.
    """
    # TODO
    global MAX_CARS_LOC_1, MAX_CARS_LOC_2, MAX_CARS_MOVED
    policy_stable = False
    while not policy_stable:
        changed = False
        for state in [(i, j) for i in range(MAX_CARS_LOC_1+1) for j in range(MAX_CARS_LOC_2+1)]:
            old_a = policy[state]
            best_q = (None, None) # (q, a) tuple
            for a in range(-MAX_CARS_MOVED, MAX_CARS_MOVED+1):
                q_val = q(state, a, V, add_non_linearity)
                # print(f"q_val: {q_val}  a:{a} old_a:{old_a}")
                if not best_q[0] or best_q[0] < q_val:
                    best_q = (q_val, a)
            # print(f"best_q: {best_q}")
            if best_q[1] != old_a:
                policy[state] = best_q[1]
                changed = True
        # print(f"policy: {policy}")
        if changed:
            policy_evaluation(policy, V, add_non_linearity, threshold)
        else:
            policy_stable = True


def value_iteration(V: np.ndarray, add_non_linearity: bool, threshold: float) -> np.ndarray:
    """
    Run value iteration and compute the optimal value functions.
    You should run the value iteration loop until the maximum change in `V` is less than `threshold`.
    Finally, return the optimal policy that greedily picks the q-value maximizing action at each state.

    For reference, the staff solution ~2 minutes to run.
    """
    # TODO
    max_change = 10*threshold
    while max_change > threshold:
        change = 0
        for state in [(i, j) for i in range(MAX_CARS_LOC_1+1) for j in range(MAX_CARS_LOC_2+1)]:
            old_v = V[state]
            max_v = None
            for a in range(-MAX_CARS_MOVED, MAX_CARS_MOVED+1):
                new_v = q(state, a, V, add_non_linearity)
                if max_v is None or max_v < new_v:
                    max_v = new_v
            change = max(change, abs(max_v - old_v))
            V[state] = max_v
            # print(f"V[{state}] = {new_v}")
        max_change = change
    
    policy = np.zeros((MAX_CARS_LOC_1 + 1, MAX_CARS_LOC_2 + 1)).astype(np.int32)
    for state in [(i, j) for i in range(MAX_CARS_LOC_1+1) for j in range(MAX_CARS_LOC_2+1)]:
        max_av = (None, None) # max a-V pair
        for a in range(-MAX_CARS_MOVED, MAX_CARS_MOVED+1):
            new_v = q(state, a, V, add_non_linearity)
            if max_av[1] is None or max_av[1] < new_v:
                max_av = (a, new_v)
        policy[state] = max_av[0]

    return policy


def init_policy() -> np.ndarray:
    global MAX_CARS_LOC_1, MAX_CARS_LOC_2
    return np.zeros((MAX_CARS_LOC_1 + 1, MAX_CARS_LOC_2 + 1)).astype(np.int32)

def init_value() -> np.ndarray:
    global MAX_CARS_LOC_1, MAX_CARS_LOC_2
    return np.zeros((MAX_CARS_LOC_1 + 1, MAX_CARS_LOC_2 + 1))

def plot_policy_value(policy: np.ndarray, V: np.ndarray, part: str) -> None:
    cmap = plt.get_cmap('RdBu', 11)
    mat = plt.matshow(policy.T, cmap=cmap, vmin=-5.5, vmax=5.5)
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.colorbar(mat, ticks=np.arange(-5, 6))
    plt.xlabel("Number of cars at location 1")
    plt.ylabel("Number of cars at location 2")
    plt.savefig(f'policy_{part}.png', dpi=300, bbox_inches='tight')
    plt.clf()

    plt.matshow(V.T)
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.colorbar()
    plt.xlabel("Number of cars at location 1")
    plt.ylabel("Number of cars at location 2")
    plt.savefig(f'V_{part}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    
if __name__ == "__main__":

    state_space = np.array(list(product(range(MAX_CARS_LOC_1 + 1), range(MAX_CARS_LOC_2 + 1)))).astype(np.int32)
    action_space = np.arange(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1).astype(np.int32)

    policy = init_policy()
    V = init_value()    
    policy_improvement(policy, V, add_non_linearity=False, threshold=THRESHOLD)
    plot_policy_value(policy, V, "part(b)")

    # ===================================================================================

    V = init_value()
    policy = value_iteration(V, add_non_linearity=False, threshold=THRESHOLD)
    plot_policy_value(policy, V, "part(c)")

    # ===================================================================================

    policy = init_policy()
    V = init_value()
    policy_improvement(policy, V, add_non_linearity=True, threshold=THRESHOLD)
    plot_policy_value(policy, V, "part(d)")

    # ===================================================================================

    V = init_value()
    policy = value_iteration(V, add_non_linearity=True, threshold=THRESHOLD)
    plot_policy_value(policy, V, "part(e)")
