import numpy as np

def bellman_expectation_value(P, R, policy, gamma):
    """
    Compute the state-value function V^pi for a given policy using
    the Bellman expectation equation.
    
    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states)
        R: Rewards, shape (num_states, num_actions, num_states)
        policy: Stochastic policy, shape (num_states, num_actions)
        gamma: Discount factor
    
    Returns:
        State-value function as numpy array of shape (num_states,)
    """
    num_states, num_actions = policy.shape
    P_pi = np.zeros((num_states, num_states))
    R_pi = np.zeros(num_states)

    for state in range(num_states):
        for action in range(num_actions):
            act_prob = policy[state][action]

            for next_state in range(num_states):
                transition_prob = P[state][action][next_state]
                reward = R[state][action][next_state]

                P_pi[state][next_state] += act_prob * transition_prob

                R_pi[state] += act_prob * transition_prob * reward

    I = np.eye(num_states)
    V = np.linalg.solve(I - gamma * P_pi, R_pi)

    return V 

