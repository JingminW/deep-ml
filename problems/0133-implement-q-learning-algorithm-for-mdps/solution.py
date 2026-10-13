import numpy as np
def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):

    Q_table = np.zeros((num_states, num_actions))
    terminal_states = set(terminal_states)
    non_terminal_states = [s for s in range(num_states) if s not in terminal_states]

    for episode in range(num_episodes):
        state = np.random.choice(non_terminal_states)

        while state not in terminal_states:

            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(Q_table[state])
            current_Q = Q_table[state][action]

            probs = P[state][action]
            next_state = np.random.choice(num_states, p = probs)
            reward = R[state][action]

            if next_state in terminal_states:
                error = reward - current_Q
            else:
                error = reward + gamma * np.max(Q_table[next_state]) - current_Q
            current_Q += alpha * error
            Q_table[state][action] = current_Q

            state = next_state

    return Q_table

