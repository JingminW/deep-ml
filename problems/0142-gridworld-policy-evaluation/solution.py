def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats
    """
    grid = [[0.0] * 5 for _ in range(5)]
    max_value_delta = float('inf')

    directions = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (-1, 0),
        'down': (1, 0)
    }

    terminal_states = {(0, 0), (0, 4), (4, 0), (4, 4)}

    while max_value_delta >= threshold:
        max_value_delta = 0.0
        new_grid = [[0.0] * 5 for _ in range(5)]

        for i in range(5):
            for j in range(5):
                if (i, j) in terminal_states:
                    continue

                new_value = 0.0

                for direction in directions:
                    prob = policy[(i, j)][direction]
                    dx, dy = directions[direction]
                    new_i, new_j = i + dx, j + dy

                    if 0 <= new_i < 5 and 0 <= new_j < 5:
                        next_value = grid[new_i][new_j]
                    else:
                        next_value = grid[i][j]

                    new_value += prob * (-1.0 + gamma * next_value)

                max_value_delta = max(max_value_delta, abs(new_value - grid[i][j]))

                new_grid[i][j] = new_value

        grid = new_grid

    return grid
                
