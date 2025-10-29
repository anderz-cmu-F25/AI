import heapq
import time

def best_first_search(starting_state):

    frontier = []
    heapq.heappush(frontier, starting_state)
    visited_states = {starting_state: (None, 0)}

    start_time = time.time()
    while frontier:
        current = heapq.heappop(frontier)
        if current.is_goal():
            print(len(visited_states))
            return backtrack(visited_states, current)

        for n in current.get_neighbors():
            if n not in visited_states or n.dist_from_start < visited_states[n][1]:
                visited_states[n] = (current, n.dist_from_start)
                heapq.heappush(frontier, n)

        if time.time() - start_time > 5:
            break

    print(visited_states)

    return []

def backtrack(visited_states, goal_state):

    path = []

    current = goal_state
    while current:
        path.insert(0, current)
        current = visited_states[current][0]

    return path
