from fym.envs import CartPoleEnv

env = CartPoleEnv
num_episodes = 100
policy = ...

for episode in range(num_episodes):
    state = env.initial()
    time = 0
    while True:
        action = policy(env.embedding(time, state))
        next_state = env.transition(time, state, action)
        reward = env.reward(time, state, action, next_state)
        if env.terminal(next_state) or env.timeout(time):
            break

        time += 1
        state = next_state
