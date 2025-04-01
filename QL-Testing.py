import streamlit as st
import numpy as np
import random
import time

# -----------------------------------
# Q-learning for Jerry's Maze
# -----------------------------------

# Maze configuration:
# We define a 2x3 grid with states as follows:
#  S0: Start (top-left)
#  S1: Cheese (top-middle)
#  S2: Empty (top-right)
#  S3: Empty (bottom-left)
#  S4: Tom (cat) - penalty (bottom-middle)
#  S5: Home - goal (bottom-right)

# Define actions:
# 0: Up, 1: Down, 2: Left, 3: Right
ACTIONS = [0, 1, 2, 3]
action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

ROWS = 2
COLS = 3

def state_to_pos(state):
    return (state // COLS, state % COLS)

def pos_to_state(pos):
    r, c = pos
    return r * COLS + c

class MazeEnv:
    def __init__(self):
        self.max_steps = 5
        self.reset()

    def reset(self):
        # Start at S0; no cheese collected yet.
        self.state = 0
        self.steps = 0
        self.cheese_collected = False
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            raise Exception("Episode finished. Call reset()")
        current_row, current_col = state_to_pos(self.state)
        next_row, next_col = current_row, current_col

        if action == 0:  # Up
            next_row -= 1
        elif action == 1:  # Down
            next_row += 1
        elif action == 2:  # Left
            next_col -= 1
        elif action == 3:  # Right
            next_col += 1

        # If move is out-of-bound, stay in current cell.
        if next_row < 0 or next_row >= ROWS or next_col < 0 or next_col >= COLS:
            next_row, next_col = current_row, current_col

        next_state = pos_to_state((next_row, next_col))
        reward = 0

        # Reward conditions:
        if next_state == 4:  # Tom's cell
            reward = -10
            self.done = True
        elif next_state == 5:  # Home
            reward = +10
            self.done = True
        elif next_state == 1:  # Cheese cell
            if not self.cheese_collected:
                reward = +1
                self.cheese_collected = True
            else:
                reward = 0
        else:
            reward = 0

        self.state = next_state
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self.state, reward, self.done

# -----------------------------------
# Q-learning Training
# -----------------------------------

# Q-learning parameters:
alpha = 0.1    # learning rate
gamma = 0.99   # discount factor
epsilon = 0.2  # exploration rate
num_episodes = 1000

# Initialize Q-table (all zeros) for 6 states x 4 actions.
initial_Q_table = np.zeros((ROWS * COLS, len(ACTIONS)))
Q_table = np.copy(initial_Q_table)
env = MazeEnv()

# Display the initial Q-table in sidebar.
st.sidebar.subheader("Initial Q-Table (All Zeros)")
for s in range(ROWS * COLS):
    pos = state_to_pos(s)
    st.sidebar.write(f"State S{s} {pos}: {initial_Q_table[s, :]}")

# Run Q-learning algorithm:
for episode in range(num_episodes):
    state = env.reset()
    while True:
        # Epsilon-greedy action selection:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = int(np.argmax(Q_table[state, :]))
        next_state, reward, done = env.step(action)
        best_next = np.argmax(Q_table[next_state, :]) if not done else 0
        td_target = reward + (gamma * Q_table[next_state, best_next] if not done else 0)
        td_error = td_target - Q_table[state, action]
        Q_table[state, action] += alpha * td_error
        state = next_state
        if done:
            break

# Derive optimal policy from the Q-table.
policy = {s: int(np.argmax(Q_table[s, :])) for s in range(ROWS * COLS)}

# Simulate one episode using the learned policy to get the optimal path (state sequence).
state = env.reset()
optimal_path = [state]
actions_taken = []
while not env.done:
    act = policy[state]
    actions_taken.append(action_map[act])
    next_state, r, done = env.step(act)
    optimal_path.append(next_state)
    state = next_state

# -----------------------------------
# Streamlit App Interface & Image Animation
# -----------------------------------

st.title("Jerry's Maze Animation with Images")
st.write("""
Jerry, the mouse, navigates a tiny maze. He always starts at the same starting point (S0).  
His aim is to grab the cheese (in S1) and reach his home (S5) safely. However, if he goes down from S1,  
he may enter S4 where Tom is, and get caught!
""")
# Display maze overview image for visualization.
st.image("MAze.png", use_column_width=True)

# Slider for animation speed.
animation_speed = st.slider("Animation Speed (seconds per step)", 0.1, 2.0, 1.5)

# --- Define Image Paths for the Two Scenarios ---
optimal_path_images = ["Jerry.png", "Cheese .png", "Blank1.png", "Goal.png"]
caught_path_images = ["Jerry.png", "CHEESE.png", "TomJErry.png"]

scenario = st.selectbox("Select Scenario", 
                          ["Optimal Path: Cheese then Home", "Alternate: Gets Caught by Tom"])

if scenario == "Optimal Path: Cheese then Home":
    path_images = optimal_path_images
    st.write("**Optimal Path Selected:** S0 → S1 → S2 → S5")
else:
    path_images = caught_path_images
    st.write("**Alternate Path Selected:** S0 → S1 → S4 (Jerry gets caught)")

st.write("Click the button below to start the image animation.")

if st.button("Start Animation"):
    placeholder = st.empty()
    for img_file in path_images:
        placeholder.image(img_file, use_column_width=True)
        time.sleep(animation_speed)
    st.success("Animation complete!")

st.subheader("Q-learning Results")
st.write("Final Q-Table (State vs. Action Values):")
for s in range(ROWS * COLS):
    pos = state_to_pos(s)
    st.write(f"State S{s} {pos}: {Q_table[s, :]}")

st.write("Optimal Policy:")
for s in range(ROWS * COLS):
    pos = state_to_pos(s)
    if s in [4, 5]:
        st.write(f"State S{s} {pos}: Terminal")
    else:
        st.write(f"State S{s} {pos}: {action_map[policy[s]]}")

st.write("Optimal Path (State Sequence):", optimal_path)
st.write("Actions Taken:", actions_taken)
