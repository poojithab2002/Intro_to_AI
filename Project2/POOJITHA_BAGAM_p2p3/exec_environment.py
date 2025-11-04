"""
This code communicates with the CoppeliaSim software and simulates shaking a container to mix objects of different colors.

Install dependencies:
https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append(r'\zmqRemoteApi')  # Change to the path of your ZMQ Python API
from zmqRemoteApi import RemoteAPIClient
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import random
import time

folder_path = Path(__file__).resolve().parent
model_name = 'mixing_strategy_model.h5'


class Simulation:
    def __init__(self, sim_port=23000):
        self.sim_port = sim_port
        self.directions = ['Up', 'Down', 'Left', 'Right']
        self.initialize_simulation()

    def initialize_simulation(self):
        self.client = RemoteAPIClient('localhost', port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')

        self.default_idle_fps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        self.get_object_handles()
        self.sim.startSimulation()
        self.drop_objects()
        self.get_objects_in_container_handles()

    def get_object_handles(self):
        self.table_handle = self.sim.getObject('/Table')
        self.container_handle = self.sim.getObject('/Table/Box')

    def drop_objects(self):
        self.blocks = 18
        friction_cube = 0.06
        friction_cup = 0.8
        block_length = 0.016
        block_mass = 14.375e-03

        script_handle = self.sim.getScript(self.sim.scripttype_childscript, self.table_handle)
        self.client.step()
        self.sim.callScriptFunction(
            'setNumberOfBlocks', script_handle,
            [self.blocks], [block_mass, block_length, friction_cube, friction_cup], ['cylinder']
        )

        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signal_value = self.sim.getFloatSignal('toPython')
            if signal_value == 99:
                for _ in range(20):
                    self.client.step()
                break

    def get_objects_in_container_handles(self):
        self.block_handles = []
        self.block_type = "Cylinder"
        for idx in range(self.blocks):
            handle = self.sim.getObjectHandle(f'{self.block_type}{idx}')
            self.block_handles.append(handle)

    def fetch_container_position(self):
        return self.sim.getObjectPosition(self.container_handle, self.sim.handle_world)

    def fetch_all_block_positions(self):
        positions = []
        for block_handle in self.block_handles:
            block_position = self.sim.getObjectPosition(block_handle, self.sim.handle_world)
            positions.append(list(block_position[:2]))
        return positions

    def analyze_object_distribution(self, blocks, container_position):
        distribution = [0, 0, 0, 0]
        for block_position in blocks:
            if block_position[0] >= container_position[0]:
                if block_position[1] >= container_position[1]:
                    distribution[0] += 1
                else:
                    distribution[3] += 1
            else:
                if block_position[1] >= container_position[1]:
                    distribution[1] += 1
                else:
                    distribution[2] += 1
        return distribution

    def compute_current_state(self):
        container_position = self.fetch_container_position()
        block_positions = self.fetch_all_block_positions()
        blue_blocks = block_positions[:9]
        red_blocks = block_positions[9:]

        blue_distribution = self.analyze_object_distribution(blue_blocks, container_position)
        red_distribution = self.analyze_object_distribution(red_blocks, container_position)

        state_representation = ""
        for i in range(4):
            if blue_distribution[i] >= 1 and red_distribution[i] >= 1 and abs(blue_distribution[i] - red_distribution[i]) == 0:
                state_representation += "1"
            else:
                state_representation += "0"

        return state_representation

    def move_container(self, direction=None):
        if direction not in self.directions:
            print(f"Invalid direction: {direction}. Choose from {self.directions}")
            return

        container_position = self.sim.getObjectPosition(self.container_handle, self.sim.handle_world)
        movement = 0.02
        steps = 5
        axis_index, direction_values = (1, [1, -1]) if direction in ['Up', 'Down'] else (0, [1, -1])

        for value in direction_values:
            for _ in range(steps):
                container_position[axis_index] += value * movement / steps
                self.sim.setObjectPosition(self.container_handle, self.sim.handle_world, container_position)
                self.client.step()

    def stop_simulation(self):
        self.sim.stopSimulation()


class DQN_learning:
    def __init__(self, input_size, output_size, pretrained_model=None):
        self.input_size = input_size
        self.output_size = output_size
        self.memory_buffer = []
        self.discount_factor = 0.85
        self.exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        self.learning_rate = 0.005

        self.model = pretrained_model if pretrained_model else self.build_neural_network()

    def build_neural_network(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.input_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.output_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def store_experience(self, current_state, action_taken, reward_received, next_state, goal_reached):
        self.memory_buffer.append((current_state, action_taken, reward_received, next_state, goal_reached))

    def select_action(self, current_state):
        if np.random.rand() <= self.exploration_rate:
            return np.random.choice(self.output_size)
        q_values = self.model.predict(current_state, verbose=0)
        return np.argmax(q_values[0])

    def optimize_model(self, batch_size):
        if len(self.memory_buffer) < batch_size:
            return

        minibatch = random.sample(self.memory_buffer, batch_size)
        for current_state, action, reward, next_state, goal_reached in minibatch:
            target = reward
            if not goal_reached:
                target += self.discount_factor * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_values = self.model.predict(current_state, verbose=0)
            target_values[0][action] = target
            self.model.fit(current_state, target_values, epochs=1, verbose=0)

        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay


def binary_to_array(binary):
    return np.array([[int(x) for x in binary]])


def calculate_reward(state):
    return sum(1 if x == "1" else -1 for x in state)


def train_mixing_strategy():
    input_size = 4
    output_size = 4
    agent = DQN_learning(input_size, output_size)

    episodes = 300
    max_steps = 30
    batch_size = 32

    training_logs = []  # To store logs for training

    for episode in range(episodes):
        print(f"Running Episode: {episode + 1}")
        simulation_env = Simulation()
        current_state_binary = simulation_env.compute_current_state()
        current_state = binary_to_array(current_state_binary)

        episode_reward = 0
        td_errors = []  # Store TD errors for each episode

        for step in range(max_steps):
            action = agent.select_action(current_state)
            simulation_env.move_container(simulation_env.directions[action])
            next_state_binary = simulation_env.compute_current_state()
            next_state = binary_to_array(next_state_binary)
            reward = calculate_reward(next_state_binary)
            goal_reached = next_state_binary == "1111"

            agent.store_experience(current_state, action, reward, next_state, goal_reached)

            # Calculate TD error
            target = reward
            if not goal_reached:
                target += agent.discount_factor * np.amax(agent.model.predict(next_state, verbose=0)[0])
            predicted_q_value = agent.model.predict(current_state, verbose=0)[0][action]
            td_error = target - predicted_q_value
            td_errors.append(td_error)

            episode_reward += reward
            current_state = next_state

            if goal_reached:
                break

        # Update model
        agent.optimize_model(batch_size)
        simulation_env.stop_simulation()

        # Log the episode details
        training_logs.append(
            f"Episode {episode + 1}: Reward = {episode_reward}, Exploration Rate = {agent.exploration_rate:.3f}, TD Errors = {td_errors}"
        )
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    # Save training logs to a file
    training_log_file = folder_path / "training_log.txt"
    with open(training_log_file, "w") as log_file:
        for log in training_logs:
            log_file.write(log + "\n")
    print(f"Training log saved to {training_log_file}")

    # Save the trained model
    agent.model.save(model_name)
    print(f"Model saved as {model_name}")


def evaluate_mixing_strategy():
    input_size = 4
    output_size = 4
    pretrained_model = load_model(model_name)
    agent = DQN_learning(input_size, output_size, pretrained_model=pretrained_model)

    episodes = 100
    max_steps = 30
    evaluation_results = []  # To store testing results
    successful_mixes = 0  # Counter for successful mixes

    for episode in range(episodes):
        print(f"Running Test Episode: {episode + 1}")
        simulation_env = Simulation()
        current_state_binary = simulation_env.compute_current_state()
        current_state = binary_to_array(current_state_binary)

        success = False
        for step in range(max_steps):
            q_values = agent.model.predict(current_state, verbose=0)
            action = np.argmax(q_values[0])
            simulation_env.move_container(simulation_env.directions[action])
            next_state_binary = simulation_env.compute_current_state()
            current_state = binary_to_array(next_state_binary)

            if next_state_binary == "1111":
                evaluation_results.append(f"Episode {episode + 1}: Success in {step + 1} steps")
                successful_mixes += 1  # Increment the success counter
                success = True
                break

        if not success:
            evaluation_results.append(f"Episode {episode + 1}: Failed to reach goal state")

        simulation_env.stop_simulation()

    # Add total successes to the test results
    evaluation_results.append(f"\nTotal Successful Mixes: {successful_mixes} out of {episodes}")

    # Save testing results to a file
    evaluation_results_file = folder_path / "test_results.txt"
    with open(evaluation_results_file, "w") as results_file:
        for result in evaluation_results:
            results_file.write(result + "\n")
    print(f"Test results saved to {evaluation_results_file}")
    print(f"Total Successful Mixes: {successful_mixes} out of {episodes}")


if __name__ == "__main__":
    print("Starting DQN Training and Testing...")
    # Train the DQN agent
    print("Training the Mixing Strategy...")
    train_mixing_strategy()

    # Test the trained DQN agent

    print("\nTesting the trained Mixing Strategy...")
    evaluate_mixing_strategy()

    print("\nTraining and Testing Completed!")
