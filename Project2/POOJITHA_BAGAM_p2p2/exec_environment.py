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
import time

folder_path = Path(__file__).resolve().parent


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

    def action(self, direction=None):
        """
        Move the container in the specified direction ('Up', 'Down', 'Left', 'Right').
        """
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
        """
        Stop the simulation in CoppeliaSim.
        """
        self.sim.stopSimulation()


def binary_to_integer(binary_str):
    return int(binary_str, 2)


def compute_state_reward(state_str):
    return sum(1 if char == "1" else -1 for char in state_str)


def train_agent():
    q_table = np.zeros((16, 4))
    exploration_prob = 1.0
    decay_rate = 0.001
    min_exploration_prob = 0.01
    discount_factor = 0.85
    learning_rate = 0.2
    rewards = []

    for episode in range(100):
        print(f"Training Episode {episode + 1}")
        environment = Simulation()
        total_reward = 0
        current_state = binary_to_integer(environment.compute_current_state())

        for _ in range(30):
            if np.random.rand() < exploration_prob:
                action = np.random.choice(len(environment.directions))
            else:
                action = np.argmax(q_table[current_state])

            environment.action(direction=environment.directions[action])
            next_state = binary_to_integer(environment.compute_current_state())
            reward = compute_state_reward(environment.compute_current_state())

            q_table[current_state, action] = (1 - learning_rate) * q_table[current_state, action] + \
                                              learning_rate * (reward + discount_factor * np.max(q_table[next_state]))
            total_reward += reward
            current_state = next_state

        exploration_prob = max(min_exploration_prob, np.exp(-decay_rate * episode))
        rewards.append(total_reward)
        environment.stop_simulation()

    print("Final Q-Table:")
    print(q_table)

    with open(folder_path / "episode_rewards.txt", "w") as reward_file:
        for i, reward in enumerate(rewards):
            reward_file.write(f"Episode {i + 1}: {reward}\n")

    np.savetxt(folder_path / "q_table.txt", q_table, fmt="%.10f")


def test_agent():
    q_table = np.loadtxt(folder_path / "q_table.txt")
    random_results = []
    q_learning_results = []

    for test in range(10):
        print(f"Testing Episode {test + 1} - Random Actions")
        environment = Simulation()
        for step in range(30):
            environment.action(direction=np.random.choice(environment.directions))
            if environment.compute_current_state() == "1111":
                random_results.append((True, step + 1))
                break
        else:
            random_results.append((False, 30))
        environment.stop_simulation()

        print(f"Testing Episode {test + 1} - Q-Learning Actions")
        environment = Simulation()
        current_state = binary_to_integer(environment.compute_current_state())
        for step in range(30):
            action = np.argmax(q_table[current_state])
            environment.action(direction=environment.directions[action])
            if environment.compute_current_state() == "1111":
                q_learning_results.append((True, step + 1))
                break
        else:
            q_learning_results.append((False, 30))
        environment.stop_simulation()

    print("Random Strategy Results:", random_results)
    print("Q-Learning Strategy Results:", q_learning_results)

    with open(folder_path / "test_results.txt", "w") as result_file:
        result_file.write("Random Strategy Results:\n")
        for idx, (success, steps) in enumerate(random_results):
            result_file.write(f"Test {idx + 1}: {'Pass' if success else 'Fail'} - {steps} steps\n")
        result_file.write("\nQ-Learning Strategy Results:\n")
        for idx, (success, steps) in enumerate(q_learning_results):
            result_file.write(f"Test {idx + 1}: {'Pass' if success else 'Fail'} - {steps} steps\n")


def main():
    train_agent()
    test_agent()


if __name__ == '__main__':
    main()
