import pickle
import json
import random
import math
from FOON_class import Object

# Checks if an ingredient exists in the kitchen
def check_if_exist_in_kitchen(kitchen_items, ingredient):
    for item in kitchen_items:
        if item["label"] == ingredient.label \
                and sorted(item["states"]) == sorted(ingredient.states) \
                and sorted(item["ingredients"]) == sorted(ingredient.ingredients) \
                and item["container"] == ingredient.container:
            return True
    return False

def MCTS(kitchen_stock, goal_item, obj_node_list=[], func_unit_list=[], obj_to_unit_dict=[], num_simulations=1000, max_depth=50):
    # List to hold the selected task tree functional units
    task_tree_units = []

    # A dictionary to store the success count (wins) and the number of attempts (trials) for each functional unit (FU)
    fu_statistics = {i: {"wins": 0, "attempts": 0} for i in range(len(func_unit_list))}

    # Load the motion success probabilities from the motion.txt file
    motion_probabilities = load_motion_probabilities("motion.txt")

    # Perform the MCTS search for the target node
    def execute_mcts(target, depth=0):
        # List to hold the sequence of units selected during the MCTS search
        selected_units = []
        visited_nodes = set()  # Keeps track of visited nodes to avoid revisiting

        # Simulate the task tree search starting from the target node
        def traverse_task_tree(current_node, current_depth):
            # Prevent the search from going too deep
            if current_depth > max_depth:
                return False

            # Base case: if the node exists in the kitchen, mark the search as successful
            if check_if_exist_in_kitchen(kitchen_stock, current_node):
                return True

            # If the node has been visited, skip to prevent infinite loops
            if current_node.id in visited_nodes:
                return False

            visited_nodes.add(current_node.id)  # Mark the node as visited

            # Retrieve the functional units responsible for producing the node
            available_units = obj_to_unit_dict[current_node.id]
            best_unit = None
            best_ucb_score = -float('inf')

            # Simulate each functional unit and select the best one based on the UCB1 score
            for fu_index in available_units:
                fu_statistics[fu_index]["attempts"] += 1  # Increment the number of attempts for this FU
                successful_trials = 0

                # Simulate the unit multiple times
                for _ in range(num_simulations):
                    if simulate_fu(func_unit_list[fu_index], motion_probabilities):
                        successful_trials += 1

                # Update the number of successful executions
                fu_statistics[fu_index]["wins"] += successful_trials

                # Calculate the UCB1 score
                total_trials = sum([stats["attempts"] for stats in fu_statistics.values()])
                win_ratio = fu_statistics[fu_index]["wins"] / fu_statistics[fu_index]["attempts"]
                exploration_term = math.sqrt(2 * math.log(total_trials) / fu_statistics[fu_index]["attempts"])
                ucb1_value = win_ratio + exploration_term

                # Track the best functional unit based on UCB1 score
                if ucb1_value > best_ucb_score:
                    best_ucb_score = ucb1_value
                    best_unit = fu_index

            # Add the best functional unit to the task tree
            if best_unit is not None:
                selected_units.append(best_unit)
                for input_node in func_unit_list[best_unit].input_nodes:
                    traverse_task_tree(obj_node_list[input_node.id], current_depth + 1)  # Recurse with increased depth

        traverse_task_tree(target, depth)
        return selected_units

    # Execute the MCTS for the target node
    task_unit_indices = execute_mcts(goal_item)

    # Convert the indices into functional units
    task_tree_units = [func_unit_list[i] for i in task_unit_indices]
    return task_tree_units


# Simulate the execution of a functional unit using the success probabilities
def simulate_fu(functional_unit, motion_prob_dict):
    # Get the motion type for the functional unit
    motion_type = functional_unit.motion_node
    # Look up the success probability for the motion, default to 50% if not found
    success_chance = motion_prob_dict.get(motion_type, 0.5)
    return random.uniform(0, 1) <= success_chance

# Load success probabilities from the motion.txt file
def load_motion_probabilities(file_path):
    success_rates = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            motion_type, success_rate = line.split('\t')
            try:
                success_rates[motion_type] = float(success_rate)
            except ValueError:
                print(f"Skipping invalid entry for motion '{motion_type}' in {file_path}")
    return success_rates

# Save the task tree to a file
def save_paths_to_file(task_tree, path):
    print('writing generated task tree to ', path)
    with open(path, 'w') as _file:
        _file.write('//\n')
        for FU in task_tree:
            _file.write(FU.get_FU_as_text() + "\n")

# Reads the FOON graph from a pickle file
def read_universal_foon(filepath='FOON.pkl'):
    pickle_data = pickle.load(open(filepath, 'rb'))
    functional_units = pickle_data["functional_units"]
    object_nodes = pickle_data["object_nodes"]
    object_to_FU_map = pickle_data["object_to_FU_map"]

    return functional_units, object_nodes, object_to_FU_map

# Main function
if __name__ == '__main__':
    foon_functional_units, foon_object_nodes, foon_object_to_FU_map = read_universal_foon()

    utensils = []
    with open('utensils.txt', 'r') as f:
        for line in f:
            utensils.append(line.rstrip())

    kitchen_items = json.load(open('kitchen.json'))
    goal_nodes = json.load(open("goal_nodes.json"))
    
    for node in goal_nodes:
        node_object = Object(node["label"])
        node_object.states = node["states"]
        node_object.ingredients = node["ingredients"]
        node_object.container = node["container"]
        for object in foon_object_nodes:
            if object.check_object_equal(node_object):
                output_task_tree = MCTS(kitchen_items, object, foon_object_nodes, foon_functional_units, foon_object_to_FU_map)
                break

