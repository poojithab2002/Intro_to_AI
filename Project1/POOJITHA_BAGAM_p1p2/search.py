import pickle
import json
import heapq  # Priority queue for A*

from FOON_class import Object

# -----------------------------------------------------------------------------------------------------------------------------#
# Function to verify if an item exists in the kitchen
def is_item_in_kitchen(kitchen, item):
    """
    Checks whether a specific item is available in the kitchen.
    
    Args:
    kitchen (list): List of items in the kitchen.
    item (Object): Object representing the desired item.
    
    Returns:
    bool: True if item exists in the kitchen, False otherwise.
    """
    for kitchen_item in kitchen:
        if (kitchen_item["label"] == item.label and
                sorted(kitchen_item["states"]) == sorted(item.states) and
                sorted(kitchen_item["ingredients"]) == sorted(item.ingredients) and
                kitchen_item["container"] == item.container):
            return True
    return False

# -----------------------------------------------------------------------------------------------------------------------------#
# Function to load the success rates from motion.txt file
def fetch_success_rates(file_path="motion.txt"):
    """
    Reads the success rates of functional units from the motion.txt file.

    Args:
    file_path (str): Path to the motion.txt file.

    Returns:
    dict: Dictionary mapping functional unit IDs to their success rates.
    """
    rates = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        fu_id = parts[0]
                        rate = float(parts[1])
                        rates[fu_id] = rate
                    except ValueError:
                        continue
    return rates

# -----------------------------------------------------------------------------------------------------------------------------#

# A* Search Algorithm
def astar_search(kitchen, target_object, rates, foon_objects, foon_units, obj_to_unit_map, tools):
    """
    Perform A* search to retrieve the task tree.

    Args:
    kitchen (list): Items in the kitchen.
    target_object (Object): The goal object to search for.
    rates (dict): Success rates from the motion.txt file.
    foon_objects (list): List of object nodes in FOON.
    foon_units (list): List of functional units in FOON.
    obj_to_unit_map (dict): Mapping of object nodes to functional units.
    tools (list): List of utensils available.

    Returns:
    list: Task tree represented by functional units.
    """
    priority_queue = []
    heapq.heappush(priority_queue, (0, target_object.id, 0))  # (f(n), node_id, g(n))
    task_tree = []
    visited = {}

    while priority_queue:
        _, current_node_id, accumulated_cost = heapq.heappop(priority_queue)

        if current_node_id in visited and accumulated_cost >= visited[current_node_id]:
            continue
        visited[current_node_id] = accumulated_cost

        current_object = foon_objects[current_node_id]

        if is_item_in_kitchen(kitchen, current_object):
            continue

        candidate_units = obj_to_unit_map[current_node_id]
        chosen_unit_idx = candidate_units[0]  # Always select the first path for simplicity

        if chosen_unit_idx in task_tree:
            continue
        task_tree.append(chosen_unit_idx)

        # Calculate the cost using the inverse of the success rate
        unit_rate = rates.get(chosen_unit_idx, 1)
        unit_cost = 1 / unit_rate

        # Add input nodes (children) to the priority queue
        for input_object in foon_units[chosen_unit_idx].input_nodes:
            input_object_id = input_object.id
            new_cost = accumulated_cost + unit_cost
            heuristic = len(foon_units[chosen_unit_idx].input_nodes)
            estimated_total = new_cost + heuristic
            heapq.heappush(priority_queue, (estimated_total, input_object_id, new_cost))

    task_tree.reverse()
    return [foon_units[i] for i in task_tree]

# -----------------------------------------------------------------------------------------------------------------------------#

# Iterative Deepening Search (IDS)
def iterative_search(kitchen, goal_obj, foon_objects, foon_units, obj_to_unit_map, tools):
    """
    Perform IDS to retrieve the task tree.

    Args:
    kitchen (list): Items in the kitchen.
    goal_obj (Object): The goal object to search for.
    foon_objects (list): List of object nodes in FOON.
    foon_units (list): List of functional units in FOON.
    obj_to_unit_map (dict): Mapping of object nodes to functional units.
    tools (list): List of utensils available.

    Returns:
    list: Task tree represented by functional units.
    """
    max_depth = 0

    while True:
        task_tree = []
        search_stack = [[goal_obj.id, 0]]
        explored_nodes = []
        skipped_nodes = []

        while search_stack:
            node_id, depth = search_stack.pop(0)

            if depth > max_depth:
                skipped_nodes.append(node_id)
                continue

            if node_id in explored_nodes:
                continue
            explored_nodes.append(node_id)

            current_node = foon_objects[node_id]

            if not is_item_in_kitchen(kitchen, current_node):
                candidate_units = obj_to_unit_map[node_id]
                chosen_unit_idx = candidate_units[0]

                if chosen_unit_idx in task_tree:
                    continue
                task_tree.append(chosen_unit_idx)

                child_nodes = []
                for input_node in foon_units[chosen_unit_idx].input_nodes:
                    input_node_id = input_node.id
                    add_to_search = True
                    if input_node.label in tools and len(input_node.ingredients) == 1:
                        for sibling_node in foon_units[chosen_unit_idx].input_nodes:
                            if sibling_node.label == input_node.ingredients[0] and sibling_node.container == input_node.label:
                                add_to_search = False
                                break
                    if add_to_search:
                        child_nodes.append(input_node_id)

                search_stack = [[child_node, depth + 1] for child_node in child_nodes] + search_stack

        if not skipped_nodes:
            task_tree.reverse()
            return [foon_units[i] for i in task_tree]
        max_depth += 1

# -----------------------------------------------------------------------------------------------------------------------------#

# Function to save task trees to a file
def write_task_tree(task_tree, filename):
    """
    Saves the task tree into a file.

    Args:
    task_tree (list): List of functional units to be saved.
    filename (str): Name of the file where the task tree will be written.
    """
    with open(filename, 'w') as file:
        file.write('// Task tree starts here\n')
        for unit in task_tree:
            file.write(unit.get_FU_as_text() + "\n")

# -----------------------------------------------------------------------------------------------------------------------------#

# Load FOON from a pickle file
def load_foon_data(file_path='FOON.pkl'):
    """
    Loads FOON data from a pickle file.

    Args:
    file_path (str): Path to the FOON pickle file.

    Returns:
    tuple: (functional_units, object_nodes, object_to_FU_map)
    """
    with open(file_path, 'rb') as file:
        foon_data = pickle.load(file)
    return foon_data["functional_units"], foon_data["object_nodes"], foon_data["object_to_FU_map"]

# -----------------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    foon_units, foon_objects, obj_to_FU_map = load_foon_data()

    with open('utensils.txt', 'r') as utensil_file:
        utensils = [line.strip() for line in utensil_file]

    kitchen_data = json.load(open('kitchen.json'))
    goal_items = json.load(open('goal_nodes.json'))

    success_rates = fetch_success_rates()

    for goal in goal_items:
        goal_obj = Object(goal["label"])
        goal_obj.states = goal["states"]
        goal_obj.ingredients = goal["ingredients"]
        goal_obj.container = goal["container"]

        found = False
        for foon_obj in foon_objects:
            if foon_obj.check_object_equal(goal_obj):
                found = True

                # IDS
                ids_task_tree = iterative_search(kitchen_data, foon_obj, foon_objects, foon_units, obj_to_FU_map, utensils)
                write_task_tree(ids_task_tree, f'IDS_output_{goal["label"]}.txt')

                # A*
                astar_task_tree = astar_search(kitchen_data, foon_obj, success_rates, foon_objects, foon_units, obj_to_FU_map, utensils)
                write_task_tree(astar_task_tree, f'Astar_output_{goal["label"]}.txt')

                break

        if not found:
            print(f'Goal object {goal_obj.label} not found in FOON.')
