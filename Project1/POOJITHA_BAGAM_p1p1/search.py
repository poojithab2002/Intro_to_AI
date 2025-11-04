import pickle
import json
from FOON_class import Object

# -----------------------------------------------------------------------------------------------------------------------------#

# Checks an ingredient exists in kitchen


def check_if_exist_in_kitchen(kitchen_items, ingredient):
    """
        parameters: a list of all kitchen items,
                    an ingredient to be searched in the kitchen
        returns: True if ingredient exists in the kitchen
    """

    for item in kitchen_items:
        if item["label"] == ingredient.label \
                and sorted(item["states"]) == sorted(ingredient.states) \
                and sorted(item["ingredients"]) == sorted(ingredient.ingredients) \
                and item["container"] == ingredient.container:
            return True

    return False


# -----------------------------------------------------------------------------------------------------------------------------#


def search_BFS(kitchen_items=[], target_node=None, object_nodes=None, functional_units=None, object_to_FU_map=None, utensil_list=None):
    # List to track the indices of functional units in the task tree
    task_tree_indices = []

    # Queue to hold object indices that need further exploration
    search_queue = []

    # Begin by adding the goal node's index to the search queue
    search_queue.append(target_node.id)

    # Keep track of already explored items to avoid redundant work
    explored_items = set()

    # Perform BFS to build the task tree
    while search_queue:
        current_object_index = search_queue.pop(0)  # Remove the first element
        if current_object_index in explored_items:
            continue  # Skip if this item has already been processed
        else:
            explored_items.add(current_object_index)

        # Retrieve the current object node from the list of object nodes
        current_object = object_nodes[current_object_index]

        # If the current object does not exist in the kitchen
        if not check_if_exist_in_kitchen(kitchen_items, current_object):
            # Get the functional units that generate this object
            candidate_units = object_to_FU_map[current_object_index]

            # Choose the first functional unit to continue the search
            selected_unit_idx = candidate_units[0]

            # Avoid re-adding functional units that are already part of the task tree
            if selected_unit_idx in task_tree_indices:
                continue

            # Add the selected functional unit to the task tree
            task_tree_indices.append(selected_unit_idx)

            # Explore all input nodes of the selected functional unit
            for input_node in functional_units[selected_unit_idx].input_nodes:
                input_index = input_node.id
                if input_index not in search_queue:
                    should_add_to_queue = True
                    if input_node.label in utensil_list and len(input_node.ingredients) == 1:
                        for sibling_node in functional_units[selected_unit_idx].input_nodes:
                            if sibling_node.label == input_node.ingredients[0] and sibling_node.container == input_node.label:
                                should_add_to_queue = False
                                break
                    if should_add_to_queue:
                        search_queue.append(input_index)

    # Reverse the task tree for proper ordering of functional units
    task_tree_indices.reverse()

    # Build a list of functional units corresponding to the collected indices
    final_task_tree = [functional_units[i] for i in task_tree_indices]

    return final_task_tree




def save_paths_to_file(task_tree, path):

    print('writing generated task tree to ', path)
    _file = open(path, 'w')

    _file.write('//\n')
    for FU in task_tree:
        _file.write(FU.get_FU_as_text() + "\n")
    _file.close()


# -----------------------------------------------------------------------------------------------------------------------------#

# creates the graph using adjacency list
# each object has a list of functional list where it is an output


def read_universal_foon(filepath='FOON.pkl'):
    """
        parameters: path of universal foon (pickle file)
        returns: a map. key = object, value = list of functional units
    """
    pickle_data = pickle.load(open(filepath, 'rb'))
    functional_units = pickle_data["functional_units"]
    object_nodes = pickle_data["object_nodes"]
    object_to_FU_map = pickle_data["object_to_FU_map"]

    return functional_units, object_nodes, object_to_FU_map


# -----------------------------------------------------------------------------------------------------------------------------#

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
                output_task_tree = search_BFS(kitchen_items, object, foon_object_nodes, foon_functional_units, foon_object_to_FU_map, utensils)
                save_paths_to_file(output_task_tree, 'output_BFS_{}.txt'.format(node["label"]))
                break
