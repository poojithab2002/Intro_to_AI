import json
from FOON_class import Object
from MCTS import MCTS, read_universal_foon, save_paths_to_file  # Assuming your MCTS code is saved as search_MCTS.py

# Main function for testing MCTS and generating task trees for different goal objects
if __name__ == '__main__':
    # Load the FOON graph
    foon_functional_units, foon_object_nodes, foon_object_to_FU_map = read_universal_foon('FOON.pkl')

    # Load utensils and kitchen items
    utensils = []
    with open('utensils.txt', 'r') as f:
        for line in f:
            utensils.append(line.rstrip())

    # Load kitchen items and goal nodes
    kitchen_items = json.load(open('kitchen.json'))
    goal_nodes = json.load(open("goal_nodes.json"))

    # Iterate over the goal nodes and generate task trees using MCTS
    for node in goal_nodes:
        node_object = Object(node["label"])
        node_object.states = node["states"]
        node_object.ingredients = node["ingredients"]
        node_object.container = node["container"]

        # Find the corresponding object in the FOON graph
        goal_found = False
        for object in foon_object_nodes:
            if object.check_object_equal(node_object):
                goal_found = True
                print(f"Generating task tree for goal: {node['label']}")
                
                # Call MCTS to generate the task tree for the given goal node
                output_task_tree = MCTS(kitchen_items, object, foon_object_nodes, foon_functional_units, foon_object_to_FU_map)
                
                # Save the generated task tree to a file
                output_file_path = f'output_MCTS_{node["label"].replace(" ", "_")}.txt'
                save_paths_to_file(output_task_tree, output_file_path)
                print(f"Task tree for {node['label']} saved to {output_file_path}")
                break

        # If the goal node was not found, print an error message
        if not goal_found:
            print(f'{node_object.label} - Goal node not found in FOON')
