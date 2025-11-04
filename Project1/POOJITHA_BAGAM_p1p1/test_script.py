import json
from search import read_universal_foon, search_BFS, save_paths_to_file
from FOON_class import Object


def load_items_and_utensils(kitchen_file_path, utensils_file_path):
    """
    Load kitchen items and utensils from their respective files.
    """
    with open(kitchen_file_path, 'r') as kitchen_file:
        kitchen_items = json.load(kitchen_file)

    utensil_list = []
    with open(utensils_file_path, 'r') as utensil_file:
        for line in utensil_file:
            utensil_list.append(line.strip())

    return kitchen_items, utensil_list


def log_functional_units_for_items(kitchen_items, utensil_list, functional_units, object_nodes):
    """
    Identify functional units for each kitchen item and utensil, and log them to a text file.
    """
    with open("kitchen_FU_log.txt", "w") as output_file:
        for item in kitchen_items:
            item_obj = Object(item["label"])
            item_obj.states = item.get("states", [])
            item_obj.ingredients = item.get("ingredients", [])
            item_obj.container = item.get("container", None)

            output_file.write(f"Locating functional units for kitchen item: {item['label']}\n")
            locate_functional_units(item_obj, functional_units, output_file)

        for utensil in utensil_list:
            utensil_obj = Object(utensil)
            utensil_obj.states = []
            utensil_obj.ingredients = []
            utensil_obj.container = None

            output_file.write(f"Locating functional units for utensil: {utensil}\n")
            locate_functional_units(utensil_obj, functional_units, output_file)


def locate_functional_units(item_obj, functional_units, file_handler):
    """
    Search for functional units where the given item object is either an input or output, and log them to the file.
    """
    located_units = []

    for functional_unit in functional_units:
        found_in_input = any(input_node.check_object_equal(item_obj) for input_node in functional_unit.input_nodes)
        found_in_output = any(output_node.check_object_equal(item_obj) for output_node in functional_unit.output_nodes)

        if found_in_input:
            file_handler.write(f"Located in input of functional unit:\n{functional_unit.get_FU_as_text()}\n")
            located_units.append(functional_unit)

        if found_in_output:
            file_handler.write(f"Located in output of functional unit:\n{functional_unit.get_FU_as_text()}\n")
            located_units.append(functional_unit)

    if not located_units:
        file_handler.write(f"No functional units found for {item_obj.label}\n")


def locate_goal_functional_units(goal_nodes, kitchen_items, functional_units, object_nodes, object_to_FU_map, utensils):
    """
    Identify relevant functional units for each goal node, specifically where the goal node is an output.
    """
    for goal in goal_nodes:
        goal_obj = Object(goal["label"])
        goal_obj.states = goal["states"]
        goal_obj.ingredients = goal["ingredients"]
        goal_obj.container = goal["container"]

        for obj in object_nodes:
            if obj.check_object_equal(goal_obj):
                # Execute BFS to get the task tree leading to the goal
                task_tree = search_BFS(kitchen_items, obj, object_nodes, functional_units, object_to_FU_map, utensils)

                # Identify functional units where the goal node is produced as an output
                print(f"Functional units for goal node: {goal['label']}")
                goal_units = []
                for functional_unit in task_tree:
                    if any(output_node.check_object_equal(goal_obj) for output_node in functional_unit.output_nodes):
                        print(functional_unit.get_FU_as_text())  # Log the functional unit where the goal node is produced
                        goal_units.append(functional_unit)

                # Optionally, save the located functional units
                if goal_units:
                    save_paths_to_file(goal_units, f'Goal_FU_{goal["label"]}.txt')

                break


if __name__ == '__main__':
    # Load the FOON data from the pickle file
    foon_functional_units, foon_object_nodes, foon_object_to_FU_map = read_universal_foon()

    # Load kitchen items and utensils from their respective files
    kitchen_items, utensils = load_items_and_utensils('kitchen.json', 'utensils.txt')

    # Load goal nodes from the JSON file
    goal_nodes = json.load(open("goal_nodes.json"))

    # Identify functional units for each kitchen item and utensil, and log them to a file
    log_functional_units_for_items(kitchen_items, utensils, foon_functional_units, foon_object_nodes)

    # Locate and log the functional units for each goal node
    locate_goal_functional_units(goal_nodes, kitchen_items, foon_functional_units, foon_object_nodes, foon_object_to_FU_map, utensils)
