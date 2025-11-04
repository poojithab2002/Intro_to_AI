import os
import json
import pickle
from FOON_class import Object
from search import load_foon_data, fetch_success_rates, iterative_search, astar_search, write_task_tree

def run_tests():
    # Load FOON data
    print("Loading FOON data...")
    foon_units, foon_objects, obj_to_FU_map = load_foon_data('FOON.pkl')

    # Load kitchen file
    print("Loading kitchen data...")
    kitchen_file = 'kitchen.json'
    with open(kitchen_file, 'r') as file:
        kitchen_data = json.load(file)

    # Load goal nodes
    print("Loading goal nodes...")
    goal_file = 'goal_nodes.json'
    with open(goal_file, 'r') as file:
        goal_items = json.load(file)

    # Load success rates from motion.txt
    print("Loading success rates from motion.txt...")
    success_rates = fetch_success_rates('motion.txt')

    # Load utensils data
    print("Loading utensils data...")
    utensils_file = 'utensils.txt'
    with open(utensils_file, 'r') as file:
        utensils = [line.strip() for line in file]

    # For each goal object, run the search algorithms and output the results
    for goal in goal_items:
        goal_obj = Object(goal["label"])
        goal_obj.states = goal["states"]
        goal_obj.ingredients = goal["ingredients"]
        goal_obj.container = goal["container"]

        print(f"Processing goal: {goal_obj.label}")

        # Find corresponding FOON object
        found = False
        for foon_obj in foon_objects:
            if foon_obj.check_object_equal(goal_obj):
                found = True

                # Perform IDS Search
                print(f"Running IDS for {goal_obj.label}...")
                ids_task_tree = iterative_search(kitchen_data, foon_obj, foon_objects, foon_units, obj_to_FU_map, utensils)
                ids_output_file = f'Test_IDS_output_{goal["label"]}.txt'
                write_task_tree(ids_task_tree, ids_output_file)
                print(f"IDS task tree saved to {ids_output_file}")

                # Perform A* Search
                print(f"Running A* search for {goal_obj.label}...")
                astar_task_tree = astar_search(kitchen_data, foon_obj, success_rates, foon_objects, foon_units, obj_to_FU_map, utensils)
                astar_output_file = f'Test_Astar_output_{goal["label"]}.txt'
                write_task_tree(astar_task_tree, astar_output_file)
                print(f"A* task tree saved to {astar_output_file}")

                break

        if not found:
            print(f"Goal object {goal_obj.label} not found in FOON.")
            continue

if __name__ == "__main__":
    run_tests()
