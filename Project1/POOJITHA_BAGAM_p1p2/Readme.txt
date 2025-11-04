FOON Task Tree Retrieval

This project implements two search algorithms, Iterative Deepening Search (IDS) and A* Search, to extract task trees from a Functional Object-Oriented Network (FOON). The goal is to retrieve task trees that can prepare any dish represented in the FOON, based on a list of available kitchen ingredients and utensils.

Project Structure

- search.py: Contains the main implementation of the search algorithms (IDS and A*) and helper functions to load FOON data, success rates, and write task trees to files.
- test-script.py: A test script that loads the FOON data, kitchen data, and goal objects, and then runs both IDS and A* algorithms to generate the task trees for each goal. The task trees are saved to separate files for IDS and A* searches.

Files Required:
1. FOON.pkl: A serialized version of FOON containing functional units, object nodes, and a mapping from object nodes to functional units.
2. kitchen.json: A file containing the list of available ingredients and utensils in the kitchen.
3. goal_nodes.json: A file containing the list of goal objects (dishes to prepare).
4. motion.txt: A file with success rates for each functional unit, which is used to compute the cost for A* search.
5. utensils.txt: A file containing a list of available utensils.

Installation and Setup

Python Packages
Ensure you have Python 3 installed. The following packages are required:

- pickle: For loading serialized FOON data.
- json: For reading and writing JSON data.
- heapq: For implementing the priority queue in A* search.

These packages are part of the Python standard library, so no external installation is required.

FOON_class.py
Ensure that you have the FOON_class.py file with the Object class implementation. This is required for object comparison and handling the FOON data.

Input Files
Make sure the following files are available in the same directory as the code:

- FOON.pkl
- kitchen.json
- goal_nodes.json
- motion.txt
- utensils.txt

Running the Program

Main Program: search.py
This file contains the main logic for both IDS and A* search. To run the program:

1. Place all input files in the same directory as search.py.
2. The search.py file automatically loads FOON, kitchen, and goal data.
3. It then runs both IDS and A* searches for each goal object and saves the corresponding task trees in separate files.

To run search.py, use the following command:
python search.py

The output will be two text files for each goal object:
- IDS_output_<goal>.txt: Task tree from Iterative Deepening Search.
- Astar_output_<goal>.txt: Task tree from A* Search.

Testing: test-script.py
The test script loads the same data as search.py and runs both search algorithms, saving the task trees to test-specific output files for comparison and validation.

To run the test script, use:
python test-script.py

The outputs will be saved as:
- Test_IDS_output_<goal>.txt
- Test_Astar_output_<goal>.txt

Output
Each task tree is saved in a text file with the format:
// Task tree starts here
<Functional Unit 1>
<Functional Unit 2>
...

If a goal object cannot be found in FOON, the program will output:
Goal object <goal> not found in FOON.


