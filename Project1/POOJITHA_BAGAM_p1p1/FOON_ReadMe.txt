FOON Functional Unit Search and Goal Node Finder


This project is a Python script that helps find functional units in a Functional Object-Oriented Network (FOON). It looks for functional units that use specific kitchen items and utensils or that help achieve certain goal nodes (like making a specific dish).


Table of Contents


1. What the Project Does
2. Files required
3. How to Set It Up
4. How to Run the Script
5. What the Script Does
6. Output


What the Project Does


This project has two main functions:
1. It finds and logs functional units for kitchen items and utensils. These are steps in a FOON that use or create the items.
2. It finds functional units for goal nodes. A goal node is something you want to create in the kitchen, like a prepared dish. The script finds the steps (functional units) needed to make that item.


Files Required


Make sure these files are in the same folder as test_script.py before running the program:


FOON_class.py: Defines the Object class and methods for handling FOON objects.
FOON.txt: The FOON network in a text file (initial format).
FOON.pkl: A pickled file that contains the FOON network, including functional units, object nodes, and mappings.
goal_nodes.json: Lists the goal nodes you want to search for in FOON.
kitchen.json: Lists the kitchen items available for use.
utensils.txt: Contains a list of utensils available in the kitchen.
search.py: Contains search functions for performing BFS and interacting with FOON.
preprocess.py: A script for preprocessing FOON data (if needed).
test_script.py: The main script that ties everything together and runs the program.




How to Set It Up


1. Make sure you have Python 3.x installed on your computer.
2. Download or clone all the project files into a single folder on your computer.
3. Install any required Python libraries (like `json` and `pickle`) if you don’t have them. You can install them using the following command:


   pip install json pickle


How to Run the Script


1. Open a terminal (or command prompt) and navigate to the folder where the files are saved.
2. Run the Python script by typing the following command:
   python3 test_script.py
3. The script will read from the ‘kitchen.json’, ‘utensils.txt’, and ‘goal_nodes.json’ files and then output the results into text files.




What the Script Does


1. Loads FOON Data
The script loads data from a **pickle file** (`FOON.pkl`) that contains the functional units and object nodes.


2. Finds Functional Units for Kitchen Items and Utensils
- The script checks which functional units use the items in `kitchen.json` and `utensils.txt` as inputs or outputs.
- The results are saved in a file called `kitchen_FU_log.txt`.


3. Finds Functional Units for Goal Nodes
- It then searches for functional units that help achieve the goals listed in `goal_nodes.json`. These goal nodes could be things like dishes or prepared ingredients.
- The results are saved in separate files, one for each goal node. The file names will look like `Goal_FU_<goal_label>.txt` (for example, `Goal_FU_Salad.txt`).


Output


Kitchen Items and Utensils:
The script will output a file called `kitchen_FU_log.txt`, which lists the functional units where the items or utensils are used. 
Goal Nodes:
For each goal node, the script will create a file like `Goal_FU_Salad.txt` that lists the steps needed to make the item: