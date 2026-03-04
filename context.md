This project is a VRP sandbox to be used for testing multiple different optimization methodologies to solve VRP problem instances.

Your task is to receive a request on which methodology to implement, use it to solve sample instances and describe how this methodology works.

The methodologies range from exact, heuristic, metaheuristics and machine-learning.

Folder instances contains the sample instances. In model.py you define the classes for the problem. In setup.py you read a given file and build the vrp model. In solver.py you select one of the many methodologies implemented to solve the model.

Whenever you create a new solution methodology, you implement it in a new .py file that is approprately named. You place it at the appropriate folder, based on its categorization. You also create a .md file explaining in detail the basic concepts of the methodology, with code snippet examples. 