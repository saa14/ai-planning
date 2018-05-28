# Instructions to run the script
For the main solution using python constraint library(Runs by default)
- Install python constraint library
- The solver being used is BacktrackingSolver()

For the ortools based solution 
- Install ortools python library
- Uncomment the code for the two methods solveCSPUsingOrtools(data) and solveCostMatrixUsingOrtools(data)
- Execute these methods separately

# How to run the script from terminal
python aiproject2.py <filename>

(where filename maybe sample1.txt,sample2.txt etc. and they are of the same format as the sample text files given as part of the project)

# Output File
- The script creates text files for mappings in the specified pretty printed format. The text files generated have names according to the input sample file given as input
.(Eg. sample6.txt will generate mapping_sample6.txt)
- If there is no feasible solution possible for a sample file (assuming the script runs more than 60 seconds searching for the solution in search space), the program terminates
and a "No feasible solution" message is printed on the terminal.
- The script executed successfully for sample4.txt and sample5.txt and the solutions for these files are included. However, the script terminated (ran for more than 60 seconds)
in case of sample1.txt,sample2.txt and sample3.txt for which there is no feaible solution or assignment possible.




