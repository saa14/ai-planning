import string
import sys
import numpy as np
# from ortools.constraint_solver import pywrapcp
# from ortools.graph import pywrapgraph
from constraint import *
import time
import multiprocessing
def removeSpace(line):
    new_line = line.replace('\r','')
    new_line = new_line.replace('\n','')
    new_line = new_line.split(" % ")
    new_line = "".join(new_line[0])
    new_line = new_line.strip()
    return new_line

def readFile():
    infile = sys.argv[1]
    # infile = "sample1.txt"
    resources_array = []
    with open (infile,"r") as source:
        i = 0
        line_array = []
        for line in source:
            if line.startswith("%%%"):
                i=i+1
                if not(i%2):
                    i = 1
                    resources_array.append(line_array)
                    line_array = []
                else:
                    if i!=1:
                        line_array.append(removeSpace(line))
            else:
                line_array.append(removeSpace(line))

            if line.startswith("%%% Mission"):
                line_array = []
                for line in source:
                    line_array.append(removeSpace(line))
                resources_array.append(line_array)
                break
    return resources_array
def prepareData(resources_array):
    suas_models = resources_array[0]
    mission_model = []
    mission_type = []
    for t in resources_array[1]:
        type_name = t.split(" ")[1]
        mission_type.append(type_name)
        for i,model in enumerate(suas_models):
            split_t = int((t.split(" ")[2])[i]) 
            if split_t:
                mission_model.append((type_name,model))
    pilots = {}
    for p in resources_array[2]:
        split_p = p.split(" ")
        pilot_name = (split_p[1])
        pilot_q = {}
        for i,model in enumerate(suas_models):
            if int((split_p[3])[i]):
                pilot_q[model] = ((split_p[2])[i],'Yes')
            else:
                pilot_q[model] = ((split_p[2])[i],'No')
        pilots[pilot_name] = pilot_q
    number_of_suas = resources_array[3][0].split(" ")[1]
    number_suas = {}
    for i,model in enumerate(suas_models):
            number_suas[model] = int(number_of_suas[i])
    missions = []
    for m in resources_array[4]:
        split_m = m.split(" ")
        m_number = split_m[1]
        m_type = split_m[2]
        missions.append((m_number,m_type)) 
    return (suas_models,mission_model,pilots,number_suas,missions,mission_type)
def getIndexMappings(data,num_models,num_missions,num_pilots):
    suas_mapping = {}
    mission_mapping = {}
    pilot_mapping = {}
    index = 0
    for i in range(num_models):
        count = 0        
        while(count < data[3].values()[i]):
            suas_mapping[index] = (data[3].keys()[i],count)
            count = count+1
            index =index+1
    # print suas_mapping 
    for i in range(num_missions):
        mission_mapping[i] = data[4][i]
    # print mission_mapping
    for i in range(num_pilots):
        pilot_mapping[i] = data[2].keys()[i]
    # print pilot_mapping
    return (suas_mapping,mission_mapping,pilot_mapping)
def solveUsingPythonConstraint(data,queue):
    problem = Problem(BacktrackingSolver())
    
    num_models = len(data[0])
    num_types = len(data[1])
    num_pilots = len(data[2])
    num_missions = len(data[4])
    num_suas = sum(data[3].values())
    
    index_mappings =  getIndexMappings(data,num_models,num_missions,num_pilots)
    suas_mapping = index_mappings[0]
    mission_mapping = index_mappings[1]
    pilot_mapping = index_mappings[2]
    
#   Pilot x SUAS matrix with 1 if Pilot qualifies to fly for SUAS else 0 
    pilot_x_suas = np.zeros((num_pilots,num_suas),dtype=int)
#   Pilot x SUAS matrix with 1 if Pilot favors to fly the SUAS else 0 
    pilot_x_favoritesuas = np.zeros((num_pilots,num_suas),dtype=int)
    
    for i in range(num_pilots):
        pilot_dict = data[2].get(pilot_mapping.get(i))
        for j in suas_mapping.keys():
            t1 = suas_mapping[j]
            t2 = pilot_dict[t1[0]]
            if int(t2[0]):
                if (t2[1] == 'Yes'):
                    pilot_x_favoritesuas[i][j] = 1
                else:
                    pilot_x_favoritesuas[i][j] = 0
                pilot_x_suas[i][j] = 1
            else:
                pilot_x_suas[i][j] = 0
    # print pilot_x_suas
    # print pilot_x_favoritesuas
    
#   Mission x SUAS matrix with 1 if Mission requires the specific type of SUAS else 0
    mission_x_suas = np.zeros((num_missions,num_suas),dtype=int)
    for m in mission_mapping.keys():
        m_type = mission_mapping[m][1]
        for j in suas_mapping.keys():
            model = suas_mapping[j][0]
            if (m_type,model) in data[1]:
                mission_x_suas[m][j] = 1
            else:
                mission_x_suas[m][j] = 0
    # print mission_x_suas
    
    #Creating Pilot Variables
    for i in range(num_missions):               
        problem.addVariable("P %d" % i, range(num_pilots))
    pilot_variables = ["P %d"%j for j in range(num_missions)]
    #Creating Mission Variables
    for i in range(num_missions):
        problem.addVariable("M %d" % i, [i])
    mission_variables = ["M %d"%j for j in range(num_missions)]
    #Creating SUAS Variables
    for i in range(num_missions):       
        problem.addVariable("S %d" % i, suas_mapping.keys())    
    SUAS_variables = ["S %d"%j for j in range(num_missions)]
    #Constraint to not allow 2 pilots being assigned to the same SUAS or 1 pilot being assigned more than SUAS
    for i in range(num_missions):
        k = i+1
        for j in range(k, num_missions):          
            problem.addConstraint(lambda pilot_1, pilot_2, suas_1, suas_2:(pilot_1 != pilot_2 and suas_1 != suas_2) or (pilot_1 == pilot_2 and suas_1 == suas_2),("P %d" % i, "P %d" % j,"S %d" % i, "S %d" % j))
    #Constraint to assign SUAS to Pilot only if he qualifies for it
    for i in range(num_missions):
        problem.addConstraint(lambda pilot,suas:pilot_x_suas[pilot][suas] == 1,("P %d" % i,"S %d" % i))
    #Constraint to assign SUAS to Mission only if the mission type is compatible with it
    for i in range(num_missions):        
        problem.addConstraint(lambda mission,suas: mission_x_suas[mission][suas] == 1,("M %d" %i,"S %d" % i))        
    
    # Constraint to assign SUAS to Pilot only if he favors it
    # for i in range(num_missions):
    #     problem.addConstraint(lambda suas,pilot:pilot_x_favoritesuas[pilot][suas] == 1,("S %d" % i, "P %d" % i))
    
    #Constraint to limit the number of Missions assigned to a Pilot to a constant(3 in this case) 
    limit = 3
    for i in range(num_pilots):     
        problem.addConstraint(SomeNotInSetConstraint([i],num_missions-limit),pilot_variables)
    
    solution = problem.getSolution()        
    solution_tuple = (solution,pilot_x_favoritesuas)
    queue.put(solution_tuple)            
def prettyPrintOutput(solution,data,favorite_matrix):
    num_models = len(data[0])
    num_types = len(data[1])
    num_pilots = len(data[2])
    num_missions = len(data[4])
    num_suas = sum(data[3].values())
    index_mappings = getIndexMappings(data, num_models,num_missions,num_pilots)
    suas_mapping = index_mappings[0]
    mission_mapping = index_mappings[1]
    pilot_mapping = index_mappings[2]
    mission_suas = {}
    mission_pilot = {}
    for k,v in solution.iteritems():
        if "S" in k:
            mission_suas[int(k.split(' ')[1])] = v
        elif "P" in k:
            mission_pilot[int(k.split(' ')[1])] = v 
        else:
            continue
    outfile = open("mapping_"+sys.argv[1].split(".")[0]+".txt", "w")    
    for i in range(len(data[4])):
        fav = "Yes" if favorite_matrix[mission_pilot[i]][mission_suas[i]] == 1 else "No"
        outfile.write("M"+str(int(mission_mapping[i][0]))+" "+str(suas_mapping.get(mission_suas[i])[0])+" "+str(pilot_mapping.get(mission_pilot[i]))+" "+fav+"\n")       
    # print mission_suas
    # print mission_pilot
resources_array = readFile()
data = prepareData(resources_array)
# solveCSPUsingOrtools(data)
# solveCostMatrixUsingOrtools(data)
queue = multiprocessing.Queue() 
p = multiprocessing.Process(target=solveUsingPythonConstraint, name="aiproject2", args=(data,queue))
p.start()
p.join(60)
terminated = False
if p.is_alive():
    terminated = True
    print "Script is still running after 60 seconds...Killing the script"
    p.terminate()
    p.join()
    
if terminated == False:
    solution = queue.get()
    if (solution[0] != None):
        prettyPrintOutput(solution[0],data,solution[1])
    else:
        print "No feasible solution satisfying all the basic constraints for the sample exists"
else:
    print "No feasible solution satisfying all the basic constraints for the sample exists" 

##########Ortools Solution###################
# def solveCSPUsingOrtools(data):
#     solver = pywrapcp.Solver("suas_solver")
#     num_models = len(data[0])
#     num_types = len(data[1])
#     num_pilots = len(data[2])
#     num_missions = len(data[4])
#     num_suas = sum(data[3].values())
#     suas_mapping = {}
#     mission_mapping = {}
#     pilot_mapping = {}
#     index = 0
#     for i in range(num_models):
#         count = 0        
#         while(count < data[3].values()[i]):
#             suas_mapping[index] = (data[3].keys()[i],count)
#             count = count+1
#             index =index+1
#     # print suas_mapping 
#     for i in range(num_missions):
#         mission_mapping[i] = data[4][i]
#     # print mission_mapping
#     for i in range(num_pilots):
#         pilot_mapping[i] = data[2].keys()[i]
#     # print pilot_mapping  
  
#     #Creating mission assignment variable (Pilot j is assigned a mission[j,i] using suas i)
#     pilot_x_suas = {}
#     for i in range(num_pilots):
#         for j in range(num_suas):
#             pilot_x_suas[(i,j)] = solver.IntVar(0, num_missions - 1, "Pilot %d and SUAS %d" % (i,j))

#     missions_flat = [pilot_x_suas[(i, j)] for i in range(num_pilots) for j in range(num_suas)] 
#     # solver.Add(pilot_x_suas[(j, i)] >= 0)
#     # solver.Add(pilot_x_suas[(j, i)] <= num_missions - 1)
#     mission_x_suas = {}
#     #Creating mission assignment variable (Pilot j is assigned a mission[j,i] using suas i)
#     for i in range(num_missions):
#         for j in range(num_suas):
#             mission_x_suas[(i, j)] = solver.IntVar(0, num_pilots - 1, "Mission {%d} and SUAS {%d}" % (i, j)) 
#     pilots_flat = [mission_x_suas[(i, j)] for i in range(num_missions) for j in range(num_suas)] 
#     # print pilot_x_suas
#     # print mission_x_suas
    
    
#     #Constraint to ensure integrity
#     for suas in range(num_suas):
#         pilots_for_suas = [mission_x_suas[(j, suas)] for j in range(num_missions)]
#     for j in range(num_pilots):
#         m = pilot_x_suas[(j, suas)]
#         solver.Add(m.IndexOf(pilots_for_suas) == j)    
#     # Constraint of Type and Suas
#     for j in range(num_missions):
#         m_type = mission_mapping[j][1]
#         for i in range(num_suas):
#             model =  suas_mapping[i][0]
#             if (m_type,model) not in data[1]:
#                 k = 0
#                 while (k < num_pilots):
#                     solver.Add(mission_x_suas[(j,i)] != k)
#                     k = k+1
#     # Constraint of Qualification
#     for j in range(num_missions):
#         for i in range(num_suas):
#             model =  suas_mapping[i][0]
#             # model_index = data[0].index(model)
#             list_of_pilots = [k for k,v in data[2].iteritems() if v[model][0] == '1']
#             new_list = []
#             for pilot in pilot_mapping.items():
#                 if pilot[1] in list_of_pilots:
#                     new_list.append(pilot[0])
#             solver.Add(solver.IsMemberCt(mission_x_suas[(j,i)],new_list))
#     # Constraint for maximum 3 mission assignments to 
#     for i in range(num_suas):
#         for j in range(num_missions):
#             k = 0
#             count = 1
#             while (k < num_pilots and count <= 3):
#                 solver.Add(solver.IsEqualCstVar(mission_x_suas[(j,i)],k))
#                 k=k+1
#                 count+=1     
#     # Create the decision builder.
#     db = solver.Phase(pilots_flat, solver.CHOOSE_FIRST_UNBOUND,solver.ASSIGN_MIN_VALUE)
#     # Create the solution collector.
#     solution = solver.Assignment()
#     solution.Add(pilots_flat)
#     collector = solver.AllSolutionCollector(solution)
#     timelimit = solver.TimeLimit(60000)
#     solver.Solve(db, [collector,timelimit])
#     print("Solutions found:", collector.SolutionCount())
#     print("Time:", solver.WallTime(), "ms")
#     # Display a few solutions picked at random.
#     a_few_solutions = [0]
#     for sol in a_few_solutions:
#         print("Solution number" , sol, '\n')

#         for i in range(num_suas):
#             print("Suas", i)
#             for j in range(num_pilots):
#                 print("Pilot", j, "assigned to mission",collector.Value(sol, pilot_x_suas[(j, i)]))
#         print()
#     return true
# def solveCostMatrixUsingOrtools(data):
#     num_models = len(data[0])
#     num_types = len(data[1])
#     num_pilots = len(data[2])
#     num_missions = len(data[4])
#     num_suas = sum(data[3].values())
#     model_number = data[3]
#     pair_list = []
#     suas_mapping = {}
#     mission_mapping = {}
#     pilot_mapping = {}
#     index = 0
#     for i in range(num_models):
#         count = 0        
#         while(count < data[3].values()[i]):
#             suas_mapping[index] = (data[3].keys()[i],count)
#             count = count+1
#             index =index+1
#     # print suas_mapping 
#     for i in range(num_missions):
#         mission_mapping[i] = data[4][i]
#     # print mission_mapping
#     for i in range(num_pilots):
#         pilot_mapping[i] = data[2].keys()[i]
#     # print pilot_mapping
#     pair_list = []
#     for i in range(num_pilots):
#         pilot_dict = data[2].get(pilot_mapping.get(i))
#         for j in suas_mapping.keys():
#             t1 = suas_mapping[j]
#             t2 = pilot_dict[t1[0]]
#             cost = 0
#             if int(t2[0]):
#                 cost += 5
#                 if (t2[1] == 'Yes'):
#                     cost += 2
#                 else:
#                     cost -= 2
#                 pair_list.append((pilot_mapping.get(i),t1[0],t1[1],cost))
#                 # data[3].get()
#             else:
#                 pair_list.append((pilot_mapping.get(i),t1[0],t1[1],'NA'))
#     # print len(pair_list)
#     cost_matrix = []
#     for p in pair_list:
#         p_list = []
#         for m in mission_mapping.keys():
#             m_type = mission_mapping[m][1]
#             model = p[1]
#             if (m_type,model) in data[1]:
#                 if p[3] != 'NA':
#                     p_list.append(p[3]+8)
#                 else:
#                     p_list.append('NA')
#             else:
#                 p_list.append('NA')
#         cost_matrix.append(p_list)
#     cost_matrix = np.array(cost_matrix)
#     print cost_matrix
#     rows = cost_matrix.shape[0]
#     cols = cost_matrix.shape[1]
#     solver = pywrapgraph.LinearSumAssignment()
#     for pair in range(rows):
#         for mission in range(cols):
#             if cost_matrix[pair][mission] != 'NA':
#                 solver.AddArcWithCost(pair,mission,int(cost_matrix[pair][mission]))
#     solve_status = solver.Solve()
#     if solve_status == solver.OPTIMAL:
#         print "Total Cost ="+str(solver.OptimalCost())
#         for i in range(0,solver.NumNodes()):
#             pair = pair_list[i]
#             fav = 'Yes' if (solver.AssignmentCost(i) == 7) else 'No'
#             print "M"+mission_mapping[solver.RightMate(i)][0]+" "+pair[1]+" "+pair[0]+" "+fav
            
#             # print "Pilot "+pair[0]+"with SUAS of model"+" assigned  to Mission "+str(solver.RightMate(i))+" at cost "+str(solver.AssignmentCost(i))
#     elif solve_status == solver.INFEASIBLE:
#         print "No assignment possible"
#     elif solve_status == solver.POSSIBLE_OVERFLOW:
#         print "Some input costs are too large and may cause integer overflow"

  