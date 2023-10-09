# This file is used to solve the optimization problem, run this command:
# python optimality.py $dataset $N $formulation_type $break_symmetry $seed_gnn $seed_gurobi
# $dataset is the name of dataset (QM7 or QM9) 
# $N is the number of atoms (4,5,6,7,8) 
# $formulation_type denotes the type of formulation (0 for bi-linear, 1 for big-M)
# $break_symmetry is binary and indices adding symmetry-breaking constraints
# $seed_gnn is the random seed for traing GNN
# $seed_gurobi is the random seed of Gurobi

# Note: please first train and transform a GNN to ONNX format.

from omlt.io import load_onnx_neural_network_with_bounds
import math
import sys

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os

dataset = str(sys.argv[1]) # dataset
N = int(sys.argv[2]) # number of atoms
formulation_index = int(sys.argv[3]) # index for formulation types
break_symmetry = int(sys.argv[4]) # breaking symmetry or not
# BS = 0: (S1)
# BS = 1: (S1)+(S2)+(S3)
seed_gnn = int(sys.argv[5]) # random seed for training
seed_gurobi = int(sys.argv[6]) # random seed for Gurobi

if formulation_index == 0:
    formulation_type = 'bi-linear'
elif formulation_index == 1:
    formulation_type = 'big-M'

# initialize parameters and bound for both datasets
if dataset == 'QM7':
    Atom = ['C', 'N', 'O', 'S']
    Cov = [4, 3, 2, 2]
    ub = [None, max(1, (3*N)//7), max(1, N//3), max(1, N//7)]
    lb = [math.ceil(N/2), None, None, None]
    ub_ring = N//2
    ub_db = N//2
    ub_tb = N//2
    
elif dataset == 'QM9':
    Atom = ['C','N', 'O', 'F']
    Cov = [4, 3, 2, 1]
    ub = [None, (3*N)//5, (4*N)//7, (4*N)//5]
    lb = [math.ceil(N/5), None, None, None]
    ub_ring = (2*N)//3
    ub_db = N//2
    ub_tb = N//2

# load the Dense NN
model_saved = f'Dense_models/{dataset}/N={N}/Dense_{seed_gnn}.onnx'

network_definition = load_onnx_neural_network_with_bounds(model_saved)

import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.neuralnet import ReluBigMFormulation

for layer_id, layer in enumerate(network_definition.layers):
    print(f"{layer_id}\t{layer}\t{layer.activation}")

formulation = ReluBigMFormulation(network_definition)

# build MIP for CAMD
def build_molecule_formulation(m):
    m.N  = N
    m.F = 16
    m.Nt = 4
    m.Nn = 5
    m.Nh = 5
    m.It = range(0,4)
    m.In = range(4,9)
    m.Ih = range(9,14)
    m.Idb = 14
    m.Itb = 15
    m.Atom = Atom
    m.Cov = Cov

    m.X = pyo.Var(pyo.Set(initialize=range(m.N)), pyo.Set(initialize=range(m.F)), within=pyo.Binary)
    m.A = pyo.Var(pyo.Set(initialize=range(m.N)), pyo.Set(initialize=range(m.N)), within=pyo.Binary)
    m.DB = pyo.Var(pyo.Set(initialize=range(m.N)), pyo.Set(initialize=range(m.N)), within=pyo.Binary)
    m.TB = pyo.Var(pyo.Set(initialize=range(m.N)), pyo.Set(initialize=range(m.N)), within=pyo.Binary)

    # constraints for adjacency matrix
    m.Con_A = pyo.ConstraintList()

    m.Con_A.add((m.A[0,0] == 1))
    m.Con_A.add((m.A[1,1] == 1))
    m.Con_A.add((m.A[0,1] == 1))
    # for v in range(m.N-1):
     #    m.Con_A.add((m.A[v,v] >= m.A[v+1,v+1]))
    for v in range(2,m.N):
        m.Con_A.add((m.A[v,v] == 1))
    for u in range(m.N):
        for v in range(u+1,m.N):
            m.Con_A.add((m.A[u,v] == m.A[v,u]))
    for v in range(m.N):
        expr = (m.N - 1) * m.A[v,v]
        for u in range(m.N):
            if u != v:
                expr -= m.A[u,v]
        m.Con_A.add(expr >= 0)
    for v in range(2, m.N):
        expr = m.A[v,v]
        for u in range(v):
            expr -= m.A[u,v]
        m.Con_A.add(expr <= 0)
        
    expr = - (m.N - 1)
    for u in range(m.N):
        for v in range(u+1,m.N):
            expr += m.A[u,v]
    m.Con_A.add(expr <= ub_ring)
    #m.Con_A.pprint()

    # constraints for bonds, including double and triple bonds
    m.Con_B = pyo.ConstraintList()

    for v in range(m.N):
        m.Con_B.add((m.DB[v,v] == 0))
    for u in range(m.N):
        for v in range(u+1,m.N):
            m.Con_B.add((m.DB[u,v] == m.DB[v,u]))
    for v in range(m.N):
        m.Con_B.add((m.TB[v,v] == 0))
    for u in range(m.N):
        for v in range(u+1,m.N):
            m.Con_B.add((m.TB[u,v] == m.TB[v,u]))
    for u in range(m.N):
        for v in range(u+1,m.N):
            m.Con_B.add((m.DB[u,v] + m.TB[u,v] <= 1))
    
    expr = 0.
    for u in range(m.N):
        for v in range(u+1,m.N):
            expr += m.DB[u,v]
    m.Con_B.add(expr <= ub_db)
    
    expr = 0.
    for u in range(m.N):
        for v in range(u+1,m.N):
            expr += m.TB[u,v]
    m.Con_B.add(expr <= ub_tb)
    #m.Con_B.pprint()

    # constraints linking features and adjacency matrix
    m.Con_X_A = pyo.ConstraintList()

    for v in range(m.N):
        expr = m.A[v,v]
        for f in m.It:
            expr -= m.X[v,f]
        m.Con_X_A.add(expr == 0)
    for v in range(m.N):
        expr = m.A[v,v]
        for f in m.In:
            expr -= m.X[v,f]
        m.Con_X_A.add(expr == 0)
    for v in range(m.N):
        expr = m.A[v,v]
        for f in m.Ih:
            expr -= m.X[v,f]
        m.Con_X_A.add(expr == 0)
    for v in range(m.N):
        expr = 0.
        for u in range(m.N):
            if u != v:
                expr += m.A[u,v]
        for i in range(m.Nn):
            expr -= i * m.X[v,m.In[i]]
        m.Con_X_A.add(expr == 0)
    # m.Con_X_A.pprint()

    # constraints for covalence
    m.Con_X_A_B = pyo.ConstraintList()
    for u in range(m.N):
        for v in range(u+1,m.N):
            m.Con_X_A_B.add((3. * m.DB[u,v] - m.X[u,m.Idb] - m.X[v,m.Idb] - m.A[u,v] <= 0))
    for u in range(m.N):
        for v in range(u+1,m.N):
            m.Con_X_A_B.add((3. * m.TB[u,v] - m.X[u,m.Itb] - m.X[v,m.Itb] - m.A[u,v] <= 0))
    # m.Con_X_A_B.pprint()

    # constraints linking features and bonds
    m.Con_X_B = pyo.ConstraintList()
    for v in range(m.N):
        expr = 0.
        for u in range(m.N):
            if u != v:
                expr += m.DB[u,v]
        for i in range(m.Nt):
            expr -= (m.Cov[i] // 2) * m.X[v,m.It[i]]
        m.Con_X_B.add(expr <= 0)
    for v in range(m.N):
        expr = m.X[v,m.Idb]
        for u in range(m.N):
            if u != v:
                expr -= m.DB[u,v]
        m.Con_X_B.add(expr <= 0)
    for v in range(m.N):
        expr = 0.
        for u in range(m.N):
            if u != v:
                expr += m.TB[u,v]
        for i in range(m.Nt):
            expr -= (m.Cov[i] // 3) * m.X[v,m.It[i]]
        m.Con_X_B.add(expr <= 0)
    for v in range(m.N):
        expr = m.X[v,m.Itb]
        for u in range(m.N):
            if u != v:
                expr -= m.TB[u,v]
        m.Con_X_B.add(expr <= 0)
    for v in range(m.N):
        expr = 0.
        for i in range(m.Nt):
            expr += m.Cov[i] * m.X[v,m.It[i]]
        for i in range(m.Nn):
            expr -= i * m.X[v,m.In[i]]
        for i in range(m.Nh):
            expr -= i * m.X[v,m.Ih[i]]
        for u in range(m.N):
            if u != v:
                expr -= m.DB[u,v]
        for u in range(m.N):
            if u != v:
                expr -= 2. * m.TB[u,v]
        m.Con_X_B.add(expr == 0)
    # m.Con_X_B.pprint()

    # constraints for features
    m.Con_X = pyo.ConstraintList()
    for i in range(m.Nt):
        expr = 0.
        for v in range(m.N):
            expr += m.X[v,m.It[i]]
        if lb[i] is not None:
            m.Con_X.add(expr >= lb[i])
        if ub[i] is not None:
            m.Con_X.add(expr <= ub[i])
    # m.Con_X.pprint()


    # constriants for orders
    m.Con_O = pyo.ConstraintList()
    if break_symmetry:
        coef_1 = [2**i for i in range(m.F-1,-1,-1)]
        # print(coef_1)
        coef_2 = [2**i for i in range(m.N-1,-1,-1)]
        # print(coef_2)
        for v in range(1,m.N):
            expr = 0.
            for f in range(m.F):
                expr += coef_1[f] * m.X[0,f]
            for f in range(m.F):
                expr -= coef_1[f] * m.X[v,f]
            expr -= (2**m.F) * (1. - m.A[v,v])
            m.Con_O.add(expr <= 0)

        for v in range(1,m.N-1):
            expr = 0.
            for u in range(m.N):
                if u!= v and u != v+1:
                    expr += coef_2[u] * m.A[u,v]
            for u in range(m.N):
                if u != v and u != v+1:
                    expr -= coef_2[u] * m.A[u,v+1]
            m.Con_O.add(expr >= 0)
    # m.Con_O.pprint()

m = pyo.ConcreteModel()
m.nn = OmltBlock()

m.nn.gnn_layers = [1,2] # specify the indices for GNN layers
m.nn.formulation_type = formulation_type # specify the formulation type used for GNN layers

build_molecule_formulation(m.nn)

m.nn.build_formulation(formulation) # build MIP for GNN

# connect atom features to inputs of Dense NN
m.Con_Input = pyo.ConstraintList()
for v in range(m.nn.N):
    for f in range(m.nn.F):
        m.Con_Input.add((m.nn.X[v,f] == m.nn.inputs[v*m.nn.F+f]))
# m.Con_Input.pprint()

m.obj = pyo.Objective(expr=(m.nn.outputs[0])) # the output of GNN is the objective

# m.pprint()

from gurobipy import GRB
import numpy as np
import os

opt = pyo.SolverFactory('gurobi_persistent') # load Gurobi as the solver
opt.set_instance(m)
opt.set_gurobi_param('Seed', seed_gurobi) # set random seed for Gurobi
opt.set_gurobi_param('TimeLimit', 36000) # set the time limit (10 hours)

# callback function used to save each time that better solution is found
cb_times = []
cb_sols = []
def my_callback(cb_m, cb_opt, cb_where):
    if cb_where == GRB.Callback.MIPSOL:
        cb_times.append(cb_opt.cbGet(GRB.Callback.RUNTIME))
        cb_sols.append(cb_opt.cbGet(GRB.Callback.MIPSOL_OBJ))

opt.set_callback(my_callback)

result = opt.solve(tee=True)

print(cb_times)
print(cb_sols)

# retrieve the first time to find the optimal solution
for i in range(len(cb_times)):
    if np.abs(cb_sols[i]-cb_sols[-1]) < 1e-9:
        opt_time = cb_times[i]
        break

# print the results
print(opt_time)
print(result.Solver.Wallclock_time)
print(result.Problem.Upper_bound)
print(result.Problem.Lower_bound)

# save the results
result_saved = np.array([opt_time, result.Solver.Wallclock_time, result.Problem.Upper_bound, result.Problem.lower_bound])
run_idx = seed_gnn * 10 + seed_gurobi
filename = f'run_{run_idx}'
folder = f'Results/optimality/{dataset}/N={N}/{formulation_type}/BS={break_symmetry}/'
os.makedirs(folder, exist_ok=True)
np.save(folder + filename, result_saved)