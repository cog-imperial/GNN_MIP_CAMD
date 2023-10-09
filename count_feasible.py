# This file is used to count the number of all feasible solutions given N, run this command:
# python count_feasible.py $dataset $N $break_symmetry
# $dataset is the name of dataset (QM7 or QM9)
# $N is the number of atoms
# $break_symmetry controls the level of breaking symmetry

# Note: this file is independent with GNNs. One can try it without preprocessing datasets and training GNNs.

import math
import sys

dataset = str(sys.argv[1]) # dataset
N = int(sys.argv[2]) # number of atoms
break_symmetry = int(sys.argv[3]) # level of breaking symmetry
# BS = 0: (S1)
# BS = 1: (S1)+(S2)
# BS = 2: (S1)+(S2)+(S3)
# (S1),(S2),(S3) are symmetry-breaking constraints in the paper

print('N = ', N, 'BS = ', break_symmetry)

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

import pyomo.environ as pyo
import gurobipy

# build MIP for CAMD
def build_molecule_formulation(m, N, level):
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
    coef_1 = [2**i for i in range(m.F-1,-1,-1)]
    # print(coef_1)
    coef_2 = [2**i for i in range(m.N-1,-1,-1)]
    # print(coef_2)
    if level > 0:
        for v in range(1,m.N):
            expr = 0.
            for f in range(m.F):
                expr += coef_1[f] * m.X[0,f]
            for f in range(m.F):
                expr -= coef_1[f] * m.X[v,f]
            expr -= (2**m.F) * (1. - m.A[v,v])
            m.Con_O.add(expr <= 0)
    
    if level > 1:
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
build_molecule_formulation(m, N, break_symmetry)

print('number of variables: ', m.N*m.F+3*m.N*m.N)
print('number of constraints: ', len(m.Con_A)+len(m.Con_B)+len(m.Con_X_A)+len(m.Con_X_A_B)+len(m.Con_X_B)+len(m.Con_X)+len(m.Con_O))

m.obj = pyo.Objective(rule=0.) # set a constrant objective
opt = pyo.SolverFactory('gurobi_persistent') # load Gurobi as the solver
opt.set_instance(m)
opt.set_gurobi_param('TimeLimit', 172800) # set the time limit (48 hours)
opt.set_gurobi_param('PoolSolutions', 1000000000) # set the size of solution pool
opt.set_gurobi_param('PoolSearchMode', 2) # set the search mode to find all feasible solutions
results = opt.solve(m, tee=True)