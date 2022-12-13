
# ----------------------------------------
# LOCATION-TRANSPORTATION PROBLEM
# USING BENDERS DECOMPOSITION
# (using dual formulation of subproblem)
# ----------------------------------------

# Parameters

param ncells; # quantity of cells to work with (34)
param npcells; # quantity of 1 .. npcells cells to work with (4)
param nconstraints; # number of constraint to consider in A matrix (10)
param nnz; # cardinality of cells to consider in A matriz

param a {1 .. ncells}; # value of cell
param lb {1 .. ncells}; # lower bound of cell protection
param ub {1 .. ncells}; # upper bound of cell protection
param c {1 .. ncells}; # penalty of supress cells
param is_p {1 .. ncells}; # 1 if 1 .. ncellservation is 1 .. npcells
param A {1 .. nconstraints, 1 .. ncells} default 0; # matrix of coef
param b {1 .. nconstraints};

param p {1 .. npcells};
param plpl {1 .. npcells}; # niveles inferiores de celdas sensibles
param pupl {1 .. npcells}; # niveles superiores de celdas sensibles

param coef {1 .. nnz}; # suma o resta en la fila
param xcoef {1 .. nnz}; # que indice suma o resta

param begconst {1 .. (nconstraints+1)};

### ------------ SUBPROBLEM --------- ###

var lambda {1 .. nconstraints}; #### REVISAR ESTO LUEGO
var mu_l {1 .. ncells} >= 0; # 1 .. nnz duales cota inferior
var mu_u {1 .. ncells} >= 0; # 1 .. nnz duales cota superior

param y {1 .. ncells} binary; # 1 if cell is supressed

# Lower bound problem
maximize lower_bound_dual:
    sum{i in 1 .. ncells} (mu_l[i]*(lb[i]-a[i])-mu_u[i]*(ub[i]-a[i]))*y[i] +
    sum{i in 1 .. npcells} plpl[i];

subject to r1 {i in 1 .. ncells}:
    sum{j in 1 .. nconstraints} (lambda[j]*A[j,i]) + mu_l[i] - mu_u[i] = is_p[i];

subject to r2 {j in 1 .. npcells}:
    sum{i in 1 .. ncells} (mu_l[i]*(lb[i]-a[i])-mu_u[i]*(ub[i]-a[i]))*y[i] <= pupl[j];

subject to r10 {j in 1 .. npcells}:
    sum{i in 1 .. ncells} (mu_l[i]*(lb[i]-a[i])-mu_u[i]*(ub[i]-a[i]))*y[i] >= -plpl[j];

### ----------- MASTER PROBLEM ---------- ###

param nCUT >= 0 integer;
param cut_type {1..nCUT} symbolic within {"point","ray"};

#param LAMBDA {1 .. nconstraints, 1 .. nCUT}; #### REVISAR ESTO LUEGO
param MU_l {1 .. ncells, 1 .. nCUT}; # 1 .. nnz duales cota inferior
param MU_u {1 .. ncells, 1 .. nCUT}; # 1 .. nnz duales cota superior
#param x {1 .. ncells, 1 .. nCUT};

var Y {1 .. ncells} binary; # 1 if cell cell is supressed
var Max_bound;

minimize FO: sum{i in 1 .. ncells} c[i]*Y[i] + Max_bound;

subject to r4 {i in 1 .. npcells}: Y[p[i]] = 1;

subject to Cut {k in 1 .. nCUT}:
    if cut_type[k] = "point" then Max_bound >= sum{i in 1 .. ncells} (MU_l[i,k]*(lb[i]-a[i])-MU_u[i,k]*(ub[i]-a[i]))*Y[i];
