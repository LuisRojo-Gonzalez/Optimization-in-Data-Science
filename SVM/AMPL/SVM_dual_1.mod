param ntrain; #define numero de observaciones para entrenamiento

#Conjuntos
set rows := {1 .. 1000}; # i: data observations
set columns := {1 .. 4}; # p: variables independientes

#Declaracion de parametros
param d {i in rows}; #clasificacion real
param A {(i,j) in {rows, columns}}; #datos independientes
param K {(i,j) in {rows, rows}: i <= ntrain and j <= ntrain}; # Kernel
param C; #penalizacion clasificacion
param aux; #contador para determinar el error de clasificacion
param w {j in columns}; #define el plano separador
param B; #intercepto del plano

#Variables
var lambda {i in rows: i <= ntrain} >= 0; #multiplicadores de Lagrange

#Objective function
maximize z : sum{i in rows: i <= ntrain} lambda[i] - 0.5*(sum{(i,j) in {rows, rows}: i <= ntrain and j <= ntrain} lambda[i]*d[i]*lambda[j]*d[j]*K[i,j]);

#------------------------------Restricciones--------------------------

subject to rest1: sum{i in rows: i <= ntrain} lambda[i]*d[i] = 0;

subject to rest2 {i in rows: i <= ntrain}: lambda[i]  <= C;
