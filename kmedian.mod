#Hiperparametros
param nrow;

#Conjuntos
set rows := {1 .. nrow}; # i: data observations
set columns := {1 .. 4}; # p: variables

#Variables
var w {(i,j) in {rows,rows}} binary; #pertenencia del elemento i a cluster j

#Declaracion de parametros
param x {(i,p) in {rows, columns}}; #datos
param d {(i,j) in {rows,rows}}; #distancia
param k; #cluster

#Objective function
minimize z : sum {(i,j) in {rows,rows}} w[i,j]*d[i,j];

#------------------------------Restricciones--------------------------

subject to restriccion1 {i in rows}: sum {j in rows} w[i,j] = 1;

subject to restriccion2 : sum {i in rows} w[i,i] = k;

subject to restriccion3 {(i,j) in {rows,rows}}: w[j,j] >= w[i,j];
