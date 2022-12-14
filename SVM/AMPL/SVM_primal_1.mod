param ntrain; #define numero de observaciones para entrenamiento

#Conjuntos
set rows := {1 .. 1000}; # i: data observations
set columns := {1 .. 4}; # p: variables independientes

#Variables
var w {j in columns}; #combinacion lineal intercepto
var B; #Intercepto
var e {i in rows} >= 0; #error clasificacion

#Declaracion de parametros
param d {i in rows}; #clasificacion real
param A {(i,j) in {rows, columns}}; #datos independientes
param C; #penalizacion clasificacion
param aux; #contador para determinar el error de clasificacion

#Objective function
minimize z : 0.5*(sum{j in columns} w[j]^2) + C*(sum{i in rows: i <= ntrain} e[i]);

#------------------------------Restricciones--------------------------

subject to rest1 {i in rows: i <= ntrain}: d[i]*(sum{j in columns} w[j]*A[i,j] + B) >= 1-e[i];
