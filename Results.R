# -------------- K-Means Clustering results -----------

# --------------- Working directory ------------
setwd("~/Desktop/UPC/ODS/Tareas/K-median")

# -------------- Libraries --------------
library(readr)
library(ggplot2)
library(Hmisc) #escribe latex

# --------------- Data loading --------------
data = read.table("~/Desktop/UPC/ODS/Tareas/K-median/cluster.txt", quote="\"")
tiempo = read_delim("tiempo.txt", "=", escape_double = FALSE, 
                    col_names = FALSE, trim_ws = TRUE)

# --------------- clustering data structuring -------------
clusteropt_500 = as.numeric(factor(data[7655:8154, 3])) #codifica los cluster usando factores
clusterpackage_500 = kmeans$cluster #obtiene los cluster desde la heuristica
clusterPRIM_500 = prim_clus$cluster #obtiene los cluster desde la heuristica MST PRIM
clusterset = data.frame(opt = clusteropt_500, heuristic = clusterpackage_500, mst = clusterPRIM_500)

# -------------- Grafica los cluster (comparacion) -------------
par(mfrow = c(1, 3), font = 2, font.lab = 4, font.axis = 2, las = 1, pch = 16)
(k_1 = clusplot(dataset, clusteropt_500, lines = 0, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE, span = TRUE,
         main = paste('BILP model'), xlab = 'PC1', ylab = 'PC2'))
(k_2 = clusplot(dataset, clusterpackage_500, lines = 0, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE, span = TRUE,
               main = paste('K-means'), xlab = 'PC1', ylab = 'PC2'))
(k_3 = clusplot(dataset, clusterPRIM_500, lines = 0, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE, span = TRUE,
                main = paste('Minimum spanning tree clustering'), xlab = 'PC1', ylab = 'PC2'))

# -------------- Cuenta filas duplicadas para ver cuantas veces coincide el opt con la heuristica
hits_heuristic = sum(duplicated(clusterset[, c("opt","heuristic")]))/nrow(clusterset[, c("opt","heuristic")])
hits_mst = sum(duplicated(clusterset[, c("opt","mst")]))/nrow(clusterset[, c("opt","mst")])

# -------------- Time data structuring -------------
observations = tiempo[tiempo[,1] == "n", 2]
kcluster = tiempo[tiempo[,1] == "q", 2]
time = tiempo[tiempo[,1] != "n" & tiempo[,1] != "q", 2]

measures = cbind(observations, kcluster, time)
colnames(measures) = c("N", "K", "t")

# -------------- exporta codigo tabla latex ------------
measures_latex = latex(measures, file="")

# --------------- make graphics ---------------
g_t = ggplot(measures, aes(N, t, group=as.factor(K), colour=as.factor(K))) + theme_classic() + geom_line() + geom_point() +
  ylab("time (s)") + xlab("Data size (# observations)") + ggtitle("") + labs(colour = "# cluster") + theme(text=element_text(size=14),legend.position="bottom")
ggsave("Tiempos.png", width = 18, height = 12, units = "cm")
