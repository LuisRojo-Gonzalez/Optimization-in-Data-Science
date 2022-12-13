library(readr)
library(ggplot2)

setwd("~/Desktop/UPC/ODS/Tareas/DataProtection")

data <- read_table2("AMPL/solucion.txt", 
                    col_names = FALSE)

resultados = data.frame()
for (i in 1:(nrow(data)/4)) {
  for (j in 1:4) {
    resultados[i,j] = data[(i-1)*4+j,3]
  }
}
colnames(resultados) = c("Iteration","Cut","Master_Problem","Sub_Problem")

p = ggplot() + geom_line(aes(y = Cut, x = Iteration, colour = "Cut"),
                          data = resultados, stat="identity")
p = p + geom_line(aes(y = Master_Problem, x = Iteration, colour = "Master Problem"),
                         data = resultados, stat="identity")
p = p + geom_line(aes(y = Sub_Problem, x = Iteration, colour = "Sub Problem"),
                  data = resultados, stat="identity") +
  theme_bw() + theme(text=element_text(size=15),legend.position="bottom") +
        labs(title="",x="Iteration", y = "Value") + labs(colour = "")
#ggsave("Small.png", width = 5, height = 4)
#ggsave("Example2D.png", width = 5, height = 4)
ggsave("Targus.png", width = 5, height = 4)
print(p)
