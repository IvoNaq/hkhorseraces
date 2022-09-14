rm(list = ls())
#load("TP2.rds")
# --------     Paquetes      -------#

library(glmnet)
library(rpart)
library(tidyverse)
library(dplyr)
library(DMwR2)
library(performanceEstimation)
library(randomForest)
library(xgboost)
library(caret)
library(pROC)

options(scipen=999)

# --------     Carga de datasets      -------#
setwd('~/UTDT_MiM/UTDT - ML')

races <- read.table('data/races.csv', sep =',', dec = '.', header = T)
#head(races,2); tail(races,2); dim(races) # 6349 x 37

runs <- read.table('data/runs.csv', sep =',', dec = '.', header = T)
#head(runs,8); tail(runs,2); dim(runs) # 79447 x 37

# Merge de tablas en un solo data-frame:
df_list <- list(runs, races) 
raceruns <- df_list %>% reduce(full_join, by='race_id')

# Eliminamos última fila generada de forma redundante
raceruns <- raceruns[-79448,]

# Borramos información duplicada en la memoria.
rm(list = c('races','runs','df_list')) 

#-------------------------------------------------------------------------#
# --------     EDA      -------#
# 
# # Analisis Exploratorio de datos
# 
# dim(raceruns)
 
# comienzo = min(raceruns$date)
# finalizacion = max(raceruns$date)
# numero_caballos = length(unique(raceruns$horse_id))
# numero_carreras = length(unique(raceruns$race_id))
# 
# x11(); par(mfrow = c(1,2))
# # edad caballos. 80% posee 3 años
# barplot(prop.table(table(raceruns$horse_age)), ylim = c(0,1), ylab ='% Caballos', xlab = "Edad")
# 
# desbalanceo de clases, el % de win es muy bajo
# raceruns_win <- raceruns %>%
#   filter(won == 1)
# 
# unique(raceruns_win$won)
# 
# # edad caballos ganadores. 80% posee 3 años, aumento un poco mas 4 años
# barplot(prop.table(table(raceruns_win$horse_age)), ylim = c(0,1), ylab ='% Caballos', xlab = "Edad")
# 
# 
# # el 0.72% no posee el dato por lo que transformamos esta variable en yes/no
# prop.table(table(raceruns$horse_gear))
# raceruns$horse_gear <- ifelse(raceruns$horse_gear == '--','no','yes')
# 
# raceruns$won[raceruns$won == 1] <- "yes"
# raceruns$won[raceruns$won == 0] <- "no"
# 
# # Proporción que representa cada categoría (sólo features categóricos)
# features_categoricos <- sapply(raceruns, function(x) class(x) == "character")
# sapply(raceruns[, features_categoricos], function(x) prop.table(table(x)))
# 
# # Formator long para graficar
# data_plot <- raceruns[, features_categoricos] %>%
#   pivot_longer(cols = setdiff(everything(), one_of("won")),
#                names_to = "Feature",
#                values_to = "Value") %>%
#   group_by(Feature, Value, won) %>%
#   tally() %>%
#   ungroup() %>%
#   group_by(Feature, Value) %>%
#   mutate(Proportion = n/sum(n)) %>%
#   ungroup()
# 
# ggplot(data = data_plot) +
#   geom_bar(aes(x = Value, y = Proportion, fill = won),
#            stat = "identity") +
#   facet_wrap(~Feature, scales = "free") +
#   theme_minimal() +
#   theme(legend.position = "bottom",
#         plot.title = element_text(face = "bold")) +
#   scale_fill_manual(values = c("#73777B", "#EC994B")) +
#   labs(title = "Ganadores para cada categoría",
#        subtitle = "Features categóricos") +
#   coord_flip()
# 
# 
# # Qué tan bien separan las clases nuestros features numéricos
# features_numericos <- sapply(raceruns, function(x) class(x) == "numeric")
# features_numericos["won"] <- TRUE # (para conservarlo)
# 
# data_plot <- raceruns[, features_numericos] %>%
#   pivot_longer(cols = setdiff(everything(), one_of("won")),
#                names_to = "Feature",
#                values_to = "Value")
# 
# ggplot(data = data_plot) +
#   geom_density(aes(x = Value, fill = won), alpha = 0.8, adjust = 2) +
#   facet_wrap(~Feature, scales = "free") +
#   theme_minimal() +
#   theme(legend.position = "bottom",
#         plot.title = element_text(face = "bold")) +
#   scale_fill_manual(values = c("#73777B", "#EC994B")) +
#   labs(title = "Separabilidad de clases",
#        subtitle = "Features numéricos")
# 


# --------     Preprocesamiento de datos      -------#

raceruns$date <- as.Date(raceruns$date) 
raceruns$surface    <- as.factor(raceruns$surface)    
raceruns$race_class <- as.factor(raceruns$race_class) 
raceruns$draw   <- as.factor(raceruns$draw)           

# % de obs con valores faltantes
cat(
  round((1-nrow(raceruns %>% na.omit())/nrow(raceruns))*100, 3),
  "% de las obs con valores faltantes en algún feature"
)

#Corregimos prize y place odds con la media

impute_mean <- function(x) {ifelse(is.na(x), mean(x, na.rm = TRUE), x)}

raceruns <- raceruns %>% mutate(prize = impute_mean(prize))
raceruns <- raceruns %>% mutate(place_odds = impute_mean(place_odds))

raceruns$horse_gear <- ifelse(raceruns$horse_gear == '--','no','yes')

sapply(raceruns, function(x) prop.table(table(missing=is.na(x))))

#subset de varias columnas -> Nos quedamos con las top 5 categorías y "Other"
#horse_ratings
horse_ratings_subset <- raceruns %>%
  group_by(horse_ratings) %>%
  tally() %>%
  arrange(desc(n)) %>% 
  head(5) %>%
  pull(horse_ratings)

raceruns$horse_ratings <- ifelse(raceruns$horse_ratings %in% horse_ratings_subset,
                                 raceruns$horse_ratings,
                                 "OTHER")
#horse_country
horse_country_subset <- raceruns %>%
  group_by(horse_country) %>%
  tally() %>%
  arrange(desc(n)) %>% 
  head(5) %>%
  pull(horse_country)


raceruns$horse_country <- ifelse(raceruns$horse_country %in% horse_country_subset,
                                 raceruns$horse_country,
                                 "OTHER")

#Ratio de actual_weight vs declared_weight para capturar información de peso de jinete
raceruns$weight_ratio = raceruns$actual_weight/raceruns$declared_weight

race_id = unique(raceruns$race_id) # id único de cada carrera.

# Información dinámica de cada caballo/jinete en función de la carrera que está corriendo

#Edad
raceruns$max.age  <- rep(NA,dim(raceruns)[1]) # Distancia respecto del caballo más pesado. 
raceruns$min.age  <- rep(NA,dim(raceruns)[1]) # Distancia respecto del caballo más ligero.
raceruns$n_compet <- rep(NA,dim(raceruns)[1]) # Cuanto caballos compiten

for(i in race_id){
  sel = which(raceruns$race_id == i)
  raceruns$n_compet[sel]  <- length(sel)-1
  raceruns$max.age[sel]   <- max(raceruns$horse_age[sel],na.rm = T) - raceruns$horse_age[sel]
  raceruns$min.age[sel]   <- raceruns$horse_age[sel] - min(raceruns$horse_age[sel],na.rm = T)
}

#Peso
raceruns$max.weight  <- rep(NA,dim(raceruns)[1]) # Distancia respecto del caballo más pesado. 
raceruns$min.weight  <- rep(NA,dim(raceruns)[1]) # Distancia respecto del caballo más ligero.

for(i in race_id){
  sel = which(raceruns$race_id == i)
  raceruns$n_compet[sel]  <- length(sel)-1
  raceruns$max.weight[sel]   <- max(raceruns$declared_weight[sel],na.rm = T) - raceruns$declared_weight[sel]
  raceruns$min.weight[sel]   <- raceruns$declared_weight[sel] - min(raceruns$declared_weight[sel],na.rm = T)
}

#para evitar data leakage, partimos del set de entrenamiento
#trainer_id , jockey_id y horse_id  ganadores marca (solo train set)

train.id = which(raceruns[,1]<3001)
raceruns_train_set <- raceruns[train.id,]
raceruns_valid_set <- raceruns[-train.id,]
raceruns_win <- raceruns_train_set %>% filter(won == 1)

#top 10 jockey
jockey_winner_subset <- raceruns_win %>%
  group_by(jockey_id) %>%
  tally() %>%
  arrange(desc(n)) %>% 
  head(10) %>%
  pull(jockey_id)

raceruns$jockey_winner <- ifelse(raceruns$jockey_id %in% jockey_winner_subset, 1, 0)

#top 10 trainer
trainer_winner_subset <- raceruns_win %>%
  group_by(trainer_id) %>%
  tally() %>%
  arrange(desc(n)) %>% 
  head(10) %>%
  pull(trainer_id)

raceruns$trainer_winner <- ifelse(raceruns$trainer_id %in% trainer_winner_subset, 1, 0)

#top 10 horse
horse_winner_subset <- raceruns_win %>%
  group_by(horse_id) %>%
  tally() %>%
  arrange(desc(n)) %>% 
  head(10) %>%
  pull(horse_id)

raceruns$horse_winner <- ifelse(raceruns$horse_id %in% horse_winner_subset, 1, 0)

#Vuelvo a hacer separación para incorporar a los datasets estos nuevos features
train.id = which(raceruns[,1]<3001)
raceruns_train_set <- raceruns[train.id,]
raceruns_valid_set <- raceruns[-train.id,]
#dim(raceruns) #79447x82


# --------     Set entrenamiento y validacion       -------#

# Solo tomo algunas de las variables del data set ya que no cuento con mucha de la información antes de comenzar la carrera
X_train = raceruns_train_set[,c(1,5,7:10,12:14,34,35,41:45,74:82)]
X_valid = raceruns_valid_set[,c(1,5,7:10,12:14,34,35,41:45,74:82)]
X_train = na.omit(X_train)
X_valid = na.omit(X_valid)
race_id_train = unique(X_train[,1]) # Vuelvo a crear este vector porque eliminé filas
race_id_valid = unique(X_valid[,1]) # Vuelvo a crear este vector porque eliminé filas

# --------     Métricas de performance a usar       -------#

metricas <- function(conf_matrix) {
  accuracy <- sum(diag(prop.table(conf_matrix)))
  precision <- prop.table(conf_matrix, margin = 2)[2,2]
  recall <- prop.table(conf_matrix, margin = 1)[2,2]
  f1_score <- (2*precision*recall)/(precision+recall)
  
  print(paste("Accuracy:", round(accuracy, 3)))
  print(paste("Precision:", round(precision, 3)))
  print(paste("Recall:", round(recall, 3)))
  print(paste("F1 score:", round(f1_score, 3)))
}

# --------     Regresion Logística con Lasso regularizada       -------#
X_train_1 = model.matrix(~.,data = X_train)[ , -1]
X_valid_1 = model.matrix(~.,data = X_valid)[ , -1]

# Fitting lasso
grid.l =exp(seq(5 , -5 , length = 100))

cv.out = cv.glmnet( x=data.matrix(X_train_1[,-2]), 
                    y = as.matrix(X_train_1[, 2]), 
                    family  = 'binomial',
                    type.measure = 'deviance',
                    lambda = grid.l,  
                    alpha = 1,
                    nfolds = 5)
plot(cv.out)
bestlam = cv.out$lambda.1se #extraigo el lambda que minimiza la varianza

logit = glmnet( x=data.matrix(X_train_1[,-2]) ,
                y = as.matrix(X_train_1[, 2]), 
                family  = 'binomial',
                lambda = bestlam,  
                alpha = 1)

pred_0 <- predict(logit, s = bestlam , newx = X_valid_1[,-2], type = 'response')

# Indicadores
# Propongo que si la probabilidad del modelo logístico para una determinada carrera para un caballo es máxima, 
# ese caballo debería ganar. No considera umbrales de corte

winner.basic_logit <- data.frame(matrix(NA,
                                        nrow=dim(X_valid_1)[1],
                                        ncol = 4))
colnames(winner.basic_logit) <- c('race_id','prob_predicted','outcome_predicted','real_outcome')
                                 
for (j in 1:dim(winner.basic_logit)[1]){  
  race_selected = X_valid_1[j,1]
  winner.basic_logit[j,'race_id'] = race_selected
  winner.basic_logit[j,'prob_predicted'] = pred_0[j]
  winner.basic_logit[j,'real_outcome'] = X_valid_1[j,2]}

winner.basic_logit$prob_max <- rep(NA,dim(winner.basic_logit)[1])

for (j in 1:dim(winner.basic_logit)[1]){
  sel = which(winner.basic_logit[,'race_id']==winner.basic_logit$race_id[j])
  winner.basic_logit[j,'prob_max'] = max(winner.basic_logit$prob_predicted[sel])
}

winner.basic_logit <- winner.basic_logit %>% mutate(outcome_predicted = ifelse(round(prob_predicted,3)-round(prob_max,3)<0,0,1))

winner.basic_logit$winner_predicted <- rep(NA,dim(winner.basic_logit)[1])

for (j in 1:dim(winner.basic_logit)[1]){
  ifelse(winner.basic_logit$real_outcome[j]==1,
         ifelse(winner.basic_logit$outcome_predicted==1,
                winner.basic_logit$winner_predicted[j]<-1,
                winner.basic_logit$winner_predicted[j]<-0),
         winner.basic_logit$winner_predicted[j]<-0)
}
sum(winner.basic_logit$winner_predicted)
  
conf_matrix <- table(winner.basic_logit$real_outcome, winner.basic_logit$outcome_predicted)
metricas(conf_matrix)

# Área bajo la curva de ROC
roc(winner.basic_logit$real_outcome ~ winner.basic_logit$outcome_predicted, plot = TRUE, print.auc = TRUE, legacy.axes = TRUE)


# --------     Random Forest       -------#

X2 = raceruns_train_set[,c(1,5,7:10,12:14,34,35,41:45,74:82)]
sum(is.na(X2))
X2 = na.omit(X2) # Sería mejor tratar los datos faltantes.
race_id = unique(X2[,1]) # Como eliminamos algunas filas vuelvo a crear este vector.

X_valid_2 = raceruns_valid_set[,c(1,5,7:10,12:14,34,35,41:45,74:82)]

plot_classes <- function(y_test, y_pred) {
  ggplot(data = data.frame(y_test = factor(y_test), y_pred = y_pred)) +
    geom_density(aes(x = y_pred, fill = y_test), alpha = 0.7) +
    theme_minimal() +
    labs(title = "Separación de clases") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5), legend.position = "bottom")
}

oob <- trainControl(method = "oob",
                    classProbs = TRUE,
                    verboseIter = TRUE)

grid <- data.frame(mtry = seq(2,16,2)) #seq(2, 24, 2)

rf <- train(won ~ ., 
            data = X2 %>% mutate(won = ifelse(won == 1, "Yes", "No")), 
            method = "rf", 
            trControl = oob,
            tuneGrid = grid,
            metric = "Accuracy")

#rf <- readRDS("rf.RDS")

y_pred_2 <- predict(rf, X_valid_2 %>% select(-won), type = "prob")[2] 

winner.rf <- data.frame(matrix(NA,
                               nrow=dim(X_valid_2)[1],
                               ncol = 4))

colnames(winner.rf) <- c('race_id','prob_predicted','outcome_predicted','real_outcome')

for (j in 1:dim(winner.rf)[1]){  
  race_selected = X_valid_2[j,1]
  winner.rf[j,'race_id'] = race_selected
  winner.rf[j,'prob_predicted'] = y_pred_2[j,'Yes']
  winner.rf[j,'real_outcome'] = X_valid_2[j,2]}

winner.rf$prob_max <- rep(NA,dim(winner.basic_logit)[1])

for (j in 1:dim(winner.rf)[1]){
  sel = which(winner.rf[,'race_id']==winner.rf$race_id[j])
  winner.rf[j,'prob_max'] = max(winner.rf$prob_predicted[sel])
}

winner.rf <- winner.rf %>% mutate(outcome_predicted = ifelse(round(prob_predicted,3)-round(prob_max,3)<0,0,1))

winner.rf$winner_predicted <- rep(NA,dim(winner.rf)[1])
winner.rf <- winner.rf %>% mutate(winner_predicted = ifelse(real_outcome==1 & outcome_predicted==1,1,0)
  
ganadores_acertados_RF <- sum(winner.rf$winner_predicted)  
carreras_totales_RF <- as.numeric(length(unique(X_valid_2$race_id)))
ratio_ganadores <- ganadores_acertados_RF/carreras_totales_RF
print(paste("Acertamos ", round(ratio_ganadores,2)))

conf_matrix <- table(winner.rf$real_outcome, winner.rf$outcome_predicted)
metricas(conf_matrix)

# Área bajo la curva de ROC
roc(winner.rf$real_outcome ~ winner.rf$outcome_predicted, plot = TRUE, print.auc = TRUE)

plot_classes(y_test, y_pred)
# save.image("TP2.rds")

# --------     XGBoost       -------#

cv <- trainControl(method = "cv",
                   number = 5, #cambiar a 3
                   classProbs = TRUE,
                   verboseIter = TRUE,
                   summaryFunction = twoClassSummary)

tune_grid <- expand.grid(nrounds = seq(from = 20, to = 50, by = 4),
                         eta = c(0.01, 0.025, 0.05, 0.1, 0.3, 0.4),
                         max_depth = 4:8,
                         gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
                         colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
                         min_child_weight = 1:6,
                         subsample = c(0.5, 0.75, 1.0)) %>% sample_n(30)
xgb <- train(won ~ ., 
             data = X2 %>% mutate(won = ifelse(won == 0, "No", "Yes")), 
             method = "xgbTree", 
             trControl = cv,
             tuneGrid = tune_grid,
             metric = "ROC")

# xgb <- readRDS("xgb.RDS")

y_pred_3 <- predict(xgb, X_valid_2 %>% select(-won), type = "prob")[2] 

winner.xgb <- data.frame(matrix(NA,
                               nrow=dim(X_valid_2)[1],
                               ncol = 4))

colnames(winner.xgb) <- c('race_id','prob_predicted','outcome_predicted','real_outcome')

for (j in 1:dim(winner.xgb)[1]){  
  race_selected = X_valid_2[j,1]
  winner.xgb[j,'race_id'] = race_selected
  winner.xgb[j,'prob_predicted'] = y_pred_3[j,'Yes']
  winner.xgb[j,'real_outcome'] = X_valid_2[j,2]}

winner.xgb$prob_max <- rep(NA,dim(winner.xgb)[1])

for (j in 1:dim(winner.xgb)[1]){
  sel = which(winner.rf[,'race_id']==winner.xgb$race_id[j])
  winner.xgb[j,'prob_max'] = max(winner.xgb$prob_predicted[sel])
}

winner.xgb <- winner.xgb %>% mutate(outcome_predicted = ifelse(round(prob_predicted,3)-round(prob_max,3)<0,0,1))
winner.xgb$winner_predicted <- rep(NA,dim(winner.xgb)[1])
winner.xgb <- winner.xgb %>% mutate(winner_predicted = ifelse(real_outcome==1 & outcome_predicted==1,1,0))
                                  
ganadores_acertados_xgb <- sum(winner.xgb$winner_predicted)  
carreras_totales_xgb <- as.numeric(length(unique(X_valid_2$race_id)))
ratio_ganadores <- ganadores_acertados_xgb/carreras_totales_xgb
print(paste("Acertamos ", round(ratio_ganadores,2)))
                                    
conf_matrix <- table(winner.xgb$real_outcome, winner.xgb$outcome_predicted)
metricas(conf_matrix)
                                    
# Área bajo la curva de ROC
roc(winner.xgb$real_outcome ~ winner.xgb$outcome_predicted, plot = TRUE, print.auc = TRUE)
                                    
plot_classes(y_test, y_pred)
save.image("TP2.rds")

#-------------------------------------------------------------------------#
# Modelos de inversión
# Creo columnas auxiliares de ingreso en set de validación para facilitar cuentas
X_valid_2$IngresoGanador <- rep(NA,dim(X_valid_2)[1])
X_valid_2 <- X_valid_2 %>% mutate(IngresoGanador = ifelse(won==1,win_odds,0))
X_valid_2$IngresoPole <- rep(NA,dim(X_valid_2)[1])
X_valid_2 <- X_valid_2 %>% mutate(IngresoPole = ifelse(draw==2|draw==3,place_odds,0))

# Creo columna auxiliar en winner.xgb para ver si salio 2do o 3ero en mi predicción
winner.xgb$prob_promedio <- rep(NA,dim(winner.xgb)[1])
for (j in 1:dim(winner.xgb)[1]){
  sel = which(winner.xgb[,'race_id']==winner.xgb$race_id[j])
  winner.xgb[j,'prob_promedio'] = mean(winner.xgb$prob_predicted[sel])
}


# Estrategia 1 - 100% Naive
# Invierto $1 por caballo "a la cabeza" en todas las carreras y otro dolar "a la posición" en todas las carreras

inversion_win_estr1 = as.numeric(dim(winner.xgb)[1])
inversion_place_estr1 = as.numeric(dim(winner.xgb)[1])
ingresos_estr1 = sum(X_valid_2$IngresoGanador)+sum((X_valid_2$IngresoPole))
retorno_estr1 = ingresos_estr1 /(inversion_place_estr1+inversion_win_estr1)-1 #-12.6%

# Estrategia 2 - Solo invierto $1 "a la cabeza" con distintos umbrales 
umbrales <- c(0.1,0.15,0.2,0.25,0.3)
umbral_seleccionado <- umbrales[5]
ingresos_estr2 <-0
inversion_estr2 <-0

for (i in 1:dim(X_valid_2)[1]){
  ifelse(winner.xgb$outcome_predicted[i]==1 & winner.xgb$prob_max[i]>umbral_seleccionado,ingresos_estr2<-ingresos_estr2+X_valid_2$IngresoGanador[i],0)
  ifelse(winner.xgb$outcome_predicted[i]==1 & winner.xgb$prob_max[i]>umbral_seleccionado,inversion_estr2<-inversion_estr2+1,0)
  
}

# Umbral 0.1 -> Ingresos 3062, Inversion 3473 -11.8% Retorno
# Con XGBoost el modelo tira siempre probabilidad altas por lo que el umbral casi no cambia; ver que pasa con RF

# Estrategia 3 - Invierto "a la cabeza" y a los que superan la P promedio sin umbrales
ingresos_estr3_won <- 0
ingresos_estr3_place <- 0
inversion_estr3_won <- 0
inversion_estr3_place <- 0

for (i in 1:dim(X_valid_2)[1]){
  ifelse(winner.xgb$outcome_predicted[i]==1,ingresos_estr3_won<-ingresos_estr3_won+X_valid_2$IngresoGanador[i],0)
  ifelse(winner.xgb$outcome_predicted[i]==1,inversion_estr3_won<-inversion_estr3_won+1,0)
  ifelse(winner.xgb$prob_predicted[i]>winner.xgb$prob_promedio[i],ingresos_estr3_place <- ingresos_estr3_won + X_valid_2$IngresoPole[i],0)
  ifelse(winner.xgb$prob_predicted[i]>winner.xgb$prob_promedio[i],inversion_estr3_place <- inversion_estr3_place + 1,0)
  
}



