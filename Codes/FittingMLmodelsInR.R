################## A complete guide to fitting usual ML models in R ##################
library(dplyr)
library(MASS) # for train unction : useful for 
library(randomForest) # for RandomForest
library(nnet) # for avNNet
library(caret) # for train control (i.e. Cross-Validation)
library(e1071) # for svm


################### 0 - Import dataset, define cross-validation ########################
df <- read.csv('https://raw.githubusercontent.com/Seb943/PCA_and_ML_R/main/Data/Example_dataset.csv') %>% dplyr::select(-X)

# Build datasets for winner predictions
df <- df %>% dplyr::select(Winner, B365_OT_H, B365_OT_A, TotWins_home, TotLose_home, 
                                             TotWins_away, TotLose_away, Elo_home, Elo_away, 
                                             Moy_scored_season_home, Moy_against_season_home, 
                                             Moy_scored_season_away, Moy_against_season_away)
df['Winner'] <- lapply(df['Winner'] , factor)
df_train <- head(df, 0.8*dim(df)[1])
df_test <- tail(df, 0.2*dim(df)[1])

# Define 10-fold CV 
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE)

############################# I - Random Forests ########################################
rf_fit <- randomForest(Winner ~ ., 
                       family = "binomial", 
                       data = df_train, 
                       trControl = fitControl,
                       ntree = 500,
                       metric = "Accuracy")
model_rf_Medium <- rf_fit
save(model_rf_Medium, file="model_rf_Medium.Rdata")

###################################### II - LDA, QDA, etc... ############################################
#  QDA
qda_fit <- train (Winner ~ .,
                  data = df_train ,
                  method = 'qda' ,
                  trControl = fitControl ,
                  metric =  'Accuracy' )
model_qda_Medium <- qda_fit
save(model_qda_Medium, file="model_qda_Medium.Rdata")


###################################### III - SVM ####################################################
# SVM linear 
svm_fit <- svm(Winner ~ ., 
               type = 'C-classification',
               kernel = 'linear', # Could've chosen 'radial'
               data = df_train, 
               cross = 10, # 10-fold CV
               probability = TRUE,
               metric = "Accuracy")
model_svm_linear_Medium <- svm_fit
save(model_svm_linear_Medium, file="model_svm_linear_Medium.Rdata")

########################### IV - averaged Neural Networks (avNNet) #####################################
avNNET_fit <- avNNet(Winner ~ ., 
                     repeats = 20, 
                     data = df_train, 
                     bag = TRUE,
                     trControl = fitControl,
                     size = 1,
                     maxit = 10000)

model_avNNET_s1_Medium <- avNNET_fit
save(model_avNNET_s1_Medium, file = "model_avNNET_s1_Medium.Rdata")

########################### V - Make predictions #################################################
# (a) Load model
rf_fit <- load("model_rf_Medium.Rdata")

# (b) Make predictions
preds<-as.data.frame(predict(get(rf_fit), df_test, type = "class"))
#preds_lda <-as.data.frame(predict(get(lda_fit), df_test ,type="raw")) # for LDA and QDA only

# (c) Compare to the ground truth data and display accuracy
cat('Accuracy is', 100*mean(preds[,1] == df_test$Winner), '% over the test set')

    