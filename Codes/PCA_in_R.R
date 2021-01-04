###### Applying PCA in R
library(dplyr)
library(ggcorrplot)

####################### 0 - Load the data #####################################
df <- read.csv('https://raw.githubusercontent.com/Seb943/PCA_and_ML_R/main/Data/Example_dataset.csv') %>% dplyr::select(-X)

# Build datasets for winner predictions
df <- df %>% dplyr::select(B365_OT_H, B365_OT_A, TotWins_home, TotLose_home, 
                           TotWins_away, TotLose_away, Elo_home, Elo_away, 
                           Moy_scored_season_home, Moy_against_season_home, 
                           Moy_scored_season_away, Moy_against_season_away)
tableTRAIN <- head(df, 0.8*dim(df)[1])
tableTEST <- tail(df, 0.2*dim(df)[1])

# Plot correlation matrix
mcor <- cor(tableTRAIN)
ggcorrplot(mcor)

######################### I - Fit the PCA #####################################

# Perform PCA, save result
prin_comp <- prcomp(tableTRAIN, scale. = T)
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",  # 
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b", main = "CPVE graph")
# scree plot
plot(prop_varex, xlab = "Principal Component",  # 
     ylab = "Proportion of Variance Explained",
     type = "b", main = "PVE graph")

#Save result : we can export the PCA in this manner
write.csv(as.data.frame(prin_comp$sdev), "sdev_PCA_Medium.csv")
write.csv(as.data.frame(prin_comp$rotation), "rotation_PCA_Medium.csv")
write.csv(as.data.frame(prin_comp$center), "center_PCA_Medium.csv")
write.csv(as.data.frame(prin_comp$scale), "scale_PCA_Medium.csv")


################# II : Make predictions ##############################

train.data <- predict(prin_comp, newdata = tableTRAIN)
test.data <- predict(prin_comp, newdata = tableTEST)


# compute PCA projection from the saved components, assert we have good computation
train.data2  <- scale(tableTRAIN, prin_comp$center, prin_comp$scale) %*% prin_comp$rotation 
test.data2 <- scale(tableTEST, prin_comp$center, prin_comp$scale) %*% prin_comp$rotation 
train.data2 == train.data
test.data2 == test.data
