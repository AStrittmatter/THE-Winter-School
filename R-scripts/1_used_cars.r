
########################  Load Packages  ########################

# List of required packages
pkgs <- c('tidyverse','glmnet','corrplot','plotmo')

# Load packages
for(pkg in pkgs){
    install.packages(pkg)
    library(pkg, character.only = TRUE)
}

print('All packages successfully installed and loaded.')

########################  Load Data Frame  ########################

# Load data 
data_raw <- read.csv("Data/mylemon.csv",header=TRUE, sep=",")

set.seed(1001) # set starting value for random number generator
# Selection of Subsample size, max. 104,721 observations
# Select smaller subsample to decrease computation time
n_obs <- 200
df <- data_raw %>%
  dplyr::sample_n(n_obs) 

print('Data frame successfully loaded.')

########################  Take Hold-Out-Sample  ########################
set.seed(1001) # set starting value for random number generator

# Partition data in training and test sample
df_part <- modelr::resample_partition(df, c(train = 0.8, test = 0.2))
df_train <- as.data.frame(df_part$train) # Training sample
df_test <- as.data.frame(df_part$test) # Test sample

# Outcome
price_train <- as.matrix(df_train[,2])
price_test <- as.matrix(df_test[,2])

# Covariates
covariates_train <- as.matrix(df_train[,c(3:ncol(df_train))])
covariates_test <- as.matrix(df_test[,c(3:ncol(df_test))])

print('The data is now ready for your first analysis!')

########################  Correlation Matrix  ########################

corr = cor(covariates_train)
corrplot(corr, type = "upper", tl.col = "black")


########################  OLS Model  ######################## 

# Estimate OLS model
ols <- lm(price_train ~., as.data.frame(covariates_train))
# Some variables might be dropped because of perfect colinearity 
summary(ols) # Plot table of coefficients

# Training sample fitted values
fit1_train <- predict.lm(ols)

# Test sample fitted values
fit1_test <- predict.lm(ols, newdata = data.frame(covariates_test))

print('Fitted values are calculated.')

# R-squared in training sample
rsquared_train <- round(1-mean((price_train- fit1_train)^2)/mean((price_train - mean(price_train))^2),digits=3)
#print(paste0("In-Sample MSE OLS: ", mse1_in))

# R-squared in test sample
rsquared_test <- round(1-mean((price_test - fit1_test)^2)/mean((price_test - mean(price_test))^2),digits=3)

print(paste0("Training Sample R-squared OLS: ", rsquared_train))
print(paste0("Test Sample R-squared OLS: ", rsquared_test))

########################  CV-LASSO  ######################## 
p = 1 # 1 for LASSO, 0 for Ridge

set.seed(10101)
lasso <- cv.glmnet(covariates_train, price_train, alpha=p, family = "gaussian", 
                          nlambda = 100, type.measure = 'mse')
# nlambda specifies the number of different lambda values on the grid (log-scale)
# type.measure spciefies that the optimality criteria is the MSE in CV-samples
# alpha allows to select between Ridge and LASSO
# family allows to specify whether the outcome variable has limited support
# family = "gaussian" is the default (no limited support)

# Plot MSE in CV-Samples for different values of lambda
plot(lasso)

# Optimal lambda values
# Minimal MSE
print(paste0("Lambda minimising CV-MSE: ", round(lasso$lambda.min,digits=3)))
# 1 standard error rule reduces the number of included covariates
print(paste0("Lambda 1 standard error rule: ", round(lasso$lambda.1se,digits=3)))

# Number of non-zero coefficients
# Minimal MSE
print(paste0("Number of selected covariates (lambda.min): ",lasso$glmnet.fit$df[lasso$glmnet.fit$lambda==lasso$lambda.min]))
# 1 standard error rule reduces the number of included covariates
print(paste0("Number of selected covariates (lambda.1se): ",lasso$glmnet.fit$df[lasso$glmnet.fit$lambda==lasso$lambda.1se]))


########################  Visualisation of LASSO  ######################## 

set.seed(10101)
mod <- glmnet(covariates_train, price_train, lambda.min.ratio = lasso$lambda.min, alpha=p)
maxcoef<-coef(lasso)
coef<-dimnames(maxcoef[maxcoef[,1]!=0,0])[[1]]
allnames<-dimnames(maxcoef[maxcoef[,1]!=0,0])[[1]][order(maxcoef[maxcoef[,1]!=0,ncol(maxcoef)],decreasing=TRUE)]
allnames<-setdiff(allnames,allnames[grep("Intercept",allnames)])

plot_glmnet(mod,label=TRUE,s=lasso$lambda.1se)

########################  Plot LASSO Coefficients  ########################

print('LASSO coefficients')

glmcoef<-coef(lasso, lasso$lambda.1se)
print(glmcoef)
# the LASSO coefficients are biased because of the penalty term

######################## Training Sample Performance of LASSO  ######################## 

# Estimate LASSO model 
# Use Lambda that minizes CV-MSE
set.seed(10101)
lasso.fit.min <- glmnet(covariates_train, price_train, lambda = lasso$lambda.min)
yhat.lasso.min <- predict(lasso.fit.min, covariates_train)

# Use 1 standard error rule
set.seed(10101)
lasso.fit.1se <- glmnet(covariates_train, price_train, lambda = lasso$lambda.1se)
yhat.lasso.1se <- predict(lasso.fit.1se, covariates_train)

# In-sample performance measures
print(paste0("In-Sample R-squared OLS: ", rsquared_train))

rsquared2_train <- round(1-mean((price_train - yhat.lasso.min)^2)/mean((price_train - mean(price_train))^2),digits=3)
print(paste0("In-Sample R-squared Lasso (lambda.min): ", rsquared2_train))

rsquared3_train <- round(1-mean((price_train - yhat.lasso.1se)^2)/mean((price_train - mean(price_train))^2),digits=3)
print(paste0("In-Sample R-squared Lasso (lambda.1se): ", rsquared3_train))

######################## Test Sample Performance of LASSO  ######################## 

# Extrapolate Lasso fitted values totest sample
yhat.lasso.min <- predict(lasso.fit.min, covariates_test)
yhat.lasso.1se <- predict(lasso.fit.1se, covariates_test)

# Out-of-sample performance measures
print(paste0("Out-of-Sample R-squared OLS: ", rsquared_test))

rsquared2_test <- round(1-mean((price_test - yhat.lasso.min)^2)/mean((price_test - mean(price_test))^2),digits=3)
print(paste0("Out-of-Sample R-squared Lasso (lambda.min): ", rsquared2_test))

rsquared3_test <- round(1-mean((price_test - yhat.lasso.1se)^2)/mean((price_test - mean(price_test))^2),digits=3)
print(paste0("Out-of-Sample R-squared Lasso (lambda.1se): ", rsquared3_test))

######## Put Your Code Here ########







####################################
