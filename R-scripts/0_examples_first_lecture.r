
# Set seed
set.seed(12345)

# Load data frame
df_train <- read.csv("Data/used_cars_train.csv",header=TRUE, sep=",")
df_test <- read.csv("Data/used_cars_test.csv",header=TRUE, sep=",")

# Specify Outcome Variable
first_price_train <- as.matrix(df_train[,2])
first_price_test <- as.matrix(df_test[,2])

# Specify Covariates
#First Variable is the Intercept
covariates_train <- as.matrix(cbind(rep(1,nrow(df_train)),df_train[,c(3:ncol(df_train))]))
covariates_test <- as.matrix(cbind(rep(1,nrow(df_test)),df_test[,c(3:ncol(df_test))])) 

print('Data frame successfully loaded.')

########################  Estimation  ########################

# Generate Matrices to Store the Results
mse <- matrix(NA, nrow = ncol(covariates_train), ncol = 2) 
y_hat_train <- matrix(NA,nrow = nrow(first_price_train), ncol = ncol(covariates_train)) #Training sample
y_hat_test <- matrix(NA,nrow = nrow(first_price_test), ncol = ncol(covariates_train)) # Test sample

# Estimate Different OLS Models
# Start with a model containing only an intercept
# Add covariates one-by-one
for (c in (1:ncol(covariates_train))){
  formular <- lm.fit(as.matrix(covariates_train[,c(1:c)]),first_price_train) # OLS regression
  y_hat_train[,c] <- formular$fitted.values # Fitted values in training sample
  coef <- as.matrix(formular$coefficients) # Store vector of coefficients
  coef[is.na(coef)] <- 0 # Replace NAs with 0 (in case of perfect multicolinearity)
  y_hat_test[,c] <- covariates_test[,c(1:c)] %*% coef # Fitted values in test sample
  mse[c,1] <- round(mean((y_hat_train[,c] - first_price_train)^2),digits=3) # MSE of training sample
  mse[c,2] <- mean((y_hat_test[,c] - first_price_test)^2) # MSE of test sample
}

# Add Column with Number of Covariates
mse <- cbind(mse,seq(1,nrow(mse)))

print('Models are estimated.')

##################### Plot MSE in Training Sample ##################### 

plot(mse[,3],mse[,1], type = "n", ylab = "MSE", xlab = "Number of Covariates")
lines(mse[,3],mse[,1])

print(paste0("MSE for K = 1: ",mse[1,1]))
print(paste0("MSE for K = 10: ",mse[10,1]))
print(paste0("MSE for K = 40: ",mse[40,1]))

##################### Scatterplot Predicted Prices in Training Sample ##################### 

plot(first_price_train,y_hat_train[,1],xlim=c(5,30),ylim=c(5,30), col= "darkgreen", xlab = "Observed Price", ylab = "Predicted Price" )
par(new=TRUE)
plot(first_price_train,y_hat_train[,10],xlim=c(5,30),ylim=c(5,30), col= "blue", xlab = "Observed Price", ylab = "Predicted Price" )
par(new=TRUE)
plot(first_price_train,y_hat_train[,40],xlim=c(5,30),ylim=c(5,30), col= "red", xlab = "Observed Price", ylab = "Predicted Price" )
abline(a=0,b=1)
legend(24.5, 14, c("K=1", "K=10", "K=40"), col = c("darkgreen", "blue", "red"), pch = c(21, 21, 21))

##################### Plot MSE in Test Sample ##################### 

plot(mse[,3],mse[,2], type = "n", ylab = "MSE", xlab = "Number of Covariates")
lines(mse[,3],mse[,2])

print(paste0("MSE for K = 1: ", round(mse[1,2], digits =3)))
print(paste0("MSE for K = 10: ", round(mse[10,2], digits =3)))
print(paste0("MSE for K = 40: ", round(mse[40,2], digits =3)))

##################### Scatterplot Predicted Prices in Test Sample ##################### 

plot(first_price_test,y_hat_test[,1],xlim=c(5,30),ylim=c(5,30), col= "darkgreen", xlab = "Observed Price", ylab = "Predicted Price")
par(new=TRUE)
plot(first_price_test,y_hat_test[,10],xlim=c(5,30),ylim=c(5,30), col= "blue", xlab = "Observed Price", ylab = "Predicted Price")
par(new=TRUE)
plot(first_price_test,y_hat_test[,40],xlim=c(5,30),ylim=c(5,30), col= "red", xlab = "Observed Price", ylab = "Predicted Price")
abline(a=0,b=1)
legend(24.5, 14, c("K=1", "K=10", "K=40"), col = c("darkgreen", "blue", "red"), pch = c(21, 21, 21))

############################# Simulation of Bias-Variance Trade-Off ############################# 

# Set input parameters for the simulation
sub = 1500 # number of subamples
sd = 3 # irreducable noise

# Load data
data_raw <- read.csv("Data/mylemon.csv",header=TRUE, sep=",") # Load larger data set

# Set starting values for random number generator
set.seed(100001)

# Select training and test sample
df_train <- data_raw[-1,] # drops the first observation
df_test <- data_raw[1,] # contains only the first observation

# Define outcome and control variables
first_price_train <- as.matrix(df_train[,2])
first_price_test <- as.matrix(df_test[,2])
covariates_train <- as.matrix(cbind(rep(1,nrow(df_train)),df_train[,c(3:ncol(df_train))]))
covariates_test <- as.matrix(cbind(rep(1,nrow(df_test)),df_test[,c(3:ncol(df_test))]))

# Simulate the Data Generating Process
u_tr <- matrix(rnorm(nrow(first_price_train),0,sd),nrow= nrow(first_price_train), ncol =1) # Irreducable noise
# Estimate the empirical coefficients
formular <- lm.fit(rbind(covariates_train[,c(1:10)],covariates_test[,c(1:10)]),rbind(first_price_train,first_price_test))
coef <- as.matrix(formular$coefficients)
coef[is.na(coef)] <- 0
# Simulate the price based on empricial coeffcents, observed covariates, and noise
y_new_train <- covariates_train[,c(1:10)] %*% coef  +u_tr 
y_0 <- covariates_test[,c(1:10)] %*% coef 
p <- as.matrix(ceiling(runif(nrow(df_train),0,sub))) # define sample partitions

# Estimate different OLS models on simulated price
mse <- matrix(NA, nrow = ncol(covariates_train), ncol = sub)
y_hat_test <- matrix(NA, nrow = ncol(covariates_train), ncol = sub)
for (n in (1:sub)) { # iterate through replications of DGP
    for (c in (1:ncol(covariates_train))){ # iterate through OLS models with different number ofcovariates
        formular <- lm.fit(as.matrix(covariates_train[p==n,c(1:c)]),y_new_train[p==n,])
        y_hat_train <- formular$fitted.values
        coef <- as.matrix(formular$coefficients)
        coef[is.na(coef)] <- 0
        y_hat_test[c,n] <- covariates_test[,c(1:c)] %*% coef
        mse[c,n] <- mean((y_hat_test[c,n] - y_0 - rnorm(1,0,sd))^2)
    }
}

# Aggregate results accross all subsamples
test <- matrix(NA, nrow = ncol(covariates_train), ncol = 3)
for (c in (1:ncol(covariates_train))){
    test[c,1] <- var(y_hat_test[c,])
    test[c,2] <- (mean(y_hat_test[c,]) - y_0)^2
    test[c,3] <- mean(mse[c,]) 
}
test <- cbind(test,seq(1,nrow(test)))

# Plot Bias-Variance Trade-Off
plot(test[,4],test[,3], type = "n", ylab = "", xlab = "Number of Covariates", ylim = c(0,45))
lines(test[,4],test[,3], col = "red",lwd = 2)
par(new=TRUE)
lines(test[,4],test[,2], col = "darkgreen",lwd = 2)
par(new=TRUE)
lines(test[,4],test[,1], col = "orange",lwd = 2)
abline(h=sd^2,lty = 3)
legend(20, 40, c("MSE", "Squared-Bias", "Variance"), col = c("red", "darkgreen", "orange"), lty = c(1,1, 1),lwd = 2)

# Load Packages
library("glmnet")
library("dplyr")

# Set starting values for random number generator
set.seed(10001)

# Load data
data_raw <- read.csv("Data/mylemon.csv",header=TRUE, sep=",") # Load data set


n_obs <- 100
data_use <- data_raw[,-c(1,3:5,13:37,40:41)] %>%
  dplyr::sample_n(n_obs)


# Generate variable with the rows in training data
size <- floor(0.5 * nrow(data_use))
training_set <- sample(seq_len(nrow(data_use)), size = size)

# Select training and test sample
df_train <- data_use[training_set,] # drops the first observation
df_test <- data_use[-training_set,] # contains only the first observation

# Define outcome and control variables
first_price_train <- as.matrix(df_train[,1])
first_price_test <- as.matrix(df_test[,1])
covariates_train <- as.matrix(df_train[,c(2:ncol(df_train))])
covariates_test <- as.matrix(df_test[,c(2:ncol(df_test))])

print('Data prepared.')

ols <- lm(first_price ~., data=df_train)
summary(ols)

# Calculate the MSE
predols_train <- predict(ols, newdata = df_train)
predols_test <- predict(ols, newdata = df_test)

R2_ols_train <- 1- mean((first_price_train - predols_train)^2)/mean((first_price_train - mean(first_price_train))^2)
R2_ols_test <- 1- mean((first_price_test - predols_test)^2)/mean((first_price_test - mean(first_price_test))^2)

print(paste0("Training R-squared: ", round(R2_ols_train, digits =3)))
print(paste0("Test R-squared: ", round(R2_ols_test, digits =3)))

set.seed(1001)

lasso.cv <- cv.glmnet(covariates_train, first_price_train, type.measure = "mse", nfolds = 10)
coef_lasso <- coef(lasso.cv, s = "lambda.min") # save for later comparison
print(coef_lasso)

# Calculate the MSE
predlasso_train <- predict(lasso.cv, newx = covariates_train, s = lasso.cv$lambda.min)
predlasso_test <- predict(lasso.cv, newx = covariates_test, s = lasso.cv$lambda.min)

R2_lasso_train <- 1- mean((first_price_train - predlasso_train)^2)/mean((first_price_train - mean(first_price_train))^2)
R2_lasso_test <- 1- mean((first_price_test - predlasso_test)^2)/mean((first_price_test - mean(first_price_test))^2)

print(paste0("Training R-squared: ", round(R2_lasso_train, digits =3)))
print(paste0("Test R-squared: ", round(R2_lasso_test, digits =3)))

# Set starting values for random number generator
set.seed(1299995)

n_obs <- 100

data_use <- data_raw[-training_set,-c(1,3:5,13:37,40:41)] %>%
  dplyr::sample_n(n_obs)

# Generate variable with the rows in training data
size <- floor(0.5 * nrow(data_use))
training_set <- sample(seq_len(nrow(data_use)), size = size)

# Select training and test sample
df_train <- data_use[training_set,] # drops the first observation


# Define outcome and control variables
first_price_train <- as.matrix(df_train[,1])
covariates_train <- as.matrix(df_train[,c(2:ncol(df_train))])

lasso.cv <- cv.glmnet(covariates_train, first_price_train, type.measure = "mse", nfolds = 10)
coef_lasso_1 <- coef(lasso.cv, s = "lambda.min") # save for later comparison

# Calculate the MSE
predlasso_train_1 <- predict(lasso.cv, newx = covariates_train, s = lasso.cv$lambda.min)
predlasso_test_1 <- predict(lasso.cv, newx = covariates_test, s = lasso.cv$lambda.min)

R2_lasso_train <- 1- mean((first_price_train - predlasso_train_1)^2)/mean((first_price_train - mean(first_price_train))^2)
R2_lasso_test <- 1- mean((first_price_test - predlasso_test_1)^2)/mean((first_price_test - mean(first_price_test))^2)
print(paste0("Training R-squared: ", round(R2_lasso_train, digits =3)))
print(paste0("Test R-squared: ", round(R2_lasso_test, digits =3)))

# Set starting values for random number generator
set.seed(1234564578)

n_obs <- 100

data_use <- data_raw[-training_set,-c(1,3:5,13:37,40:41)] %>%
  dplyr::sample_n(n_obs)

# Generate variable with the rows in training data
size <- floor(0.5 * nrow(data_use))
training_set <- sample(seq_len(nrow(data_use)), size = size)

# Select training and test sample
df_train <- data_use[training_set,] # drops the first observation

# Define outcome and control variables
first_price_train <- as.matrix(df_train[,1])
covariates_train <- as.matrix(df_train[,c(2:ncol(df_train))])

lasso.cv <- cv.glmnet(covariates_train, first_price_train, type.measure = "mse", nfolds = 10)
coef_lasso_2 <- coef(lasso.cv, s = "lambda.min") # save for later comparison

# Calculate the MSE
predlasso_train_2 <- predict(lasso.cv, newx = covariates_train, s = lasso.cv$lambda.min)
predlasso_test_2 <- predict(lasso.cv, newx = covariates_test, s = lasso.cv$lambda.min)

R2_lasso_train <- 1- mean((first_price_train - predlasso_train_2)^2)/mean((first_price_train - mean(first_price_train))^2)
R2_lasso_test <- 1- mean((first_price_test - predlasso_test_2)^2)/mean((first_price_test - mean(first_price_test))^2)
print(paste0("Training R-squared: ", round(R2_lasso_train, digits =3)))
print(paste0("Test R-squared: ", round(R2_lasso_test, digits =3)))

# Set starting values for random number generator
set.seed(15678)

n_obs <- 100

data_use <- data_raw[-training_set,-c(1,3:5,13:37,40:41)] %>%
  dplyr::sample_n(n_obs)


# Generate variable with the rows in training data
size <- floor(0.5 * nrow(data_use))
training_set <- sample(seq_len(nrow(data_use)), size = size)

# Select training and test sample
df_train <- data_use[training_set,] # drops the first observation

# Define outcome and control variables
first_price_train <- as.matrix(df_train[,1])
covariates_train <- as.matrix(df_train[,c(2:ncol(df_train))])

lasso.cv <- cv.glmnet(covariates_train, first_price_train, type.measure = "mse", nfolds = 10)
coef_lasso_3 <- coef(lasso.cv, s = "lambda.min") # save for later comparison

# Calculate the MSE
predlasso_train_3 <- predict(lasso.cv, newx = covariates_train, s = lasso.cv$lambda.min)
predlasso_test_3 <- predict(lasso.cv, newx = covariates_test, s = lasso.cv$lambda.min)

R2_lasso_train <- 1- mean((first_price_train - predlasso_train_3)^2)/mean((first_price_train - mean(first_price_train))^2)
R2_lasso_test <- 1- mean((first_price_test - predlasso_test_3)^2)/mean((first_price_test - mean(first_price_test))^2)
print(paste0("Training R-squared: ", round(R2_lasso_train, digits =3)))
print(paste0("Test R-squared: ", round(R2_lasso_test, digits =3)))

# Set starting values for random number generator
set.seed(12345678)

n_obs <- 100

data_use <- data_raw[-training_set,-c(1,3:5,13:37,40:41)] %>%
  dplyr::sample_n(n_obs)

# Generate variable with the rows in training data
size <- floor(0.5 * nrow(data_use))
training_set <- sample(seq_len(nrow(data_use)), size = size)

# Select training and test sample
df_train <- data_use[training_set,] # drops the first observation

# Define outcome and control variables
first_price_train <- as.matrix(df_train[,1])
covariates_train <- as.matrix(df_train[,c(2:ncol(df_train))])

lasso.cv <- cv.glmnet(covariates_train, first_price_train, type.measure = "mse", nfolds = 10)
coef_lasso_4 <- coef(lasso.cv, s = "lambda.min") # save for later comparison

# Calculate the MSE
predlasso_train_4 <- predict(lasso.cv, newx = covariates_train, s = lasso.cv$lambda.min)
predlasso_test_4 <- predict(lasso.cv, newx = covariates_test, s = lasso.cv$lambda.min)

R2_lasso_train <- 1- mean((first_price_train - predlasso_train_4)^2)/mean((first_price_train - mean(first_price_train))^2)
R2_lasso_test <- 1- mean((first_price_test - predlasso_test_4)^2)/mean((first_price_test - mean(first_price_test))^2)
print(paste0("Training R-squared: ", round(R2_lasso_train, digits =3)))
print(paste0("Test R-squared: ", round(R2_lasso_test, digits =3)))

print(cbind(coef_lasso,coef_lasso_1,coef_lasso_2,coef_lasso_3,coef_lasso_4))

print(cor(cbind(predlasso_test,predlasso_test_1,predlasso_test_2,predlasso_test_3,predlasso_test_4)))


