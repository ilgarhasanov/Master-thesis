
library("mvtnorm")
## set the seed for reproducibility 
set.seed(123)

#1) The simulation setup

## correlation matrix
R <- matrix(c(1, .5, .5, .5, 1, .5, .5, .5, 1), nrow = 3)

## simulation
nobs <- 1000
x <- rmvnorm(nobs, sigma = R) 
colnames(x) <- c("x1", "x2", "x3")

eta <- -1 + 0.5 * x[ ,1] + 0.5 * x[, 2] - 2 * x[ ,3]
pr <- exp(eta) / (1 + exp(eta))
y <- rbinom(nobs, size = 1, prob = pr)

df = data.frame(y = y,x = x)
library("caTools")

#Split train and test set
sample = sample.split(df,SplitRatio = 0.8) 
train = subset(df,sample == TRUE) 
test = subset(df, sample == FALSE)


#2) The first simulation setup

#create a list for datasets
mylist <- list()

#Implementing logistic regression with all predictors
mod <- glm(y ~ x.x1 + x.x2 + x.x3, data = train, family = binomial)
summary(mod)

#Check with only one predictor
mod2 <- glm(y ~ x.x1, data = train, family= binomial)
summary(mod2) 

#Add multiplication of two predictors to mod
mod3 <- update(mod, .~. + x.x1 * x.x2)
summary(mod3)  

mod4 <- update(mod, .~. + x.x1 * x.x2 * x.x3)  
summary(mod4) 

#predict the model
glm.probs <- predict(mod, newdata = test, type = "response")
preds <- glm.probs > 0.5 
tb <- table(preds = preds, true = test$y)
ac <- (tb[1,1] + tb[2,2]) / (tb[1,1] + tb[1,2] + tb[2,1] + tb[2,2])
pr <- tb[2,2]/(tb[2,2]+tb[2,1])
rc <- tb[2,2]/(tb[2,2]+tb[1,2])

#Implemention of support vector machines
library(e1071)

#Svm with the linear kernel
svm_mod <- svm(y ~., data = train, type = "C-classification", kernel = "linear", scale = FALSE)
summary(svm_mod)
mean(pred_train == train$y)

#Change svm model with different cost to see if there is improvement in test error
svm_mod2 <- svm(y ~., data = train, type = "C-classification", kernel = "linear", cost = 10, scale = FALSE)
pred_test2 <- predict(svm_mod2, test)
mean(pred_test2 == test$y)

#SVM with the polynomial kernel
svm_mod3 <- svm(y ~., data = train, type = "C-classification", kernel = "polynomial", degree = 2)
pred_test3 <- predict(svm_mod3, test)
mean(pred_test3 == test$y)

#Tuning the polynomial kernel
poly.tune <- tune.svm(y~., data = train, kernel="polynomial",degree =2, cost = 1, gamma = c(0.1,1,10),coef0 = c(0.1,1,10), scale = FALSE)
poly.tune$best.model$cost
poly.tune$best.model$gamma
poly.tune$best.model$coef0

#apply the best parameters to the new model
tune_poly_best <- svm(y~., data = train,type = "C-classification", kernel = "polynomial",
                      degree = 2, gamma = 0.1, coef0 = 1, cost = 1)
pred_poly_best <- predict(tune_poly_best, test)
mean(pred_poly_best == test$y)

#higher degree
svm_model_high <- svm(y ~., data = train, type = "C-classification", kernel = "polynomial",degree = 3, gamma = 0.1, coef0 = 1, cost = 1)
pred_test_high <- predict(svm_model_high, test)
mean(pred_test_high == test$y)

#higher degree
svm_model_high4 <- svm(y ~., data = train, type = "C-classification", kernel = "polynomial",degree = 4, gamma = 0.1, coef0 = 1, cost = 1 )
pred_test_high4 <- predict(svm_model_high4, test)
mean(pred_test_high4 == test$y)

#radial
svm_model_rad <- svm(y ~., data = train, type = "C-classification", kernel = "radial", degree = 2)
pred_test_rad <- predict(svm_model_rad, test)
mean(pred_test_rad == test$y)

#Tuning the radial kernel
tune_out_rad <- tune.svm(x = train[,-1],y = train[,1],kernel = "radial",
                         gamma = 5 * 10 ^ (-2:2), coef0 = c(0.1,1,10), cost = c(0.01, 0.1, 1, 10, 100) )
tune_out_rad$best.model$gamma                
tune_out_rad$best.model$cost
tune_out_rad$best.model$coef0

#apply the best parameters to the new model
tune_out_rad_best <- svm(y ~., data = train, type = "C-classification", kernel = "radial",
                         gamma = 0.05, coef0 = 0.1, cost = 100)
pred_rad_best<-predict(tune_out_rad_best,test)
mean(pred_rad_best == test$y)

#Visualize accuracy scores of models with the boxplots
acc_log <- data.frame(Reduce(rbind, acc))
acc_svm <- data.frame(Reduce(rbind, acc_svm))
c <- data.frame(A=acc_log,B=acc_svm)
colnames(c) <- c("Logistic regression","Support vector machine")
boxplot(c, main = "Accuracy scores of methods", xlab = "Accuracy of logistic and support vector machine", ylab="Accuracy Score",  border = "black")

#Visualize precision scores of models with the boxplots
precision_log <- data.frame(Reduce(rbind, pr))
precision_svm <- data.frame(Reduce(rbind, precision_svm))
c <- data.frame(A=precision_log,B=precision_svm)
colnames(c) <- c("Logistic regression","Support vector machine")
boxplot(c, main = "Precision scores of methods", border = "black")

#Visualize recall scores of models with the boxplots
recall_log <- data.frame(Reduce(rbind, rc))
recall_svm <- data.frame(Reduce(rbind, recall_svm))
c <- data.frame(A=recall_log,B=recall_svm)
colnames(c) <- c("Logistic regression","Support vector machine")
boxplot(c, main = "Recall scores of methods",  border = "black")


#3) Varying the threshold in the logistic regression
#create 1000 datasets
for (i in 1:1000){
  nobs <- 500
  x <- rmvnorm(nobs, sigma = R) 
  colnames(x) <- c("x1", "x2", "x3")
  eta <- -1 + 0.5 * x[ ,1] + 0.5 * x[, 2] - 2 * x[ ,3]
  pr <- exp(eta) / (1 + exp(eta))
  y <- rbinom(nobs, size = 1, prob = pr)
  mylist[[i]] = data.frame(y=y,x=x)
}

#split datasets into train and test sets
library("caTools")
train <- list()
test <- list()

for (i in 1:1000){
  sample = sample.split(mylist[[i]],SplitRatio = 0.8)
  train[[i]] =subset(mylist[[i]],sample ==TRUE) 
  test[[i]]=subset(mylist[[i]], sample==FALSE)
}

#Creata lists
glm.fit <- list(); probs<- list(); pred_log <- list(); acc_log <- list()
svm.fit <- list(); pred_svm <- list(); acc_svm <- list()
table_log <- list(); precision_log <- list(); recall_log <- list();  f1_log <- list()
table_svm <- list(); precision_svm <- list(); recall_svm <- list(); f1_svm <- list()
threshold <- list(); glm.probs <- list(); accuracy <- list(); threshold_fram <- list()

#Implementation logistic regression
for (i in 1:1000){
  glm.fit[[i]] <- glm(y ~ x.x1 + x.x2 + x.x3, data = train[[i]], family = binomial)
  probs[[i]] <- predict(glm.fit[[i]], newdata = test[[i]], type = "response")
  glm.probs[[i]] <- roc(test[[i]]$y ~ probs[[i]], plot = FALSE, print.auc = FALSE)
  cc <- coords(glm.probs[[i]], x = "best", ret = "all")
  threshold[[i]] <- cc["threshold"]$threshold
  pred_log[[i]] <- probs[[i]] > threshold[[i]]
  table_log <- table(preds = pred_log[[i]], true = test[[i]]$y)
  acc_log[[i]] <- (table_log[1,1] + table_log[2,2]) / (table_log[1,1] + table_log[1,2] + table_log[2,1] + table_log[2,2])
  precision_log[[i]] <- table_log[2,2] / (table_log[2,2] + table_log[2,1])
  recall_log[[i]] <- table_log[2,2] / (table_log[2,2] + table_log[1,2])
}

#Visualize the thresholds
threhsold_frame <- data.frame(Reduce(rbind, threshold))
c <- data.frame(threhsold_frame)
colnames(c) <- c("The thresholds")
boxplot(c, main = "The optimal thresholds of each data" ,  border = "black")

#Implementing the Support Vector Machine
library(e1071)
for (i in 1:1000){
  svm.fit[[i]] <- svm(y ~., data = train[[i]], type = "C-classification", kernel = "polynomial",degree = 3, gamma = 0.1, coef0 = 1, cost = 1)
  pred_svm[[i]] <- predict(svm.fit[[i]], test[[i]])
  table_svm <- table(preds = pred_svm[[i]], true = test[[i]]$y)
  acc_svm[[i]] <- (table_svm[1,1] + table_svm[2,2]) / (table_svm[1,1] + table_svm[1,2] + table_svm[2,1] + table_svm[2,2])
  precision_svm[[i]] <- table_svm[2,2] / (table_svm[2,2] + table_svm[2,1])
  recall_svm[[i]] <- table_svm[2,2] / (table_svm[2,2] + table_svm[1,2])
}

#Visualize accuracy scores of models with the boxplots
acc_log <- data.frame(Reduce(rbind, acc_log))
acc_svm <- data.frame(Reduce(rbind, acc_svm))
c <- data.frame(A=acc_log,B=acc_svm)
colnames(c) <- c("Logistic regression","Support vector machine")
boxplot(c, main = "Accuracy scores of methods", xlab = "Accuracy of logistic and support vector machine", ylab="Accuracy Score",  border = "black")

#Visualize precision scores of models with the boxplots
precision_log <- data.frame(Reduce(rbind, precision_log))
precision_svm <- data.frame(Reduce(rbind, precision_svm))
c <- data.frame(A=precision_log,B=precision_svm)
colnames(c) <- c("Logistic regression","Support vector machine")
boxplot(c, main = "Precision scores of methods", border = "black")

#Visualize recall scores of models with the boxplots
recall_log <- data.frame(Reduce(rbind, recall_log))
recall_svm <- data.frame(Reduce(rbind, recall_svm))
c <- data.frame(A=recall_log,B=recall_svm)
colnames(c) <- c("Logistic regression","Support vector machine")
boxplot(c, main = "Recall scores of methods",  border = "black")


#4) Results with different levels of correlation
#Create different correlations
m1 <- matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), nrow = 3)
m2 <- matrix(c(1, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 1), nrow = 3)
m3 <- matrix(c(1, 0.6, 0.6, 0.6, 1, 0.6, 0.6, 0.6, 1), nrow = 3)
m4 <- matrix(c(1, 0.9, 0.9, 0.9, 1, 0.9, 0.9, 0.9, 1), nrow = 3)
R <- list(m1, m2, m3, m4) # use 'list' instead of 'c' to create a list of matrices

#Create lists for the datasets
x_new <- list(); y_new <- list()
df_new <- list(); sample_new <- list()
train_new <- list(); test_new <- list()

#Create dataset
nobs <- 500
for (i in 1:1000){
  x_new[[i]] <-  rmvnorm(nobs, sigma = R[[4]]) #Change the R[[i]] for the every run
  colnames(x_new[[i]]) <- c("x1", "x2", "x3")
  eta <- -1 + 0.5 * x_new[[i]][ ,1] + 0.5 * x_new[[i]][, 2] - 2 * x_new[[i]][ ,3]
  pr <- exp(eta) / (1 + exp(eta))
  y_new[[i]] <- rbinom(nobs, size = 1, prob = pr)
  df_new[[i]] = data.frame(y = y_new[[i]],x = x_new[[i]])
  sample_new[[i]] = sample.split(df_new[[i]],SplitRatio = 0.8) 
  train_new[[i]] = subset(df_new[[i]],sample_new[[i]] == TRUE) 
  test_new[[i]] = subset(df_new[[i]], sample_new[[i]] == FALSE)
}

#Create lists for logistic regression
mod_new <- list(); probs_new <- list(); glm.probs_new <- list()
threshold_new <- list(); pred_log_new <- list()
table_log_new <- list(); acc_log_new <- list()
precision_log_new <- list(); recall_log_new <- list()

#Apply logistic regression 
for (i in 1:1000){
  mod_new[[i]] <- glm(y ~ x.x1 + x.x2 + x.x3, data = train_new[[i]], family = binomial)
  probs_new[[i]] <- predict(mod_new[[i]], newdata = test_new[[i]], type = "response")
  glm.probs_new[[i]] <- roc(test_new[[i]]$y ~ probs_new[[i]], plot = FALSE, print.auc = FALSE)
  cc <- coords(glm.probs_new[[i]], x = "best", ret = "all")
  threshold_new[[i]] <- cc["threshold"]$threshold
  pred_log_new[[i]] <- probs_new[[i]] > threshold_new[[i]]
  table_log <- table(preds = pred_log_new[[i]], true = test_new[[i]]$y)
  acc_log_new[[i]] <- (table_log[1,1] + table_log[2,2]) / (table_log[1,1] + table_log[1,2] + table_log[2,1] + table_log[2,2])
  precision_log_new[[i]] <- table_log[2,2] / (table_log[2,2] + table_log[2,1])
  recall_log_new[[i]] <- table_log[2,2] / (table_log[2,2] + table_log[1,2])
}

#Implementation of support vector machines
#Create lists for support vector machines
svm.fit <- list(); pred_svm <- list(); acc_svm <- list()
table_svm <- list(); precision_svm <- list(); recall_svm <- list()

for (i in 1:1000){
  svm.fit[[i]] <- svm(y ~., data = train_new[[i]], type = "C-classification", kernel = "polynomial",degree = 3, gamma = 0.1, coef0 = 1, cost = 1)
  pred_svm[[i]] <- predict(svm.fit[[i]], test_new[[i]])
  table_svm <- table(preds = pred_svm[[i]], true = test_new[[i]]$y)
  acc_svm[[i]] <- (table_svm[1,1] + table_svm[2,2]) / (table_svm[1,1] + table_svm[1,2] + table_svm[2,1] + table_svm[2,2])
  precision_svm[[i]] <- table_svm[2,2] / (table_svm[2,2] + table_svm[2,1])
  recall_svm[[i]] <- table_svm[2,2] / (table_svm[2,2] + table_svm[1,2])
}

#Visualize results
acc_log_new <- data.frame(Reduce(rbind, acc_log_new))
acc_svm_new <- data.frame(Reduce(rbind, acc_svm))
precision_log_new <- data.frame(Reduce(rbind, precision_log_new))
precision_svm_new <- data.frame(Reduce(rbind, precision_svm))
recall_log_new <- data.frame(Reduce(rbind, recall_log_new))
recall_svm_new <- data.frame(Reduce(rbind, recall_svm))

a <- data.frame(A=acc_log_new,B=acc_svm_new)
b <- data.frame(A = precision_log_new, B = precision_svm_new)
c <- data.frame(A = recall_log_new, B = recall_svm_new)
colnames(a) <- c("Logistic regression","Support vector machine")
colnames(b) <- c("Logistic regression","Support vector machine")
colnames(c) <- c("Logistic regression","Support vector machine")

par(mfrow=c(1,3))
boxplot(a, main = "Accuracy scores of methods",  border = "black",cex.axes = 4)
boxplot(b, main = "Precision scores of methods", border = "black")
boxplot(c, main = "Recall scores of methods", border = "black")
