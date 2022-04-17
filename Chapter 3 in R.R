#Import libraries
library(tidyverse)
library(ggplot2)
library(corrplot)
library("caTools")
library("scorecard")
library(caret)
library(glue)
library(highcharter)
library(inspectdf)
library(glue)

#Import dataset
data <- read.csv("bank.csv", stringsAsFactors = FALSE)

#Make factors to some columns
data$job <- data$job %>% as.factor()
data$marital <- data$marital %>% as.factor()
data$education <- data$education %>% as.factor()
data$default <- data$default %>% as.factor()
data$housing <- data$housing %>% as.factor()
data$loan <- data$loan %>% as.factor()
data$contact <- data$contact %>% as.factor()
data$month <- data$month %>% as.factor()
data$poutcome <- data$poutcome %>% as.factor()
data$deposit <- data$deposit %>% as.factor()
data$deposit <- data$deposit %>% factor(levels = c('no','yes'),labels = c(1,0))

#1) Data decription
#Summary of target variable
summary(data)

#2) Exploratory data analysis

#Split train and test set
sample = sample.split(data,SplitRatio = 0.8) 
train = subset(data,sample == TRUE) 
test = subset(data, sample == FALSE)

#Apply the logistic regression
mod <- glm(deposit ~ ., data = train, family = binomial)
summary(mod)

#Find the optimal threshold
library(pROC)
glm.probs <- predict(mod, newdata = test, type = "response")
glm.roc <- roc(test$deposit ~ glm.probs, plot = TRUE, print.auc = TRUE)
coords(glm.roc, x = "best", ret = "all")

#Lasso regression
library(glmnet)
x <- model.matrix(deposit ~ . , data = data)
x <- x[, -1]
y <- data$deposit
grid <- 10 ^ seq(10, -2, length = 100)

set.seed(1)
train <- sample(1:nrow(x), nrow(x) / 2)
test <- (-train)
y.test <- y[test]
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid, family="binomial")
plot(lasso.mod)
summary(lasso.mod)

set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1, family="binomial")
plot(cv.out) 

bestlam <- cv.out$lambda.min

#Visualize the path plot:
plot(lasso.mod, xvar = "lambda", xlim=c(-3,-1)) 
abline(v = log(bestlam), lwd = 1.2, lty = "dashed")


lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])

a <- roc(y.test ~lasso.pred, plot = TRUE, print.auc = TRUE)
coords(a, x = "best", ret = "all")

#Print coeffiecnts
coef(cv.out, bestlam)

#Implement the Support Vector Machine
library(e1071)

#Implement svm with the linear kernel
svm_mod <- svm(deposit ~., data = train, type = "C-classification", kernel = "linear", scale = TRUE)
summary(svm_mod)

#Set up function for svm models
svm_metrics <- function(mod){
  pred_test <- predict(mod, test)
  tb <- table(predict = pred_test, truth = test$deposit)
  ac <- (tb[1,1] + tb[2,2]) / (tb[1,1] + tb[1,2] + tb[2,1] + tb[2,2])
  pr <- tb[2,2]/(tb[2,2]+tb[2,1])
  rc <- tb[2,2]/(tb[2,2]+tb[1,2])
  list(tb,ac,pr,rc)
}
svm_metrics(svm_mod)[2]
#SVM with the linear kernel
svm_mod2 <- svm(deposit ~., data = train, type = "C-classification", kernel = "linear", cost = 10, scale = TRUE)
svm_metrics(svm_mod2)[1]
svm_metrics(svm_mod2)[2]

#SVM with the polynomial kernel
svm_mod3 <- svm(deposit ~., data = train, type = "C-classification", kernel = "polynomial", degree = 2)
svm_metrics(svm_mod3)[2]

#Tuning the polynomial kernel
poly.tune <- tune.svm(deposit ~., data = train, kernel="polynomial",degree =2, cost = 1, gamma = c(0.1,1,10), scale = TRUE)

#apply the best parameters to the new model
tune_out_poly_best <- svm(deposit ~., data = train, type = "C-classification", kernel = "polynomial",degree = 4, gamma = 0.1, coef0 = 1, cost = 1)
svm_metrics(tune_out_poly_best)[2]

#apply the best parameters to the new model
svm_model_high <- svm(deposit ~., data = train, type = "C-classification", kernel = "polynomial", degree = 3,gamma = 0.1, coef0 = 1, cost = 1)
svm_metrics(svm_model_high)[2]

#radial
svm_model_rad <- svm(deposit ~., data = train, type = "C-classification", kernel = "radial", degree = 2)
svm_metrics(svm_model_rad)[2]

#Tuning radial
tune_out_rad_best <- svm(deposit ~., data = train, type = "C-classification", kernel = "radial",degree = 4,
                         gamma = 0.05, coef0 = 0.1, cost = 100)
svm_metrics(tune_out_rad_best)[2]

#The best is svm_model_rad
svm_metrics(svm_model_high)[1]
svm_metrics(svm_model_high)[2]
svm_metrics(svm_model_high)[3]
svm_metrics(svm_model_high)[4]

