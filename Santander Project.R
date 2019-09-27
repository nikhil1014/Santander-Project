rm(list=ls())

                                                       ###Load required libraries
library("corrgram")
library("DMwR")
library("caret")
library("class")
library("randomForest")
library("dplyr")
library("dmm")
library("tidyverse")
library("Matrix")
library("caTools")
library("DataExplorer")
library("mlr")
library("e1071")
library("corrplot")
library("ggplot2")
library("pROC")
library("mlbench")
    
                                                        ###IMPORTING DATASET
#Set working directory
setwd("E:/Analytics/Edwisor/R Programming")
getwd()

#Load train dataset
train=read.csv("train.csv",header=T)

#Load test dataset
test=read.csv("test.csv",header=T)

                                                        ###DATA COLLECTION
head(train)
head(test)
dim(train)
dim(test)
str(train)
str(test)
#Count of target variable
table(train$target)
#Percentage counts of target class
table(train$target)/length(train$target)*100

#Checking Multicollinearity
library(usdm)
vif(train[,3:202])
vifcor(train[,3:202],th=0.9)


                                                       ###DATA VISUALIZATION

##Typecasting the target variable 
#convert numeric to factor
train$target=as.factor(train$target)
#Barplot for count of target classes
ggplot(train,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightblue')

#Outliers (using boxplot)
boxplot(train[,2:27])
boxplot(train[,28:53])
boxplot(train[,53:78])
boxplot(train[,78:103])
boxplot(train[,103:128])
boxplot(train[,128:153])
boxplot(train[,153:178])
boxplot(train[,178:202])

##Distribution of train attributes 
#From 3 to 102
for(var in names(train)[c(3:102)]){
  target=train$target
  plot=ggplot(train,aes(x=train[[var]],fill=target))+geom_density(kernel='gaussian')+
  ggtitle(var)+theme_classic()
  print(plot)
}

#From 103 to 202
for(var in names(train)[c(103:202)]){
  target=train$target
  plot=ggplot(train,aes(x=train[[var]],fill=target))+geom_density(kernel='gaussian')+
  ggtitle(var)+theme_classic()
  print(plot)
}

##Distribution of test attributes 
#From 2 to 101
plot_density(test[,c(2:101)],ggtheme=theme_classic(),geom_density_args = list(color='blue'))

#From 102 to 201
plot_density(test_df[,c(102:201)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))



                                                        ###DATA PRE-PROCESSING    

#Finding missing values in train data
missing_val=data.frame(missing_val=apply(train,2,function(x){sum(is.na(x))}))
missing_val=sum(missing_val)
print(missing_val)

#Finding missing values in test data
missing_val=data.frame(missing_val=apply(test,2,function(x){sum(is.na(x))}))
missing_val=sum(missing_val)
print(missing_val)

##Correlation Analysis
#Correlations in train data
#convert factor to int
train$target=as.numeric(train$target)
train_correlations=cor(train[,c(2:201)])
#Plotting Correlation plot for some values
train_corr=cor(train[,c(1:30)])
corrplot(train_correlations,type="upper",order="hclust",tl.col="black")

#Correlations in test data
test_correlations=cor(test[,c(2:201)])

                                                         ###FEATURE ENGINEERING

##Variable Importance
#convert numeric to factor
train$target=as.factor(train$target)
#Split the training data using simple random sampling
train_index=sample(1:nrow(train),0.75*nrow(train))
#Train data
train_data=train[train_index,]
#Validation data
test_data=train[-train_index,]
#Dimensions of train and validation data
dim(train_data)
dim(test_data)

#Let us build simple model to find features which are important
set.seed(2732)
train_data$target=as.factor(train_data$target)
mtry=sqrt(200)
tuneGrid=expand.grid(.mtry=mtry)
#Develop model
RF_Model=randomForest(target~.,train_data[,-c(1)],mtry=mtry,ntree=100,importance=TRUE)
#Feature Importance by Random Forest
varImp=importance(RF_Model,type=2)
varImp



                                                          ###SPLITTING DATASET USING RANDOM SAMPLING

#convert numeric to factor
train$target=as.factor(train$target)
#Split the training data using simple random sampling
train_index=sample(1:nrow(train),0.80*nrow(train))
#Training Dataset
train_data=train[train_index,]
#Validation data
test_data=train[-train_index,]
#Dimensions of train and validation data
dim(train_data)
dim(valid_data)



                                                          ###LOGISTIC REGRESSION

#Logistic Regression Model
LGR_Model=glm(target~.,data=train_data,family="binomial")
summary(LGR_Model)
#Predict using logistic regression
LGR_Predictions=predict(LGR_Model,newdata=test_data[,-c(1,2)],type="response")
#Convert Probabilities
LGR_Predictions=ifelse(LGR_Predictions>0.5,1,0)
#Evaluate the performance of model
ConfMatrix_RF=table(test_data$target,LGR_Predictions)
confusionMatrix(ConfMatrix_RF)
#Model Performance on TEST Dataset
Predict_Transaction=predict(LGR_Model,test,type = 'response')
Predict_Transaction=ifelse(Predict_Transaction>0.5,1,0)
print(Predict_Transaction)
#ROC Plot & AUC Score
predictionwithprobs=predict(LGR_Model,test_data,type = 'class')
predictionwithprobs=as.numeric(predictionwithprobs)
roc=roc(test_data$target,predictionwithprobs,ordered=TRUE,plot=TRUE)
plot(roc,col="red",lwd=3,main="ROC Curve for LGR_Model")
auc=auc(roc)


                                                          ###Algorithm 1 (RANDOM FOREST)

#convert numeric to factor
train$target=as.factor(train$target)
#Setting no.of variables for no.of trees
mtry=sqrt(200)
tuneGrid=expand.grid(.mtry=mtry)
#Develop model
RF_Model=randomForest(target~.,train_data[,-c(1,2)],mtry=mtry,ntree=100,importance=TRUE)
print(RF_Model)
#Predict using Random Forest
RF_Predictions=predict(RF_Model,test_data[,-c(1,2)])
#Evaluate the performance of model
ConfMatrix_RF=table(test_data,RF_Predictions)
confusionMatrix(ConfMatrix_RF)
#Model Performance on TEST Dataset
Predict_Transaction=predict(RF_Model,test,type = 'class')
print(Predict_Transaction)
#ROC Plot & AUC Score
predictionwithprobs=predict(RF_Model,test_data,type = 'class')
predictionwithprobs=as.numeric(predictionwithprobs)
roc=roc(test_data$target,predictionwithprobs,ordered=TRUE,plot=TRUE)
plot(roc,col="red",lwd=3,main="ROC Curve for RF_Model")
auc=auc(roc)


                                                           ###Algorithm 2 (NAIVE BAYES)

#Develop Model
NB_Model=naiveBayes(target~.,data=train_data)
#Predict on test cases
NB_Predictions=predict(NB_Model,test_data[,-c(1,2)],type='class')
print(NB_Predictions)
#Evaluate the performance of model
ConfMatrix_RF=table(observed=test_data$target,predicted=NB_Predictions)
confusionMatrix(ConfMatrix_RF)
#Model Performance on TEST Dataset
Predict_Transaction=predict(NB_Model,test,type = 'class')
print(Predict_Transaction)
#ROC Plot & AUC Score
predictionwithprobs=predict(NB_Model,test_data,type = 'class')
predictionwithprobs=as.numeric(predictionwithprobs)
roc=roc(test_data$target,predictionwithprobs,ordered=TRUE,plot=TRUE)
plot(roc,col="red",lwd=3,main="ROC Curve for NB_Model")
auc=auc(roc)


                                                           ###Algorithm 3 (SVM-STATE VECTOR MACHINE)

#Normalization
normalize=function(x){
  return((x-min(x))/(max(x)-min(x)))
}

#Taking 1lac obs. with same variables and scaled it
train_scaled=as.data.frame(lapply(train[1:100000,-1],normalize))
#convert numeric to factor
train_scaled$target=as.factor(train_scaled$target)

#Split the training data using simple random sampling
train_index=sample(1:nrow(train_scaled),0.80*nrow(train_scaled))
#Training Dataset
train_data=train_scaled[train_index,]
#Validation data
test_data=train_scaled[-train_index,]
#Dimensions of train and validation data
dim(train_data)
dim(valid_data)

#Develop Model
SVM_Model=svm(target~.,data=train_scaled)
#Predict on test cases
SVM_Predictions=predict(SVM_Model,test_data[,-1])
#Evaluate the performance of model
ConfMatrix_RF=table(observed=test_data$target,predicted=SVM_Predictions)
confusionMatrix(ConfMatrix_RF)
#Model Performance on TEST Dataset
Predict_Transaction=predict(SVM_Model,test,type = 'class')
print(Predict_Transaction)
#ROC Plot & AUC Score
predictionwithprobs=predict(SVM_Model,test_data,type = 'class')
predictionwithprobs=as.numeric(predictionwithprobs)
roc=roc(test_data$target,predictionwithprobs,ordered=TRUE,plot=TRUE)
plot(roc,col="red",lwd=3,main="ROC Curve for SVM")
auc=auc(roc)

