#Load the required libraries
library(pROC) 
library(DMwR)
library(glmnet)
library(ROSE) 
library(yardstick)
library(dplyr)
library(pdp)
library(xgboost)
library(randomForest)
library(tidyverse) 
library(boot)
library(moments)
library(DataExplorer) 
library(caret)
library(mlr)
library(unbalanced)
library(vita) 
library(rBayesianOptimization) 
library(Matrix) 
library(mlbench)
library(caTools)

setwd("C:/Users/headway/Desktop/m/Projects/edwisor")

#loading data
df_train = read.csv("train.csv", header = T)
df_test = read.csv("test.csv", header = T)
head(df_train)
head(df_test)

#Summary of the train and test dataset
str(df_train)
str(df_test)

#converting to factor 
df_train$target=as.factor(df_train$target)

#Shape of Datasets
dim(df_train)
dim(df_test)

########################### Target class count in test data ###################

#Percenatge count of the target classes
table(df_train$target)/length(df_train$target)*100
#Count of target class in train data
table(df_train$target)

#Bar plot for the count of target classes 
Barplot=ggplot(df_train,aes(target))+theme_bw()+geom_bar(stat='count',fill='green') 
print(Barplot)                                                                                            

############################# Distribution Analysis ###########################

#Distribution of train and test attributes

#Distribution of train attributes
for (var in names(df_train)[c(2:202)]){ 
   target=df_train$target
   plot=ggplot(df_train,aes(x=df_train[[var]],fill=target))+ggtitle(var)+geom_density()+theme_classic() 
   print(plot)
}

#Distribution of test attributes
test_plot=plot_density(df_test[,c(1:201)],ggtheme = theme_classic(),geom_density_args=list(color='blue'))
print(test_plot)

#Apply functions to find mean,sd,skew,kurtosis values per row in train and test data

mean_train = apply(df_train[,-c(1,2)],MARGIN=1,FUN=mean) 
mean_test = apply(df_test[,-c(1)],MARGIN=1,FUN=mean)

sd_train = apply(df_train[,-c(1,2)],MARGIN=1,FUN=sd)
sd_test = apply(df_test[,-c(1)],MARGIN=1,FUN=sd)

skew_train = apply(df_train[,-c(1,2)],MARGIN=1,FUN=skewness)
skew_test = apply(df_test[,-c(1)],MARGIN=1,FUN=skewness)

kurtosis_train = apply(df_train[,-c(1,2)],MARGIN=1,FUN=kurtosis) 
kurtosis_test = apply(df_test[,-c(1)],MARGIN=1,FUN=kurtosis)

#Distribution of mean,standard deviation,skew,kurtosis values per row in train and test data

ggplot()+
#Mean value of train data
geom_density(data=df_train[,-c(1,2)],aes(x=mean_train),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
#Mean value of test data
geom_density(data=df_test[,-c(1)],aes(x=mean_test),kernel='gaussian',show.legend=TRUE,color='blue')+labs(x='mean values per row',title="Mean value distribution per row in train and test dataset")

ggplot()+
#standard deviation value of train data
geom_density(data=df_train[,-c(1,2)],aes(x=sd_train),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
#standard deviation value of test data
geom_density(data=df_test[,-c(1)],aes(x=sd_test),kernel='gaussian',show.legend=TRUE,color='black')+labs(x='sd values per row',title="standard deviation value distribution per row in train and test dataset")

ggplot()+
#Skew value of train data
geom_density(data=df_train[,-c(1,2)],aes(x=skew_train),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
#Skew value of test data
geom_density(data=df_test[,-c(1)],aes(x=skew_test),kernel='gaussian',show.legend=TRUE,color='green')+labs(x='Skew values per row',title="Skew value distribution per row in train and test dataset")

ggplot()+
#kurtosis value of train data
geom_density(data=df_train[,-c(1,2)],aes(x=kurtosis_train),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
#kurtosis value of test data
geom_density(data=df_test[,-c(1)],aes(x=kurtosis_test),kernel='gaussian',show.legend=TRUE,color='blue')+labs(x='kurtosis values per row',title="kurtosis value distribution per row in train and test dataset")

#Distribution of mean,standard deviation,skew,kurtosis values per column in train and test data

ggplot()+ 
#Mean value of train data 
geom_density(aes(x=mean_train),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+ 
#Mean value of test data
geom_density(aes(x=mean_test),kernel='gaussian',show.legend=TRUE,color='green')+labs(x='mean values per column',title="Mean value distribution per column in train and test dataset")

ggplot()+ 
#standard deviation value of train data 
geom_density(aes(x=sd_train),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+ 
#standard deviation value of test data
geom_density(aes(x=sd_test),kernel='gaussian',show.legend=TRUE,color='black')+labs(x='standard deviation values per column',title="standard deviation value distribution per column in train and test dataset")

ggplot()+ 
#skew value of train data 
geom_density(aes(x=skew_train),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+ 
#skew value of test data
geom_density(aes(x=skew_test),kernel='gaussian',show.legend=TRUE,color='yellow')+labs(x='skew values per column',title="skew value distribution per column in train and test dataset")

ggplot()+ 
#kurtosis value of train data 
geom_density(aes(x=kurtosis_train),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+ 
#kurtosis value of test data
geom_density(aes(x=kurtosis_test),kernel='gaussian',show.legend=TRUE,color='blue')+labs(x='kurtosis values per column',title="kurtosis value distribution per column in train and test dataset")


########################## Missing Value Analysis #################################

#Checking missing values in train data set
missing_value_train = data.frame(missing_values=apply(df_train,2,function(x){sum(is.na(x))}))
#Sum of missing values in train set
missing_value = sum(missing_value_train)
missing_value

#Checking missing values in test data set
missing_value_test = data.frame(missing_values=apply(df_test,2,function(x){sum(is.na(x))}))
#Sum of missing values in train set
missing_value = sum(missing_value_test)
missing_value

#### Correlation Analysis ###

#correlation analysis of train and test data
df_train$target = as.numeric(df_train$target)

#correlations
correlation_train = cor(df_train[,c(2:202)])
correlations_test = cor(df_test[,c(2:201)])
#Checking
correlation_train
correlations_test

################################ Feature Engineering ##############################

#Splitting the train data 
train_index = sample(1:nrow(df_train),0.70*nrow(df_train))
data_train = df_train[train_index,] 
val_data = df_train[-train_index,]

#Training the Random forest classifier 
data_train$target = as.factor(data_train$target) 
mtry = floor(sqrt(150)) 
tuneGrid = expand.grid(.mtry=mtry) 
random_forest = randomForest(target~.,data_train[,-c(1)],mtry=mtry,ntree=8,importance=TRUE) 

#Important variables
Imp_var = importance(random_forest,type=2) 
Imp_var

################################## Modelling ######################################

##### Using CreateDataPartition splitting the data #####

train.index = createDataPartition(df_train$target,p=0.7,list=FALSE)
data.train = df_train[train_index,] 
data.val= df_train[-train_index,]

#Dataset for training
X_train = as.matrix(data.train[,-c(1,2)]) 
y_train = as.matrix(data.train$target)

#Dataset for testing 
test = as.matrix(df_test[,-c(1)])

#Dataset for validation
X_val = as.matrix(data.val[,-c(1,2)])
y_val = as.matrix(data.val$target)


################################### Logistic Regression ###########################

logreg = glmnet(X_train,y_train,family = "binomial")
summary(logreg)

#cross value prediction
crossvalidation_lr = cv.glmnet(X_train,y_train,type.measure ="class",family ="binomial")
crossvalidation_lr$lambda.min
plot(crossvalidation_lr)

#Checking performance on validation dataset 
crossvalid_predict.lr = predict(crossvalidation_lr,X_val,X="lambda.min",type="class") 
crossvalid_predict.lr

#conversion of actual and predicted target variable to factor

#Actual target variable conversion
target = data.val$target 
#convert to factor 
target = as.factor(target)

#predicted target variable conversion
crossvalid_predict.lr = as.factor(crossvalid_predict.lr)
crossvalid_predict.lr

###### Confusion matrix ######
confusionMatrix(data=crossvalid_predict.lr,reference=target)

########## ROC_AUC ###########
crossvalid_predict.lr = as.numeric(crossvalid_predict.lr) 
roc(data=data.val[,c(2)],response=target,predictor=crossvalid_predict.lr,plot=TRUE,auc=TRUE)

#predict the model on test data
logreg_test_pred = predict(logreg,test,type='class')
logreg_test_pred

##################################### ROSE #######################################

library(ROSE)
rose.train = ovun.sample(target~.,data =data.train[,-c(1)])$data
table(rose.train$target) 
rose.val = ovun.sample(target~.,data =data.val[,-c(1)])$data
table(rose.val$target)

#Apllying Logistic regression model 
logreg_rose = glmnet(as.matrix(rose.train),as.matrix(rose.train$target),family ="binomial")
summary(logreg_rose)

#Cross validation prediction
crossvalidation_rose = cv.glmnet(data.matrix(rose.val),data.matrix(rose.val$target),type.measure ="class",family ="binomial") 
crossvalidation_rose$lambda.min
plot(crossvalidation_rose)

#Checking performance on validation dataset 
crossvalidation_predict_rose = predict(crossvalidation_rose,data.matrix(rose.val),s="lambda.min",type ="class")
plot(crossvalidation_predict_rose)

#conversion of actual and predicted target variable to factor

#Actual target variable conversion
target = rose.val$target 
#convert to factor 
target = as.factor(target)

#predicted target variable conversion
crossvalidation_predict_rose = as.factor(crossvalidation_predict_rose)

###### Confusion matrix ######
confusionMatrix(data=crossvalidation_predict_rose,reference=target)

########## ROC_AUC ###########
crossvalidation_predict_rose = as.numeric(crossvalidation_predict_rose) 
roc(data=data.val[,c(2)],response=target,predictor=crossvalidation_predict_rose,plot=TRUE,auc=TRUE)

################################### XGBOOST #####################################

#Converting the data frame to matrix
X_train = as.matrix(data.train[,-c(1,2)]) 
y_train = as.matrix(data.train$target) 
X_val = as.matrix(data.val[,-c(1,2)])
y_val = as.matrix(data.val$target) 
test_data = as.matrix(df_test[,-c(1)])

#training dataset
xgb.train =  xgb.DMatrix(data=X_train,label=y_train)
#Validation dataset 
xgb.val =  xgb.DMatrix(data=X_val,label=y_val)

#Setting parameters
params = list(booster = "gbtree",objective = "binary:logistic",eta=0.3,gamma=0,max_depth=6,min_child_weight=1,subsample=1,colsample_bytree=1)
xgb = xgb.train(params = params,data = xgb.train,nrounds = 131,eval_freq = 1000,watchlist = list(val=xgb.val,train=xgb.train),print_every_n = 10, 
                early_stop_round = 10,maximize = F,eval_metric = "auc")


#lgbm model performance on test data 
xgb_pred_prob = predict(xgb,test_data) 
print(xgb_pred_prob) 
#Convert to binary output (1 and 0) with threshold 0.5 
xgb_pred = ifelse(xgb_pred_prob>0.5,1,0) 
print(xgb_pred)

#view variable importance plot
tree_imp = xgb.importance(feature_names = colnames(X_val),model = xgb)
xgb.plot.importance(importance_matrix = tree_imp,top_n = 50,measure = "Gain") 

############################# Saving the model ###################################

df_submission = data.frame(ID_code=df_test$ID_code,xgb_predict_prob=xgb_pred_prob,xgb_predict=xgb_pred) 
write.csv(df_submission,'submission.CSV',row.names=F)
head(df_submission)
