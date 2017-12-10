setwd("'E:/AML - BUAN 6341")
mydata = read.csv('bank-full.csv',sep = ";")
library(ggplot2)
library(caret)
library(caretEnsemble)
library(ROSE)
library(mlbench)
library(DMwR)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(xgboost)

#Summary on dataset
summary(mydata)

str(mydata)

p_age <- ggplot(mydata, aes(factor(y), age)) + geom_boxplot(aes(fill = factor(y)))
p_age

p_balance <- ggplot(mydata, aes(factor(y), balance)) + geom_boxplot(aes(fill = factor(y)))
p_balance

p_day <- ggplot(mydata, aes(factor(y), day)) + geom_boxplot(aes(fill = factor(y)))
p_day

p_duration <- ggplot(mydata, aes(factor(y), duration)) + geom_boxplot(aes(fill = factor(y)))
p_duration

p_campaign <- ggplot(mydata, aes(factor(y), campaign)) + geom_boxplot(aes(fill = factor(y)))
p_campaign

p_pdays <- ggplot(mydata, aes(factor(y), pdays)) + geom_boxplot(aes(fill = factor(y)))
p_pdays

p_previous <- ggplot(mydata, aes(factor(y), previous)) + geom_boxplot(aes(fill = factor(y)))
p_previous

#Remove missing data
nrow(mydata) - sum(complete.cases(mydata))
dim(mydata)
mydata<-na.omit(mydata)
sum(is.na(mydata))
mydata$Class<-factor(mydata$y,levels=c("no","yes"),labels = c("0","1"))
mydata$y = NULL


#Splitting
library(caTools)
set.seed(1)
split = sample.split(mydata$Class, SplitRatio = 0.70)
training <- subset(mydata, split == TRUE)
test_set <- subset(mydata, split == FALSE)


#Sampling
training_set<-SMOTE(Class~.,training,perc.over = 100,perc.under = 200)
training_set<-training_set[sample(1:nrow(training_set)),]

#Scaling
scaled_training<-training_set
scaled_training[,c(1,6,10,12,13,14,15)]<-scale(scaled_training[,c(1,6,10,12,13,14,15)])

scaled_test<-test_set
scaled_test[,c(1,6,10,12,13,14,15)]<-scale(scaled_test[,c(1,6,10,12,13,14,15)])

library(kernlab)
library(ROCR)
#Scaling
# training_set<-training_set
# test_set<-test_set
# training_set[,c(1,6,10,12,13,14)]<-scale(training_set[,c(1,6,10,12,13,14)])
# test_set[,c(1,6,10,12,13,14)]<-scale(test_set[,c(1,6,10,12,13,14)])
# pred<-data.frame()
# #SVM

svm_classifier<-function(train_data,test_data,k){
  time_train<-system.time(svm<-ksvm(Class~.,data=train_data,type="C-svc",kernel=k,prob.model=TRUE))
  time_test<-system.time(svm_pred<-predict(svm,newdata=test_data[,-17],type='prob'))
  svm_pred_con<-predict(svm,newdata=test_data[,-17],type='response')
  pred<-prediction(svm_pred[,2],test_data$Class)  
  performance<-performance(pred,"tpr","fpr")
  dd<-data.frame(FP = performance@x.values[[1]], TP = performance@y.values[[1]])
  confusion<-confusionMatrix(svm_pred_con,test_data$Class)
  auc<-performance(pred, measure = 'auc')@y.values[[1]]
  results<-list(Confustion_Matrix=confusion,AUC=auc,plotdata=dd,Accuracy=confusion$overall[1],time_train,time_test)
  return(results)
}

svm_train_radial<-svm_classifier(train_data = scaled_training,test_data = scaled_training,k = "rbfdot")
svm_test_radial<-svm_classifier(train_data = scaled_training,test_data = scaled_test,k = "rbfdot")
svm_train_linear<-svm_classifier(train_data = scaled_training,test_data = scaled_training,k = "vanilladot")
svm_test_linear<-svm_classifier(train_data = scaled_training,test_data = scaled_test,k = "vanilladot")
svm_train_poly<-svm_classifier(train_data = scaled_training,test_data = scaled_training,k = "polydot")
svm_test_poly<-svm_classifier(train_data = scaled_training,test_data = scaled_test,k = "polydot")

roc_radial_train <-geom_line(data =svm_train_radial$plotdata, aes(x = FP, y = TP, color = 'Radial_SVM_Train'))
roc_radial_test <-geom_line(data =svm_test_radial$plotdata, aes(x = FP, y = TP, color = 'Radial_SVM_Test'))
roc_linear_train<-geom_line(data =svm_train_linear$plotdata, aes(x = FP, y = TP, color = 'linear_SVM_Train'))
roc_linear_test<- geom_line(data =svm_test_linear$plotdata, aes(x = FP, y = TP, color = 'linear_SVM_Test'))
roc_poly_train<-geom_line(data =svm_train_poly$plotdata, aes(x = FP, y = TP, color = 'poly_SVM_Train'))
roc_poly_test<-geom_line(data =svm_test_poly$plotdata, aes(x = FP, y = TP, color = 'poly_SVM_Test'))


#plotting ROC for test
ggplot()+roc_radial_test+roc_poly_test+roc_linear_test+
  xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC")+
  scale_colour_manual(name="Legend",
                      values=c(Radial_SVM_Test="red",poly_SVM_Test="blue",linear_SVM_Test='green'))

svm_test_radial$AUC#0.905842
svm_test_linear$AUC#0.8959326
svm_test_poly$AUC#0.8959333

svm_test_radial$Accuracy#0.7303155
svm_test_linear$Accuracy#0.7446918
svm_test_poly$Accuracy#0.7446918
#Learning Curves
#Error vs Train Size
set.seed(1)
i=1
j=1


poly_Lc<-list()
poly_error_test<-c()
poly_error_train<-c()
poly_predict_test<-list()
poly_time_Train_svm<-c()
poly_time_Test_svm<-c()
range_svm<-c(100,700,1000)
for(i in range_svm){
  poly_Lc[[j]]<-svm_classifier(train_data = scaled_training[1:i,],test_data = scaled_training[1:i,],k='tanhdot')
  poly_predict_test[[j]]<-svm_classifier(train_data = scaled_training[1:i,],test_data = scaled_test[1:i,],k='tanhdot')
  poly_error_train[j]<-1-as.numeric(poly_Lc[[j]][4])
  poly_error_test[j]<-1-as.numeric(poly_predict_test[[j]][4])
  poly_time_Train_svm[j]<-as.numeric(poly_Lc[[j]][[5]][3])
  poly_time_Test_svm[j]<-as.numeric(poly_predict_test[[j]][[6]][3])
  print(i)
  j=j+1
}
i=1
j=1
radial_Lc<-list()
radial_error_test<-c()
radial_error_train<-c()
radial_predict_test<-list()
radial_time_Train_svm<-c()
radial_time_Test_svm<-c()

linear_Lc<-list()
linear_error_test<-c()
linear_error_train<-c()
linear_predict_test<-list()
linear_time_Train_svm<-c()
linear_time_Test_svm<-c()
range<-c(700,1000,1500,2000,3000,4000,5000,9000,10000,13000)
for(i in range){
  radial_Lc[[j]]<-svm_classifier(train_data = scaled_training[1:i,],test_data = scaled_training[1:i,],k='rbfdot')
  radial_predict_test[[j]]<-svm_classifier(train_data = scaled_training[1:i,],test_data = scaled_test[1:i,],k='rbfdot')
  radial_error_train[j]<-1-as.numeric(radial_Lc[[j]][4])
  radial_error_test[j]<-1-as.numeric(radial_predict_test[[j]][4])
  radial_time_Train_svm[j]<-as.numeric(radial_Lc[[j]][[5]][3])
  radial_time_Test_svm[j]<-as.numeric(radial_predict_test[[j]][[6]][3])
  
  linear_Lc[[j]]<-svm_classifier(train_data = scaled_training[1:i,],test_data = scaled_training[1:i,],k='vanilladot')
  linear_predict_test[[j]]<-svm_classifier(train_data = scaled_training[1:i,],test_data = scaled_test[1:i,],k='vanilladot')
  linear_error_train[j]<-1-as.numeric(linear_Lc[[j]][4])
  linear_error_test[j]<-1-as.numeric(linear_predict_test[[j]][4])
  linear_time_Train_svm[j]<-as.numeric(linear_Lc[[j]][[5]][3])
  linear_time_Test_svm[j]<-as.numeric(linear_predict_test[[j]][[6]][3])
  j=j+1
}


ggplot()+geom_point(aes(x=range,y=radial_error_train,color='Radial_Train'))+
  geom_line(aes(x=range,y=radial_error_train,color='Radial_Train'))+
  geom_line(aes(x=range,y=radial_error_test,color='Radial_Test'))+
  geom_point(aes(x=range,y=radial_error_test,color='Radial_Test'))+
  ggtitle("Learning Curve Radial SVM")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",values=c(Radial_Train="red",Radial_Test="blue"))

ggplot()+geom_point(aes(x=range,y=linear_error_train,color='Linear_Train'))+
  geom_line(aes(x=range,y=linear_error_train,color='Linear_Train'))+geom_line(aes(x=range,y=linear_error_test,color='Linear_Test'))+
  geom_point(aes(x=range,y=linear_error_test,color='Linear_Test'))+
  ggtitle("Learning Curve Linear SVM")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Linear_Train="green",Linear_Test="black"))

ggplot()+geom_point(aes(x=range_svm,y=poly_error_train,color='poly_Train'))+
  geom_line(aes(x=range_svm,y=poly_error_train,color='poly_Train'))+geom_line(aes(x=range_svm,y=poly_error_test,color='poly_Test'))+
  geom_point(aes(x=range_svm,y=poly_error_test,color='poly_Test'))+
  ggtitle("Learning Curve poly SVM")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(poly_Train="cyan",poly_Test="orange"))



#Error vs time

learning_Curve1<-ggplot()+geom_point(aes(x=radial_time_Train_svm,y=radial_error_train,color='Train'))+
  geom_line(aes(x=radial_time_Train_svm,y=radial_error_train,color='Train'))+geom_line(aes(x=radial_time_Test_svm,y=radial_error_test,color='Test'))+
  geom_point(aes(x=radial_time_Test_svm,y=radial_error_test,color='Test'))+
  ggtitle("Learning Curve SVM")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue"))



learning_Curve1

#cross validation with radial SVM
folds_svm = createFolds(scaled_training$Class, k = 10)
cv_svm = lapply(folds_svm, function(x) {
  training_fold_svm= scaled_training[-x, ]
  test_fold_svm= scaled_training[x, ]
  classifier_svm<-ksvm(Class~.,data=training_fold_svm,type="C-svc",kernel="rbfdot",prob.model=TRUE,C=1)
  y_pred_svm= predict(classifier_svm, newdata = test_fold_svm[-17])
  cm_svm = table(test_fold_svm[, 17], y_pred_svm)
  accuracy = (cm_svm[1,1] + cm_svm[2,2]) / (cm_svm[1,1] + cm_svm[2,2] + cm_svm[1,2] + cm_svm[2,1])
  return(accuracy)
})
accuracy_svm = mean(as.numeric(cv_svm))#accuracy increased from 0.7303155 to 0.8758102


#Decision Tree
library(rpart)
library(rpart.plot)
library(rattle)
library(caret)
library(ROCR)


tree_classifier<-function(train.data,test.data){
  
  
  time_Train_tree<-system.time(tree<-rpart(train.data$Class~.,data=train.data,parms = list(split='information'),
                                           method = 'class',cp=-1))
  time_Test_tree<-system.time(tree_pred<-predict(tree, newdata = test.data[,-17], type = 'class'))
  tree_pred_roc<-predict(tree, newdata = test.data[,-17], type = 'prob')
  pred_tree<-prediction(as.numeric(tree_pred_roc[,2]),as.numeric(test.data$Class))  
  performance_tree<-performance(pred_tree,"tpr","fpr")
  dd_tree<-data.frame(FP = performance_tree@x.values[[1]], TP = performance_tree@y.values[[1]])
  confusion_tree<-confusionMatrix(tree_pred,test.data$Class)
  auc_tree<-performance(pred_tree, measure = 'auc')@y.values[[1]]
  ptree<-prune(tree,cp=tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
  tree_pred_prune = predict(ptree, newdata = test.data[,-17], type = 'class')
  tree_pred_prune_roc = predict(ptree, newdata = test.data[,-17], type = 'prob')
  pred_tree_prune<-prediction(tree_pred_prune_roc[,2],test.data$Class)  
  performance_tree_prune<-performance(pred_tree_prune,"tpr","fpr")
  dd_tree_prune<-data.frame(FP = performance_tree_prune@x.values[[1]], TP = performance_tree_prune@y.values[[1]])
  confusion_tree_prune<-confusionMatrix(tree_pred_prune,test.data$Class)
  auc_tree_prune<-performance(pred_tree_prune, measure = 'auc')@y.values[[1]]
  results_tree<-list(Decision_Tree=tree,Confustion_Matrix=confusion_tree,AUC=auc_tree,plotdata=dd_tree,Accuracy=confusion_tree$overall[1],
                     Decision_Tree_prune=ptree,Confustion_Matrix_prune=confusion_tree_prune,AUC_prune=auc_tree_prune,plotdata_prune=dd_tree_prune,
                     Accuracy_prune=confusion_tree_prune$overall[1],time_Train_tree=time_Train_tree,time_Test_tree=time_Test_tree)
}

tree_train<-tree_classifier(train.data = training_set,test.data = training_set)
tree_test<-tree_classifier(train.data =training_set,test.data = test_set)

roc_tree_train <-geom_line(data =tree_train$plotdata, aes(x = FP, y = TP, color = 'Decision_Tree_Train'))
roc_tree_test <-geom_line(data =tree_test$plotdata, aes(x = FP, y = TP, color = 'Decision_Tree_Test'))
roc_ptree_train<-geom_line(data =tree_train$plotdata_prune, aes(x = FP, y = TP, color = 'Prune_Tree_Train'))
roc_ptree_test <-geom_line(data =tree_test$plotdata_prune, aes(x = FP, y = TP, color = 'Prune_Tree_Test'))

#plotting ROC for tree
ggplot()+roc_tree_test+roc_ptree_test+
  xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC")+
  scale_colour_manual(name="Legend",
                      values=c(Decision_Tree_Test="red",Prune_Tree_Test="blue"))

#plotting the tree
plot(tree_test$Decision_Tree_prune)#pruned tree
fancyRpartPlot(rpart(Class~.,data = training_set),tweak=0.6,cex=0.9)#default tree
plot(tree_test$Decision_Tree)#fully grown tree
#Learning Curve 
l=1
m=1
Lc_tree<-list()
error_test_tree<-c()
error_train_tree<-c()
predict_test_tree<-list()
tree_time_Train<-c()
tree_time_Test<-c()
error_train_ptree<-c()
error_test_ptree<-c()
range<-c(700,1000,1500,2000,3000,4000,5000,9000,10000,13000)
for(l in range){
  Lc_tree[[m]]<-tree_classifier(train.data = training_set[1:l,],test.data = training_set[1:l,])
  predict_test_tree[[m]]<-tree_classifier(train.data = training_set[1:l,],test.data = test_set[1:l,])
  error_train_tree[m]<-1-as.numeric(Lc_tree[[m]][5])
  error_test_tree[m]<-1-as.numeric(predict_test_tree[[m]][5])
  error_train_ptree[m]<-1-as.numeric(Lc_tree[[m]][10])
  error_test_ptree[m]<-1-as.numeric(predict_test_tree[[m]][10])
  tree_time_Train[m]<-as.numeric(Lc_tree[[m]][[11]][3])
  tree_time_Test[m]<-as.numeric(predict_test_tree[[m]][[12]][3])
  m=m+1
}

prune_plot<-ggplot()+geom_point(aes(x=range,y=error_train_ptree,color='Train_prune'))+
  geom_line(aes(x=range,y=error_train_ptree,color='Train_prune'))+geom_line(aes(x=range,y=error_test_ptree,color='Test_prune'))+
  geom_point(aes(x=range,y=error_test_ptree,color='Test_prune'))

prune_plot+ggtitle("Learning Curve Pruned Tree")+xlab("Train Size")+ylab("Error")

learning_Curve_tree<-ggplot()+geom_point(aes(x=range,y=error_train_tree,color='Train'))+
  geom_line(aes(x=range,y=error_train_tree,color='Train'))+
  geom_line(aes(x=range,y=error_test_tree,color='Test'))+
  geom_point(aes(x=range,y=error_test_tree,color='Test'))+
  ggtitle("Learning Curve Decision Tree")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue",Train_prune="Green",Test_prune="black"))


learning_Curve_tree


#Error vs time 
learning_Curve1_tree<-ggplot()+geom_point(aes(x=tree_time_Train,y=error_train_tree,color='Train'))+
  geom_line(aes(x=tree_time_Train,y=error_train_tree,color='Train'))+geom_line(aes(x=tree_time_Test,y=error_test_tree,color='Test'))+
  geom_point(aes(x=tree_time_Test,y=error_test_tree,color='Test'))+
  ggtitle("Learning Curve Decision Tree")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue"))



learning_Curve1_tree
#Cross validation
folds_tree = createFolds(training_set$Class, k = 10)
cv_tree = lapply(folds_tree, function(x) {
  training_fold_tree= training_set[-x, ]
  test_fold_tree= training_set[x, ]
  classifier_tree=rpart(Class~.,training_set,parms = list(split='information'),
                        method = 'class')
  y_pred_tree= predict(classifier_tree, newdata = test_fold_tree[,-17],type="class")
  cm_tree = table(test_fold_tree[, 17], y_pred_tree)
  accuracy = (cm_tree[1,1] + cm_tree[2,2]) / (cm_tree[1,1] + cm_tree[2,2] + cm_tree[1,2] + cm_tree[2,1])
  return(accuracy)
})
accuracy_tree = mean(as.numeric(cv_tree))#accuracy increased from 0.8351519  to 0.8390045


####Boosting
library(xgboost)
data<-read.csv("bank-full.csv",sep = ";")
train_test<-data
features = names(train_test)
for (f in features) {
  if (class(train_test[[f]])=="factor") {
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.numeric(as.integer(factor(train_test[[f]], levels=levels)))
  }
}

for(i1 in 1:45211){
  if(train_test[i1,17]==1)
  {
    train_test[i1,17]=0
  }
  else if(train_test[i1,17]==2)
  {
    train_test[i1,17]=1
  }
}

library(caTools)
set.seed(1)
split1 = sample.split(train_test$y, SplitRatio = 0.70)
training_data = subset(train_test, split1 == TRUE)
testing_data = subset(train_test, split1 == FALSE)
#====================================================================
######################## Preparing for xgboost

dtrain_best = xgb.DMatrix(as.matrix(training_data[,-17]), 
                          label=training_data[,17])
dtest_best = xgb.DMatrix(as.matrix(testing_data[,-17]))

xgb_param_adult = list(
  nrounds = c(100),
  eta = 0.057,#eta between(0.01-0.2)
  max_depth = 4, #values between(3-10)
  subsample = 0.7,#values between(0.5-1)
  colsample_bytree = 0.7,#values between(0.5-1)
  num_parallel_tree=1,
  objective='binary:logistic',
  min_child_weight = 1,
  booster='gbtree',
  scale_pos_weight=1
)

res = xgb.cv(xgb_param_adult,
             dtrain_best,
             nrounds=700,   # changed
             nfold=3,           # changed
             early_stopping_rounds=50,
             print_every_n = 10,
             verbose= 1)
best<-res$best_iteration
xgb_boost<-function(train_xgb,test_xgb){
  dtrain = xgb.DMatrix(as.matrix(train_xgb[,-17]), 
                       label=train_xgb[,17])
  dtest = xgb.DMatrix(as.matrix(test_xgb[,-17]))
  res1 = xgb.cv(xgb_param_adult,
               dtrain,
               nrounds=700,   # changed
               nfold=3,           # changed
               early_stopping_rounds=50,
               print_every_n = 10,
               verbose= 1)
  best1<-res1$best_iteration
  time_train_xgb<-system.time(xgb.fit1<-xgboost(data=dtrain,nrounds = best1,params = list(scale_pos_weight=1,colsample_bytree = 0.7)))
  preds_xgb <- ifelse(predict(xgb.fit1, newdata=as.matrix(test_xgb[,-17])) >= 0.5, 1, 0)
  time_test_xgb<-system.time(preds_xgb_roc<-predict(xgb.fit1,newdata=as.matrix(test_xgb[,-17]),type="prob"))
  confusion_xgb<-confusionMatrix(test_xgb[,17], preds_xgb)
  pred_xgb<-prediction(preds_xgb_roc,test_xgb$y)  
  performance_xgb<-performance(pred_xgb,"tpr","fpr")
  dd_xgb<-data.frame(FP = performance_xgb@x.values[[1]], TP = performance_xgb@y.values[[1]])
  auc_xgb<-performance(pred_xgb, measure = 'auc')@y.values[[1]]
  results_xgb<-list(Confustion_Matrix=confusion_xgb,AUC=auc_xgb,plotdata=dd_xgb,Accuracy=confusion_xgb$overall[1],
                    time_train_xgb,time_test_xgb)
  return(results_xgb)
}




xgb.fit_best = xgb.train(xgb_param_adult, dtrain_best, best)
preds_xgb_best_test <- ifelse(predict(xgb.fit_best, newdata=as.matrix(testing_data[,-17])) >= 0.5, 1, 0)
preds_xgb_best_train <- ifelse(predict(xgb.fit_best, newdata=as.matrix(training_data[,-17])) >= 0.5, 1, 0)

preds_xgb_roc_best_test<-predict(xgb.fit_best,newdata=as.matrix(testing_data[,-17]),type="prob")
preds_xgb_roc_best_train<-predict(xgb.fit_best,newdata=as.matrix(training_data[,-17]),type="prob")
confusion_xgb_best_test<-confusionMatrix(testing_data[,17], preds_xgb_best_test)
confusion_xgb_best_train<-confusionMatrix(training_data[,17], preds_xgb_best_train)

pred_xgb_best<-prediction(preds_xgb_roc_best_test,testing_data$y)
pred_xgb_best_train<-prediction(preds_xgb_roc_best_train,training_data$y)
performance_xgb_best<-performance(pred_xgb_best,"tpr","fpr")
dd_xgb_best<-data.frame(FP = performance_xgb_best@x.values[[1]], TP = performance_xgb_best@y.values[[1]])
auc_xgb_best<-performance(pred_xgb_best, measure = 'auc')@y.values[[1]]
auc_xgb_best_train<-performance(pred_xgb_best_train, measure = 'auc')@y.values[[1]]


xgb_train<-xgb_boost(train_xgb = training_data,test_xgb = training_data)
xgb_test<-xgb_boost(train_xgb = training_data,test_xgb = testing_data)


roc_xgb_train <-geom_line(data =xgb_train$plotdata, aes(x = FP, y = TP, color = 'XgBoost_Train'))
roc_xgb_test <-geom_line(data =xgb_test$plotdata, aes(x = FP, y = TP, color = 'XgBoost_Test'))
roc_xgb_btest <-geom_line(data =dd_xgb_best, aes(x = FP, y = TP, color = 'XgBoost_Best_model'))

ggplot()+roc_xgb_btest+roc_xgb_test+
  xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC")+
  scale_colour_manual(name="Legend",
                      values=c(XgBoost_Test="red",XgBoost_Best_model="blue"))




#Error vs Train Size
i1=1
j1=1
Lc_xgb<-list()
error_test_xgb<-c()
error_train_xgb<-c()
predict_test_xgb<-list()
time_Train_xgb<-c()
time_Test_xgb<-c()
range<-c(700,1000,1500,2000,3000,4000,5000,9000,10000,13000)
for(i1 in range){
  Lc_xgb[[j1]]<-xgb_boost(train_xgb = training_data[1:i1,],test_xgb =  training_data[1:i1,])
  predict_test_xgb[[j1]]<-xgb_boost(train_xgb =training_data[1:i1,],test_xgb = testing_data[1:i1,])
  error_train_xgb[j1]<-1-as.numeric(Lc_xgb[[j1]][4])
  error_test_xgb[j1]<-1-as.numeric(predict_test_xgb[[j1]][4])
  time_Train_xgb[j1]<-as.numeric(Lc_xgb[[j1]][[5]][3])
  time_Test_xgb[j1]<-as.numeric(predict_test_xgb[[j1]][[6]][3])
  j1=j1+1
}
learning_Curve_xgb<-ggplot()+geom_point(aes(x=range,y=error_train_xgb,color='Train'))+
  geom_line(aes(x=range,y=error_train_xgb,color='Train'))+geom_line(aes(x=range,y=error_test_xgb,color='Test'))+
  geom_point(aes(x=range,y=error_test_xgb,color='Test'))+
  ggtitle("Learning Curve XgBoost")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue"))


learning_Curve_xgb
#Error vs time

learning_Curve1_xgb<-ggplot()+geom_point(aes(x=time_Train_xgb,y=error_train_xgb,color='Train'))+
  geom_line(aes(x=time_Train_xgb,y=error_train_xgb,color='Train'))+geom_line(aes(x=time_Test_xgb,y=error_test_xgb,color='Test'))+
  geom_point(aes(x=time_Test_xgb,y=error_test_xgb,color='Test'))+
  ggtitle("Learning Curve XgBoost")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue"))



learning_Curve1_xgb
#Basic xgboost tree plot
library(DiagrammeR)
bst <- xgboost(data = dtrain_best, max.depth = 2,
               eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
#basic xgboost tree
xgb.plot.tree(feature_names = colnames(dtrain_best), model =bst,render=FALSE)
#feature importance
imp<-xgb.importance(feature_names = dimnames(dtrain_best)[[2]],model =xgb.fit_best )
head(imp)
xgb.plot.importance(imp[1:6])


#Comparing all models using roc
ggplot()+roc_xgb_btest+roc_xgb_test+roc_tree_test+roc_ptree_test+roc_radial_test+roc_poly_test+roc_linear_test+xlab("False Positive")+ylab("True Positive")+ggtitle("ROC")

ggplot()+roc_xgb_btest+roc_ptree_test+roc_radial_test+xlab("False Positive")+ylab("True Positive")+ggtitle("ROC")



