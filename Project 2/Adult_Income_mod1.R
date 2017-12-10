library(caTools)
library(kernlab)
library(ROCR)
library(rpart)
library(rpart.plot)
library(rattle)
library(caret)
library(ROCR)
library(ggplot2)
library(xgboost)
setwd("'E:/AML - BUAN 6341")
data<-read.csv('adult.data.csv',header = FALSE)
colnames(data)<-c('age','workclass','fnlwgt','education','education-num','marital-status',
                  'occupation','relationship','race','sex','capital-gain','capital-loss',
                  'hours-per-week','native_country','class')
summary(data)
str(data)
data$workclass = as.character(data$workclass)
data$occupation = as.character(data$occupation)
data$native_country = as.character(data$native_country)
data$education = as.character(data$education)
data$race = as.character(data$race)
data$marital = as.character(data$marital)

data$workclass = gsub("^Federal-gov","Federal-Govt",data$workclass)
data$workclass = gsub("^Local-gov","Other-Govt",data$workclass)
data$workclass = gsub("^State-gov","Other-Govt",data$workclass)
data$workclass = gsub("^Private","Private",data$workclass)
data$workclass = gsub("^Self-emp-inc","Self-Employed",data$workclass)
data$workclass = gsub("^Self-emp-not-inc","Self-Employed",data$workclass)
data$workclass = gsub("^Without-pay","Not-Working",data$workclass)
data$workclass = gsub("^Never-worked","Not-Working",data$workclass)

data$occupation = gsub("^Adm-clerical","Admin",data$occupation)
data$occupation = gsub("^Armed-Forces","Military",data$occupation)
data$occupation = gsub("^Craft-repair","Blue-Collar",data$occupation)
data$occupation = gsub("^Exec-managerial","White-Collar",data$occupation)
data$occupation = gsub("^Farming-fishing","Blue-Collar",data$occupation)
data$occupation = gsub("^Handlers-cleaners","Blue-Collar",data$occupation)
data$occupation = gsub("^Machine-op-inspct","Blue-Collar",data$occupation)
data$occupation = gsub("^Other-service","Service",data$occupation)
data$occupation = gsub("^Priv-house-serv","Service",data$occupation)
data$occupation = gsub("^Prof-specialty","Professional",data$occupation)
data$occupation = gsub("^Protective-serv","Other-Occupations",data$occupation)
data$occupation = gsub("^Sales","Sales",data$occupation)
data$occupation = gsub("^Tech-support","Other-Occupations",data$occupation)
data$occupation = gsub("^Transport-moving","Blue-Collar",data$occupation)

data$native_country[data$native_country=="Cambodia"] = "Asia"
data$native_country[data$native_country=="Canada"] = "North-America"    
data$native_country[data$native_country=="China"] = "Asia"     
data$native_country[data$native_country=="Columbia"] = "South-America"    
data$native_country[data$native_country=="Cuba"] = "North-America"      
data$native_country[data$native_country=="Dominican-Republic"] = "North-America"
data$native_country[data$native_country=="Ecuador"] = "South-America"     
data$native_country[data$native_country=="El-Salvador"] = "South-America" 
data$native_country[data$native_country=="England"] ="Europe"
data$native_country[data$native_country=="France"] = "Europe"
data$native_country[data$native_country=="Germany"] = "Europe"
data$native_country[data$native_country=="Greece"] = "Europe"
data$native_country[data$native_country=="Guatemala"] ="North-America"
data$native_country[data$native_country=="Haiti"] = "North-America"
data$native_country[data$native_country=="Holand-Netherlands"] = "Europe"
data$native_country[data$native_country=="Honduras"] = "North-America"
data$native_country[data$native_country=="Hong"] = "Asia"
data$native_country[data$native_country=="Hungary"] = "Europe"
data$native_country[data$native_country=="India"] = "Asia"
data$native_country[data$native_country=="Iran"] = "Asia"
data$native_country[data$native_country=="Ireland"] = "Europe"
data$native_country[data$native_country=="Italy"] = "Europe"
data$native_country[data$native_country=="Jamaica"] = "North-America"
data$native_country[data$native_country=="Japan"] = "Asia"
data$native_country[data$native_country=="Laos"] = "Asia"
data$native_country[data$native_country=="Mexico"] = "North-America"
data$native_country[data$native_country=="Nicaragua"] = "North-America"
data$native_country[data$native_country=="Outlying-US(Guam-USVI-etc)"] = "North-America"
data$native_country[data$native_country=="Peru"] = "South-America"
data$native_country[data$native_country=="Philippines"] = "Asia"
data$native_country[data$native_country=="Poland"] = "Europe"
data$native_country[data$native_country=="Portugal"] = "Europe"
data$native_country[data$native_country=="Puerto-Rico"] = "North-America"
data$native_country[data$native_country=="Scotland"] = "Europe"
data$native_country[data$native_country=="South"] = "Europe"
data$native_country[data$native_country=="Taiwan"] = "Asia"
data$native_country[data$native_country=="Thailand"] = "Asia"
data$native_country[data$native_country=="Trinadad&Tobago"] = "North-America"
data$native_country[data$native_country=="United-States"] ="North-America"
data$native_country[data$native_country=="Vietnam"] = "Asia"
data$native_country[data$native_country=="Yugoslavia"] = "Europe"

data$education = gsub("^10th","Dropout",data$education)
data$education = gsub("^11th","Dropout",data$education)
data$education = gsub("^12th","Dropout",data$education)
data$education = gsub("^1st-4th","Dropout",data$education)
data$education = gsub("^5th-6th","Dropout",data$education)
data$education = gsub("^7th-8th","Dropout",data$education)
data$education = gsub("^9th","Dropout",data$education)
data$education = gsub("^Assoc-acdm","Associates",data$education)
data$education = gsub("^Assoc-voc","Associates",data$education)
data$education = gsub("^Bachelors","Bachelors",data$education)
data$education = gsub("^Doctorate","Doctorate",data$education)
data$education = gsub("^HS-Grad","HS-Graduate",data$education)
data$education = gsub("^Masters","Masters",data$education)
data$education = gsub("^Preschool","Dropout",data$education)
data$education = gsub("^Prof-school","Prof-School",data$education)
data$education = gsub("^Some-college","HS-Graduate",data$education)

data$marital[data$marital=="Married-AF-spouse"] = "Married"
data$marital[data$marital=="Married-civ-spouse"] = "Married"
data$marital[data$marital=="Married-spouse-absent"] = "Not-Married"
data$marital[data$marital=="Separated"] = "Not-Married"
data$marital[data$marital=="Divorced"] = "Not-Married"

data$race[data$race=="Amer-Indian-Eskimo"] = "American-Indian"
data$race[data$race=="Asian-Pac-Islander"] = "Asian"

is.na(data) = data=='?'
is.na(data) = data==' ?'
data = na.omit(data)

data$marital = factor(data$marital)
data$education = factor(data$education)
data$native_country = factor(data$native_country)
data$workclass = factor(data$workclass)
data$occupation = factor(data$occupation)
data$race = factor(data$race)
data$sex = factor(data$sex)
data$relationship = factor(data$relationship)
data$class<-as.factor(ifelse(data$class==data$class[1],0,1))
summary(data)



library(caTools)
set.seed(1)
split = sample.split(data$class, SplitRatio = 0.70)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)



#SVM
library(kernlab)
library(ROCR)
#Scaling
scaled_training_set<-training_set
scaled_test_set<-test_set
scaled_training_set[,c(1,3,5,11,12,13)]<-scale(scaled_training_set[,c(1,3,5,11,12,13)])
scaled_test_set[,c(1,3,5,11,12,13)]<-scale(scaled_test_set[,c(1,3,5,11,12,13)])
pred<-data.frame()
#SVM

svm_classifier<-function(train_data,test_data,k){
time_train<-system.time(svm<-ksvm(class~.,data=train_data,type="C-svc",kernel=k,prob.model=TRUE,C=1))
time_test<-system.time(svm_pred<-predict(svm,newdata=test_data[-15],type='prob'))
svm_pred_con<-predict(svm,newdata=test_data[-15],type='response')
pred<-prediction(svm_pred[,2],test_data$class)  
performance<-performance(pred,"tpr","fpr")
dd<-data.frame(FP = performance@x.values[[1]], TP = performance@y.values[[1]])
confusion<-confusionMatrix(svm_pred_con,test_data$class)
auc<-performance(pred, measure = 'auc')@y.values[[1]]
results<-list(Confustion_Matrix=confusion,AUC=auc,plotdata=dd,Accuracy=confusion$overall[1],time_train,time_test)
return(results)
}

svm_train_radial<-svm_classifier(train_data = scaled_training_set,test_data = scaled_training_set,k = "rbfdot")
svm_test_radial<-svm_classifier(train_data = scaled_training_set,test_data = scaled_test_set,k = "rbfdot")
svm_train_linear<-svm_classifier(train_data = scaled_training_set,test_data = scaled_training_set,k = "vanilladot")
svm_test_linear<-svm_classifier(train_data = scaled_training_set,test_data = scaled_test_set,k = "vanilladot")
svm_train_tangent<-svm_classifier(train_data = scaled_training_set,test_data = scaled_training_set,k = "tanhdot")
svm_test_tangent<-svm_classifier(train_data = scaled_training_set,test_data = scaled_test_set,k = "tanhdot")

roc_radial_train <-geom_line(data =svm_train_radial$plotdata, aes(x = FP, y = TP, color = 'Radial_SVM_Train'))
roc_radial_test <-geom_line(data =svm_test_radial$plotdata, aes(x = FP, y = TP, color = 'Radial_SVM_Test'))
roc_linear_train<-geom_line(data =svm_train_linear$plotdata, aes(x = FP, y = TP, color = 'linear_SVM_Train'))
roc_linear_test<- geom_line(data =svm_test_linear$plotdata, aes(x = FP, y = TP, color = 'linear_SVM_Test'))
roc_tangent_train<-geom_line(data =svm_train_tangent$plotdata, aes(x = FP, y = TP, color = 'Tangent_SVM_Train'))
roc_tangent_test<-geom_line(data =svm_test_tangent$plotdata, aes(x = FP, y = TP, color = 'Tangent_SVM_Test'))


#plotting ROC for test
ggplot()+roc_radial_test+roc_tangent_test+roc_linear_test+
  xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC")+
  scale_colour_manual(name="Legend",
                      values=c(Radial_SVM_Test="red",Tangent_SVM_Test="blue",linear_SVM_Test='green'))
#Learning Curves
#Error vs Train Size
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

tang_Lc<-list()
tang_error_test<-c()
tang_error_train<-c()
tang_predict_test<-list()
tang_time_Train_svm<-c()
tang_time_Test_svm<-c()
range<-c(100,1000,5000,9000)
for(i in range){
radial_Lc[[j]]<-svm_classifier(train_data = scaled_training_set[1:i,],test_data = scaled_training_set[1:i,],k='rbfdot')
radial_predict_test[[j]]<-svm_classifier(train_data = scaled_training_set[1:i,],test_data = scaled_test_set[1:i,],k='rbfdot')
radial_error_train[j]<-1-as.numeric(radial_Lc[[j]][4])
radial_error_test[j]<-1-as.numeric(radial_predict_test[[j]][4])
radial_time_Train_svm[j]<-as.numeric(radial_Lc[[j]][[5]][3])
radial_time_Test_svm[j]<-as.numeric(radial_predict_test[[j]][[6]][3])

linear_Lc[[j]]<-svm_classifier(train_data = scaled_training_set[1:i,],test_data = scaled_training_set[1:i,],k='vanilladot')
linear_predict_test[[j]]<-svm_classifier(train_data = scaled_training_set[1:i,],test_data = scaled_test_set[1:i,],k='vanilladot')
linear_error_train[j]<-1-as.numeric(linear_Lc[[j]][4])
linear_error_test[j]<-1-as.numeric(linear_predict_test[[j]][4])
linear_time_Train_svm[j]<-as.numeric(linear_Lc[[j]][[5]][3])
linear_time_Test_svm[j]<-as.numeric(linear_predict_test[[j]][[6]][3])

tang_Lc[[j]]<-svm_classifier(train_data = scaled_training_set[1:i,],test_data = scaled_training_set[1:i,],k='tanhdot')
tang_predict_test[[j]]<-svm_classifier(train_data = scaled_training_set[1:i,],test_data = scaled_test_set[1:i,],k='tanhdot')
tang_error_train[j]<-1-as.numeric(tang_Lc[[j]][4])
tang_error_test[j]<-1-as.numeric(tang_predict_test[[j]][4])
tang_time_Train_svm[j]<-as.numeric(tang_Lc[[j]][[5]][3])
tang_time_Test_svm[j]<-as.numeric(tang_predict_test[[j]][[6]][3])
j=j+1
}
radial_lc<-ggplot()+geom_point(aes(x=range,y=radial_error_train,color='Radial_Train'))+
  geom_line(aes(x=range,y=radial_error_train,color='Radial_Train'))+geom_line(aes(x=range,y=radial_error_test,color='Radial_Test'))+
  geom_point(aes(x=range,y=radial_error_test,color='Radial_Test'))+
  ggtitle("Learning Curve Radial SVM")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Radial_Train="red",Radial_Test="blue"))
  
Linear_lc<-ggplot()+geom_point(aes(x=range,y=linear_error_train,color='Linear_Train'))+
  geom_line(aes(x=range,y=linear_error_train,color='Linear_Train'))+geom_line(aes(x=range,y=linear_error_test,color='Linear_Test'))+
  geom_point(aes(x=range,y=linear_error_test,color='Linear_Test'))+
  ggtitle("Learning Curve Linear SVM")+
    xlab('Train Set Size')+
    ylab('Error')+
    scale_colour_manual(name="Legend",
                        values=c(Linear_Train="green",Linear_Test="black"))
                                 
tangent_lc<-ggplot()+geom_point(aes(x=range,y=tang_error_train,color='Tangent_Train'))+
    geom_line(aes(x=range,y=tang_error_train,color='Tangent_Train'))+geom_line(aes(x=range,y=tang_error_test,color='Tangent_Test'))+
  geom_point(aes(x=range,y=tang_error_test,color='Tangent_Test'))+
  ggtitle("Learning Curve Sigmoid SVM")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Tangent_Train="cyan",Tangent_Test="orange"))



#Error vs time

learning_Curve1<-ggplot()+geom_point(aes(x=radial_time_Train_svm,y=radial_error_train,color='Train'))+
  geom_line(aes(x=radial_time_Train_svm,y=radial_error_train,color='Train'))+geom_line(aes(x=radial_time_Test_svm,y=radial_error_test,color='Test'))+
  geom_point(aes(x=radial_time_Test_svm,y=radial_error_test,color='Test'))+
  ggtitle("Learning Curve Radial SVM")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue"))



learning_Curve1

learning_Curve2<-ggplot()+geom_point(aes(x=linear_time_Train_svm,y=linear_error_train,color='Train'))+
  geom_line(aes(x=linear_time_Train_svm,y=linear_error_train,color='Train'))+geom_line(aes(x=linear_time_Test_svm,y=linear_error_test,color='Test'))+
  geom_point(aes(x=linear_time_Test_svm,y=linear_error_test,color='Test'))+
  ggtitle("Learning Curve Linear SVM")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue"))

learning_Curve2

learning_Curve3<-ggplot()+geom_point(aes(x=tang_time_Train_svm,y=tang_error_train,color='Train'))+
  geom_line(aes(x=tang_time_Train_svm,y=tang_error_train,color='Train'))+geom_line(aes(x=tang_time_Test_svm,y=tang_error_test,color='Test'))+
  geom_point(aes(x=tang_time_Test_svm,y=tang_error_test,color='Test'))+
  ggtitle("Learning Curve Sigmoid SVM")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue"))

learning_Curve3

#cross validation with radial SVM
folds_svm = createFolds(scaled_training_set$class, k = 10)
cv_svm = lapply(folds_svm, function(x) {
  training_fold_svm= scaled_training_set[-x, ]
  test_fold_svm= scaled_training_set[x, ]
  classifier_svm<-ksvm(class~.,data=training_fold_svm,type="C-svc",kernel="rbfdot",prob.model=TRUE,C=1)
  y_pred_svm= predict(classifier_svm, newdata = test_fold_svm[-15])
  cm_svm = table(test_fold_svm[, 15], y_pred_svm)
  accuracy = (cm_svm[1,1] + cm_svm[2,2]) / (cm_svm[1,1] + cm_svm[2,2] + cm_svm[1,2] + cm_svm[2,1])
  return(accuracy)
})
accuracy_svm = mean(as.numeric(cv_svm))#accuracy increased from 0.8494695 to 0.85166252


#Decision Tree
library(rpart)
library(rpart.plot)
library(rattle)
library(caret)



tree_classifier<-function(train.data,test.data){
  
  
time_Train_tree<-system.time(tree<-rpart(class~.,train.data,parms = list(split='information'),
                       method = 'class',cp=-1))
time_Test_tree<-system.time(tree_pred<-predict(tree, newdata = test.data[-15], type = 'class'))
tree_pred_roc<-predict(tree, newdata = test.data[-15], type = 'prob')
pred_tree<-prediction(tree_pred_roc[,2],test.data$class)  
performance_tree<-performance(pred_tree,"tpr","fpr")
dd_tree<-data.frame(FP = performance_tree@x.values[[1]], TP = performance_tree@y.values[[1]])
confusion_tree<-confusionMatrix(tree_pred,test.data$class)
auc_tree<-performance(pred_tree, measure = 'auc')@y.values[[1]]
ptree<-prune(tree,cp=tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
tree_pred_prune = predict(ptree, newdata = test.data[-15], type = 'class')
tree_pred_prune_roc = predict(ptree, newdata = test.data[-15], type = 'prob')
pred_tree_prune<-prediction(tree_pred_prune_roc[,2],test.data$class)  
performance_tree_prune<-performance(pred_tree_prune,"tpr","fpr")
dd_tree_prune<-data.frame(FP = performance_tree_prune@x.values[[1]], TP = performance_tree_prune@y.values[[1]])
confusion_tree_prune<-confusionMatrix(tree_pred_prune,test.data$class)
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
fancyRpartPlot(rpart(class~.,data = training_set),tweak=0.6,cex=0.9)#default tree
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
range_tree<-c(100,1000,5000,9000)
for(l in range_tree){
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

prune_plot<-ggplot()+geom_point(aes(x=range_tree,y=error_train_ptree,color='Train_prune'))+
  geom_line(aes(x=range_tree,y=error_train_ptree,color='Train_prune'))+geom_line(aes(x=range_tree,y=error_test_ptree,color='Test_prune'))+
  geom_point(aes(x=range_tree,y=error_test_ptree,color='Test_prune'))+xlab("Train Size")+ylab("Error")+ggtitle("Learning Curve Pruned Tree")

prune_plot



learning_Curve_tree<-ggplot()+geom_point(aes(x=range_tree,y=error_train_tree,color='Train'))+
  geom_line(aes(x=range_tree,y=error_train_tree,color='Train'))+geom_line(aes(x=range_tree,y=error_test_tree,color='Test'))+
  geom_point(aes(x=range_tree,y=error_test_tree,color='Test'))+
    ggtitle("Learning Curve Decision Tree")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Train="red",Test="blue"))


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
folds_tree = createFolds(training_set$class, k = 10)
cv_tree = lapply(folds_tree, function(x) {
  training_fold_tree= training_set[-x, ]
  test_fold_tree= training_set[x, ]
  classifier_tree=rpart(class~.,training_set,parms = list(split='information'),
                        method = 'class')
  y_pred_tree= predict(classifier_tree, newdata = test_fold_tree[,-15],type="class")
  cm_tree = table(test_fold_tree[, 15], y_pred_tree)
  accuracy = (cm_tree[1,1] + cm_tree[2,2]) / (cm_tree[1,1] + cm_tree[2,2] + cm_tree[1,2] + cm_tree[2,1])
  return(accuracy)
})
accuracy_tree = mean(as.numeric(cv_tree))#accuracy increased from 0.831565 to 0.842


####Boosting
train_test<-data
features = names(train_test)
for (f in features) {
  if (class(train_test[[f]])=="factor") {
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.numeric(as.integer(factor(train_test[[f]], levels=levels)))
  }
}

for(i1 in 1:30162){
  if(train_test[i1,15]==1)
  {
    train_test[i1,15]=0
  }
  else if(train_test[i1,15]==2)
  {
    train_test[i1,15]=1
  }
}

library(caTools)
set.seed(1)
split1 = sample.split(train_test$class, SplitRatio = 0.70)
training_data = subset(train_test, split1 == TRUE)
testing_data = subset(train_test, split1 == FALSE)
#====================================================================
######################## Preparing for xgboost
dtrain_best = xgb.DMatrix(as.matrix(training_data[,-15]), 
                          label=training_data[,15])
dtest_best = xgb.DMatrix(as.matrix(testing_data[,-15]))

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
dtrain = xgb.DMatrix(as.matrix(train_xgb[,-15]), 
                     label=train_xgb[,15])
dtest = xgb.DMatrix(as.matrix(test_xgb[,-15]))
res1 = xgb.cv(xgb_param_adult,
              dtrain,
              nrounds=700,   # changed
              nfold=3,           # changed
              early_stopping_rounds=50,
              print_every_n = 10,
              verbose= 1)
best1<-res1$best_iteration
time_train_xgb<-system.time(xgb.fit1<-xgboost(data=dtrain,nrounds =10,params = list(scale_pos_weight=1,colsample_bytree = 0.7)))
preds_xgb <- ifelse(predict(xgb.fit1, newdata=as.matrix(test_xgb[,-15])) >= 0.5, 1, 0)
time_test_xgb<-system.time(preds_xgb_roc<-predict(xgb.fit1,newdata=as.matrix(test_xgb[,-15]),type="prob"))
confusion_xgb<-confusionMatrix(test_xgb[,15], preds_xgb)
pred_xgb<-prediction(preds_xgb_roc,test_xgb$class)  
performance_xgb<-performance(pred_xgb,"tpr","fpr")
dd_xgb<-data.frame(FP = performance_xgb@x.values[[1]], TP = performance_xgb@y.values[[1]])
auc_xgb<-performance(pred_xgb, measure = 'auc')@y.values[[1]]
results_xgb<-list(Confustion_Matrix=confusion_xgb,AUC=auc_xgb,plotdata=dd_xgb,Accuracy=confusion_xgb$overall[1],
              time_train_xgb,time_test_xgb)
return(results_xgb)
}




xgb.fit_best = xgb.train(xgb_param_adult, dtrain_best, best)
preds_xgb_best_test <- ifelse(predict(xgb.fit_best, newdata=as.matrix(testing_data[,-15])) >= 0.5, 1, 0)
preds_xgb_best_train <- ifelse(predict(xgb.fit_best, newdata=as.matrix(training_data[,-15])) >= 0.5, 1, 0)

preds_xgb_roc_best_test<-predict(xgb.fit_best,newdata=as.matrix(testing_data[,-15]),type="prob")
preds_xgb_roc_best_train<-predict(xgb.fit_best,newdata=as.matrix(training_data[,-15]),type="prob")
confusion_xgb_best_test<-confusionMatrix(testing_data[,15], preds_xgb_best_test)
confusion_xgb_best_train<-confusionMatrix(training_data[,15], preds_xgb_best_train)

pred_xgb_best<-prediction(preds_xgb_roc_best_test,testing_data$class)
pred_xgb_best_train<-prediction(preds_xgb_roc_best_train,training_data$class)
auc_xgb_best_train<-performance(pred_xgb_best_train, measure = 'auc')@y.values[[1]]
performance_xgb_best<-performance(pred_xgb_best,"tpr","fpr")
dd_xgb_best<-data.frame(FP = performance_xgb_best@x.values[[1]], TP = performance_xgb_best@y.values[[1]])
auc_xgb_best<-performance(pred_xgb_best, measure = 'auc')@y.values[[1]]


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
range_xg<-c(100,500,700,1000,2000,3000,4000,5000,6000,7000,9000)
for(i1 in range_xg){
  Lc_xgb[[j1]]<-xgb_boost(train_xgb = training_data[1:i1,],test_xgb =  training_data[1:i1,])
  predict_test_xgb[[j1]]<-xgb_boost(train_xgb =training_data[1:i1,],test_xgb = testing_data[1:i1,])
  error_train_xgb[j1]<-1-as.numeric(Lc_xgb[[j1]][4])
  error_test_xgb[j1]<-1-as.numeric(predict_test_xgb[[j1]][4])
  time_Train_xgb[j1]<-as.numeric(Lc_xgb[[j1]][[5]][3])
  time_Test_xgb[j1]<-as.numeric(predict_test_xgb[[j1]][[6]][3])
  j1=j1+1
}
learning_Curve_xgb<-ggplot()+geom_point(aes(x=range_xg,y=error_train_xgb,color='Train'))+
  geom_line(aes(x=range_xg,y=error_train_xgb,color='Train'))+geom_line(aes(x=range_xg,y=error_test_xgb,color='Test'))+
  geom_point(aes(x=range_xg,y=error_test_xgb,color='Test'))+
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

#feature importance
imp<-xgb.importance(feature_names = dimnames(dtrain_best)[[2]],model =xgb.fit_best )
head(imp)
xgb.plot.importance(imp[1:6])

#Basic xgboost tree plot
library(DiagrammeR)
bst <- xgboost(data = dtrain_best, max.depth = 2,
               eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
#basic xgboost tree
xgb.plot.tree(feature_names = colnames(dtrain_best), model =bst,render=FALSE)



#Comparing all models using roc
ggplot()+roc_xgb_btest+roc_xgb_test+roc_tree_test+roc_ptree_test+roc_radial_test+roc_tangent_test+roc_linear_test+xlab("False Positive")+ylab("True Positive")+ggtitle("ROC")


