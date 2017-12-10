#Libraries

library(caTools)
library(kernlab)
library(ROCR)
library(rpart)
library(rpart.plot)
library(rattle)
library(caret)
library(ROCR)
library(ggplot2)
library(dplyr)
library(factoextra)
library(fastICA)
library(lattice)
library(ggplot2)
library(moments)
library(microbenchmark)
#Data import


#Data import

data<-read.csv('bank-full.csv',sep = ";")

#Data preprocessing

nrow(data) - sum(complete.cases(data))
dim(data)
data<-na.omit(data)
sum(is.na(data))
data$class<-factor(data$y,levels=c("no","yes"),labels = c("0","1"))
data$y = NULL

#Clustering using all the features
data_cat_1<-data[c(2,3,4,5,7,8,9,11,16)]
data_dum_1<-dummyVars(~.,data = data_cat_1,fullRank = T)
dataset_1<-as.data.frame(predict(data_dum_1,data_cat_1))
data_num_1<-as.data.frame(data[,c(1,6,10,12,13,14,15)])

data_final_1<-cbind(data_num_1,dataset_1)
data_scale_1<-as.data.frame(scale(data_final_1))
summary(data_final_1)


#Kmeans Clustering
#Elbow method

set.seed(1)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(data_scale_1, i,nstart = 25)$withinss)

elbow1<-ggplot()+geom_point(aes(x=1:10,y=wcss),color="red")+geom_line(aes(x=1:10,y=wcss),color="spring green")+
  ggtitle("Elbow Method")+xlab("Number of Clusters")+ylab("WCSS")
elbow1
kmeans_opt_1 <- kmeans(x=data_scale_1, centers = 3, nstart = 25)
fviz_cluster(kmeans_opt_1, data = data_scale_1,
             ellipse = FALSE,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE
             
             
)
#Expectation-maximization

library(mclust)

BIC_1 <- mclustBIC(data_scale_1,initialization = list(subset=sample(1:nrow(data_scale_1), size=100)),modelNames=mclust.options("emModelNames"),control=emControl())

plot(BIC_1)

em_1<-Mclust(data_scale_1,x = BIC_1,modelNames=mclust.options("emModelNames"))



fviz_cluster(em_1, data = data_scale_1,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE,
             ellipse = FALSE)

summary(em_1)


#Task 5 running ANN along with clustering results

#ANN on the entire dataset with kmeans clustering results
#Results
# > ann_tanh_test3$AUC
# [1] 0.9101807
# > ann_tanh_test3$Accuracy
# [1] 0.8358753


data3<-data[-17]
data3$cluster<-as.factor(kmeans_opt_1$cluster)
data3$class<-data$class
set.seed(1)
split3 = sample.split(data3$class, SplitRatio = 0.70)
training_data3 = subset(data3, split3 == TRUE)
testing_data3 = subset(data3, split3 == FALSE)

# Feature Scaling
training_data3[,c(1,6,10,12,13,14,15)] = scale(training_data3[,c(1,6,10,12,13,14,15)])
testing_data3[,c(1,6,10,12,13,14,15)] = scale(testing_data3[,c(1,6,10,12,13,14,15)])

#Neural network: 

library(h2o)
h2o.init(nthreads = -1)
ann_classifier3<-function(train,test,act,hidden_layer){
  
  train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[c(1:17)]),y ='class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
  test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test[-18])))
  performance<-h2o.performance(ann,as.h2o(test))
  confusion_tree<-h2o.confusionMatrix(performance)
  AUC<-performance@metrics$AUC
  Accuracy<-1-confusion_tree$Error[3]
  dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
  results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
  return(results_ann)
}

#Activation Function

ann_rectifier_test3<-ann_classifier3(train = training_data3,test = testing_data3,act = "Rectifier",hidden_layer = c(8))
ann_tanh_test3<-ann_classifier3(train = training_data3,test = testing_data3,act = "Tanh",hidden_layer = c(10))
ann_max_test3<-ann_classifier3(train = training_data3,test = testing_data3,act = "Maxout",hidden_layer = c(8))

roc_tanh_test1<-geom_line(data =ann_tanh_test3$plotdata, aes(x = FP, y = TP, color = 'Kmeans_allfeatures'))

#Learning Curves
l=1
m=1
tanh_Lc_ann<-list()
tanh_error_test_ann<-c()
tanh_error_train_ann<-c()
tanh_predict_test_ann<-list()
ann_time_Train<-c()
ann_time_Test<-c()
range_ann<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l in range_ann){
  tanh_Lc_ann[[m]]<-ann_classifier3(train = training_data3[1:(l*nrow(training_data3)),],test = training_data3[1:(l*nrow(training_data3)),],act = "Tanh",hidden_layer = c(10))
  tanh_predict_test_ann[[m]]<-ann_classifier3(train= training_data3[1:(l*nrow(training_data3)),],test = testing_data3[1:(l*nrow(testing_data3)),],act = "Tanh",hidden_layer = c(10))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train set Size

Kmeans1_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='kmeans_all_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='kmeans_all_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='kmeans_all_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='kmeans_all_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Sample size-ANN with kmeans and all features")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(kmeans_all_Train="red",kmeans_all_Test="blue"))

Kmeans1_plot

#Error vs Time

kmeans1_plot2<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='kmeans_all_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='kmeans_all_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='kmeans_all_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='kmeans_all_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Clock time- ANN with Kmeans and all features")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(kmeans_all_Train="red",kmeans_all_Test="blue"))


kmeans1_plot2


#Ann on em clustering results and entire dataset
#Results- Same as project 3 but here cluster count was 1
# > ann_tanh_test_em3$AUC
# [1] 0.9092745
# > ann_tanh_test_em3$Accuracy
# [1] 0.8450486



data3_em<-data[-17]
data3_em$cluster<-as.factor(em_1$classification)
data3_em$class<-data$class
library(caTools)
set.seed(1)
split3_em = sample.split(data3_em$class, SplitRatio = 0.70)
training_data_em3 = subset(data3_em, split3_em == TRUE)
testing_data_em3 = subset(data3_em, split3_em == FALSE)

# Feature Scaling
training_data_em3[,c(1,6,10,12,13,14,15)] = scale(training_data_em3[,c(1,6,10,12,13,14,15)])
testing_data_em3[,c(1,6,10,12,13,14,15)] = scale(testing_data_em3[,c(1,6,10,12,13,14,15)])


#Neural network: 

library(h2o)
h2o.init(nthreads = -1)
ann_classifier_em3<-function(train,test,act,hidden_layer){
  
  train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[c(1:17)]),y ='class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
  test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test[-18])))
  performance<-h2o.performance(ann,as.h2o(test))
  confusion_tree<-h2o.confusionMatrix(performance)
  AUC<-performance@metrics$AUC
  Accuracy<-1-confusion_tree$Error[3]
  dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
  results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
  return(results_ann)
}

#Activation Function

ann_rectifier_test_em3<-ann_classifier_em3(train = training_data_em3,test = testing_data_em3,act = "Rectifier",hidden_layer = c(8))
ann_tanh_test_em3<-ann_classifier_em3(train = training_data_em3,test = testing_data_em3,act = "Tanh",hidden_layer = c(10))
ann_max_test_em3<-ann_classifier_em3(train = training_data_em3,test = testing_data_em3,act = "Maxout",hidden_layer = c(8))

roc_tanh_test1_em<-geom_line(data =ann_tanh_test_em3$plotdata, aes(x = FP, y = TP, color = 'EM_allfeatures'))


#Learning Curves
l=1
m=1
tanh_Lc_ann<-list()
tanh_error_test_ann<-c()
tanh_error_train_ann<-c()
tanh_predict_test_ann<-list()
ann_time_Train<-c()
ann_time_Test<-c()
range_ann<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l in range_ann){
  tanh_Lc_ann[[m]]<-ann_classifier_em3(train = training_data_em3[1:(l*nrow(training_data_em3)),],test = training_data_em3[1:(l*nrow(training_data_em3)),],act = "Tanh",hidden_layer = c(10))
  tanh_predict_test_ann[[m]]<-ann_classifier_em3(train= training_data_em3[1:(l*nrow(training_data_em3)),],test = testing_data_em3[1:(l*nrow(testing_data_em3)),],act = "Tanh",hidden_layer = c(10))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train set Size

EM1_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='EM_all_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='EM_all_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='EM_all_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='EM_all_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Sample size-ANN with EM and all features")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(EM_all_Train="red",EM_all_Test="blue"))

EM1_plot

#Error vs Time

EM1_plot2<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='EM_all_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='EM_all_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='EM_all_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='EM_all_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Clock time- ANN with EM and all features")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(EM_all_Train="red",EM_all_Test="blue"))


EM1_plot2








#ANN ON kmeans results data with cluster results and class label only
# 
# > ann_tanh_test2$AUC
# [1] 0.7985022
# > ann_tanh_test2$Accuracy
# [1] 0.7064545


data2<-data3[,c(17,18)]
set.seed(1)
split2 = sample.split(data2$class, SplitRatio = 0.70)
training_data2 = subset(data2, split2 == TRUE)
testing_data2 = subset(data2, split2 == FALSE)


#Neural network: 

library(h2o)
h2o.init(nthreads = -1)
ann_classifier2<-function(train,test,act,hidden_layer){
  
  train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[1]),y ='class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
  test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test[-2])))
  performance<-h2o.performance(ann,as.h2o(test))
  confusion_tree<-h2o.confusionMatrix(performance)
  AUC<-performance@metrics$AUC
  Accuracy<-1-confusion_tree$Error[3]
  dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
  results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
  return(results_ann)
}

#Activation Function

ann_rectifier_test2<-ann_classifier2(train = training_data2,test = testing_data2,act = "Rectifier",hidden_layer = c(8))
ann_tanh_test2<-ann_classifier2(train = training_data2,test = testing_data2,act = "Tanh",hidden_layer = c(2))
ann_max_test2<-ann_classifier2(train = training_data2,test = testing_data2,act = "Maxout",hidden_layer = c(8))


roc_tanh_test2<-geom_line(data =ann_tanh_test2$plotdata, aes(x = FP, y = TP, color = 'Kmeans_classlabel'))


#Learning Curves #Error
l=1
m=1
tanh_Lc_ann<-list()
tanh_error_test_ann<-c()
tanh_error_train_ann<-c()
tanh_predict_test_ann<-list()
ann_time_Train<-c()
ann_time_Test<-c()
range_ann<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l in range_ann){
  tanh_Lc_ann[[m]]<-ann_classifier2(train = training_data2[1:(l*nrow(training_data2)),],test = training_data2[1:(l*nrow(training_data2)),],act = "Tanh",hidden_layer = c(2))
  tanh_predict_test_ann[[m]]<-ann_classifier2(train= training_data2[1:(l*nrow(training_data2)),],test = testing_data2[1:(l*nrow(testing_data2)),],act = "Tanh",hidden_layer = c(2))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train set Size

Kmeans2_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='kmeans_label_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='kmeans_label_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='kmeans_label_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='kmeans_label_Test'))+
  ggtitle("Learning Curve [Adult]:Error vs Sample size-ANN with kmeans and class label")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(kmeans_label_Train="red",kmeans_label_Test="blue"))

Kmeans2_plot

#Error vs Time

kmeans2_plot2<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='kmeans_label_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='kmeans_label_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='kmeans_label_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='kmeans_label_Test'))+
  ggtitle("Learning Curve [Adult]:Error vs Clock time- ANN with Kmeans result and class label")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(kmeans_label_Train="red",kmeans_label_Test="blue"))


kmeans2_plot2




#ANN on just the EM clustering result and class labels
#Error

data2_em<-data3_em[,c(17,18)]

library(caTools)
set.seed(1)
split2_em = sample.split(data2_em$class, SplitRatio = 0.70)
training_data_em2 = subset(data2_em, split2_em == TRUE)
testing_data_em2 = subset(data2_em, split2_em == FALSE)


#Neural network: 

library(h2o)
h2o.init(nthreads = -1)
ann_classifier_em2<-function(train,test,act,hidden_layer){
  
  train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[1]),y ='class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
  test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test)))
  performance<-h2o.performance(ann,as.h2o(test))
  confusion_tree<-h2o.confusionMatrix(performance)
  AUC<-performance@metrics$AUC
  Accuracy<-1-confusion_tree$Error[3]
  dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
  results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
  return(results_ann)
}

#Activation Function

ann_rectifier_test_em2<-ann_classifier_em2(train = training_data_em2,test = testing_data_em2,act = "Rectifier",hidden_layer = c(8))
ann_tanh_test_em2<-ann_classifier_em2(train = training_data_em2,test = testing_data_em2,act = "Tanh",hidden_layer = c(2))
ann_max_test_em2<-ann_classifier_em2(train = training_data_em2,test = testing_data_em2,act = "Maxout",hidden_layer = c(8))



roc_tanh_test_em2<-geom_line(data =ann_tanh_test_em2$plotdata, aes(x = FP, y = TP, color = 'EM_classlabel'))

#Learning Curves #Error
l=1
m=1
tanh_Lc_ann<-list()
tanh_error_test_ann<-c()
tanh_error_train_ann<-c()
tanh_predict_test_ann<-list()
ann_time_Train<-c()
ann_time_Test<-c()
range_ann<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l in range_ann){
  tanh_Lc_ann[[m]]<-ann_classifier2(train = training_data2[1:(l*nrow(training_data2)),],test = training_data2[1:(l*nrow(training_data2)),],act = "Tanh",hidden_layer = c(2))
  tanh_predict_test_ann[[m]]<-ann_classifier2(train= training_data2[1:(l*nrow(training_data2)),],test = testing_data2[1:(l*nrow(testing_data2)),],act = "Tanh",hidden_layer = c(2))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train set Size

EM2_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='kmeans_label_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='kmeans_label_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='kmeans_label_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='kmeans_label_Test'))+
  ggtitle("Learning Curve [Adult]:Error vs Sample size-ANN with kmeans and class label")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(kmeans_label_Train="red",kmeans_label_Test="blue"))

EM2_plot

#Error vs Time

EM2_plot2<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='kmeans_label_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='kmeans_label_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='kmeans_label_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='kmeans_label_Test'))+
  ggtitle("Learning Curve [Adult]:Error vs Clock time- ANN with Kmeans result and class label")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(kmeans_label_Train="red",kmeans_label_Test="blue"))


EM2_plot2





#plotting ROC for kmeans results

ggplot()+roc_tanh_test1+roc_tanh_test2+ xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC [Bank]-ANN for Kmeans Clustering Results")+
  scale_colour_manual(name="Legend",
                      values=c(Kmeans_allfeatures="red",Kmeans_classlabel="blue"))

#plotting ROC for em results

ggplot()+roc_tanh_test1_em+roc_tanh_test_em2+ xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC [Bank]-ANN for EM Clustering Results")+
  scale_colour_manual(name="Legend",
                      values=c(EM_allfeatures="red",EM_classlabel="blue"))





###################################################################################################

#Feature Selection using Decision Tree

tree<-rpart(class~.,data = data,parms = list(split='information'),
            method = 'class')
fancyRpartPlot(tree,tweak=0.6,cex=0.9)
imp<-varImp(tree,scale=FALSE)
imp
#data_imp<-data[c("age",'capital-gain',"education","marital-status","occupation","relationship","capital-loss","hours-per-week","native_country","sex")]

#Feature Selection using backward elimination
logistic<-glm(class~.,data = data,family = binomial)
b<-step(logistic,direction = "backward")
b
data_imp<-data[c("job","marital","education","balance","housing", 
                   "loan","contact","day","month","duration","campaign","previous","poutcome")]

library(caret)
#data_cat<-data[c(3,4,5,6,9,10)]

data_cat<-data_imp[c(1,2,3,5,6,7,9,13)]
data_dum<-dummyVars(~.,data = data_cat,fullRank = T)
dataset<-as.data.frame(predict(data_dum,data_cat))
data_num<-as.data.frame(data_imp[,c(4,8,10,11,12)])
#data_num<-as.data.frame(data_imp[,c(1,2,7,8)])
data_final<-cbind(data_num,dataset)
summary(data_final)

# Feature Scaling
data_scale<-as.data.frame(scale(data_final))

#Assessing the clusterability
#res <- get_clust_tendency(data_scale, 10, graph = TRUE)



#K means Clustering

#using numeric data
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

#Elbow method
set.seed(1)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(data_scale, i,nstart = 25)$withinss)

ggplot()+geom_point(aes(x=1:10,y=wcss),color="red")+geom_line(aes(x=1:10,y=wcss),color="spring green")+
  ggtitle("Elbow Method")+xlab("Number of Clusters")+ylab("WCSS")





kmeans_opt <- kmeans(x=data_scale, centers = 3, nstart = 25)
fviz_cluster(kmeans_opt, data = data_scale,
             ellipse = FALSE,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE
             
             
)



#Expectation-maximization

library(mclust)

BIC <- mclustBIC(data_scale,initialization = list(subset=sample(1:nrow(data_scale), size=100)),modelNames=mclust.options("emModelNames"),control=emControl())

plot(BIC)

em<-Mclust(data_scale,x = BIC,modelNames=mclust.options("emModelNames"))



fviz_cluster(em, data = data_scale,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE,
             ellipse = FALSE)

summary(em)


#Fit EM model
summary.em <- summary(em, data = data_scale)
# EM summary
table(summary.em$classification)

# k-means summary
table(kmeans_opt$cluster)

# k-means versus EM
table(kmeans_opt$cluster, summary.em$classification)

#Validating Cluster Solutions
# comparing 2 cluster solutions
library(fpc)
distcritmulti(data_scale, kmeans_opt$cluster, em$cluster)

#Overlap with Labels
# kmeans
table(data$class, kmeans_opt$cluster)

# EM
table(data$class, em$classification)








#ANN after feature selection

#Results
# > ann_tanh_test1$AUC
# [1] 0.9098188
# > ann_tanh_test1$Accuracy
# [1] 0.8446065

library(caTools)
data1<-data_imp
data1$class<-data$class
set.seed(1)
split1 = sample.split(data1$class, SplitRatio = 0.70)
training_data = subset(data1, split1 == TRUE)
testing_data = subset(data1, split1 == FALSE)
training_data[,c(4,8,10,11,12)] = scale(training_data[,c(4,8,10,11,12)])
testing_data[,c(4,8,10,11,12)] = scale(testing_data[,c(4,8,10,11,12)])


#Neural network: 

library(h2o)
h2o.init(nthreads = -1)
ann_classifier1<-function(train,test,act,hidden_layer){
  
  train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[c(1:13)]),y ='class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
  test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test[,-14])))
  performance<-h2o.performance(ann,as.h2o(test))
  confusion_tree<-h2o.confusionMatrix(performance)
  AUC<-performance@metrics$AUC
  Accuracy<-1-confusion_tree$Error[3]
  dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
  results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
  return(results_ann)
}

#Activation Function

ann_rectifier_test1<-ann_classifier1(train = training_data,test = testing_data,act = "Rectifier",hidden_layer = c(8))
ann_tanh_test1<-ann_classifier1(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(8))
ann_max_test1<-ann_classifier1(train = training_data,test = testing_data,act = "Maxout",hidden_layer = c(8))

roc_tanh_test1_fs<-geom_line(data =ann_tanh_test1$plotdata, aes(x = FP, y = TP, color = 'featureSelection'))

#Learning Curves
l=1
m=1
tanh_Lc_ann<-list()
tanh_error_test_ann<-c()
tanh_error_train_ann<-c()
tanh_predict_test_ann<-list()
ann_time_Train<-c()
ann_time_Test<-c()
range_ann<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l in range_ann){
  tanh_Lc_ann[[m]]<-ann_classifier1(train = training_data[1:(l*nrow(training_data)),],test = training_data[1:(l*nrow(training_data)),],act = "Tanh",hidden_layer = c(8))
  tanh_predict_test_ann[[m]]<-ann_classifier1(train= training_data[1:(l*nrow(training_data)),],test = testing_data[1:(l*nrow(testing_data)),],act = "Tanh",hidden_layer = c(8))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train set Size

FS_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='FS_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='FS_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='FS_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='FS_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Sample size-ANN with Feature Selection")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(FS_Train="red",FS_Test="blue"))
FS_plot

#Error vs Time

FS_plot2<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='FS_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='FS_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='FS_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='FS_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Clock time- ANN with Feature Selection")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(FS_Train="red",FS_Test="blue"))


FS_plot2

#PCA

data_cat_pca<-data[c(2,3,4,5,7,8,9,11,16)]
data_dum_pca<-dummyVars(~.,data = data_cat_pca,fullRank = T)
dataset_pca<-as.data.frame(predict(data_dum_pca,data_cat_pca))
data_num_pca<-as.data.frame(data[,c(1,6,10,12,13,14,15)])

data_final_pca<-cbind(data_num_pca,dataset_pca)
data_scale_dr<-as.data.frame(scale(data_final_pca))
summary(data_final_pca)


res.pca<-prcomp(data_final_pca,scale. = TRUE)
res.pca

# Extract eigenvalues/variances
get_eig(res.pca)

# Default plot
fviz_eig(res.pca, addlabels = TRUE)

# Scree plot - Eigenvalues
fviz_eig(res.pca, choice = "eigenvalue", addlabels=TRUE)
# Use only bar or line plot: geom = "bar" or geom = "line"
fviz_eig(res.pca, geom="line",ncp = 100)

#42 variables transformed into 96 dimensions wHich explains 100% variance
#25 dimensions explains 80% variance.

fviz(res.pca, "ind",labelsize = 0) # Individuals plot
fviz(res.pca, "var") # Variables plot


#
var <- get_pca_var(res.pca)
var

# Coordinates
head(var$coord)
# Cos2: quality on the factore map
head(var$cos2)
# Contributions to the principal components
head(var$contrib)

fviz_pca_var(res.pca, col.var = "black")


# Total cos2 of variables on Dim.1 and Dim.2
fviz_cos2(res.pca, choice = "var", axes = 1:2)


# Color by cos2 values: quality on the factor map
fviz_pca_var(res.pca, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE # Avoid text overlapping
)

#contribution
fviz_pca_var(res.pca, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)

#Plots: quality and contribution

fviz_pca_ind(res.pca, col.ind = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             label
             # Avoid text overlapping (slow if many points)
)

#Clustering using PCA
pca<-as.data.frame(res.pca$x[,1:25])

#Elbow method
set.seed(1)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(pca, i,nstart = 25)$withinss)

ggplot()+geom_point(aes(x=1:10,y=wcss),color="red")+geom_line(aes(x=1:10,y=wcss),color="spring green")+
  ggtitle("Elbow Method")+xlab("Number of Clusters")+ylab("WCSS")





kmeans_opt_pca <- kmeans(x=pca, centers = 3, nstart = 25)
fviz_cluster(kmeans_opt_pca, data = pca,
             ellipse = FALSE,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE
             
             
)



#Expectation-maximization

library(mclust)


BIC_pca <- mclustBIC(pca,initialization = list(subset=sample(1:nrow(pca), size=100)),modelNames=mclust.options("emModelNames"),control=emControl())

plot(BIC_pca)

em_pca<-Mclust(pca,x = BIC,modelNames=mclust.options("emModelNames"))

fviz_cluster(em_pca, data = pca,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE,
             ellipse = FALSE)

summary(em_pca)
#Overlap with Labels
# kmeans
table(data$class, kmeans_opt_pca$cluster)

# EM
table(data$class, em_pca$classification)

#Neural net after Pca
# > ann_tanh_test_pca$AUC
# [1] 0.8965011
# > ann_tanh_test_pca$Accuracy
# [1] 0.8250442

data_pca<-pca
data_pca$class<-data$class
set.seed(1)
split_pca = sample.split(data_pca$class, SplitRatio = 0.70)
training_data_pca = subset(data_pca, split_pca == TRUE)
testing_data_pca = subset(data_pca, split_pca == FALSE)
library(h2o)
h2o.init(nthreads = -1)
ann_classifier_pca<-function(train,test,act,hidden_layer){
  
  train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[c(1:25)]),y ='class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
  test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test[-26])))
  performance<-h2o.performance(ann,as.h2o(test))
  confusion_tree<-h2o.confusionMatrix(performance)
  AUC<-performance@metrics$AUC
  Accuracy<-1-confusion_tree$Error[3]
  dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
  results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
  return(results_ann)
}

#Activation Function

ann_rectifier_test_pca<-ann_classifier_pca(train = training_data_pca,test = testing_data_pca,act = "Rectifier",hidden_layer = c(8))
ann_tanh_test_pca<-ann_classifier_pca(train = training_data_pca,test = testing_data_pca,act = "Tanh",hidden_layer = c(13))
ann_max_test_pca<-ann_classifier_pca(train = training_data_pca,test = testing_data_pca,act = "Maxout",hidden_layer = c(8))

roc_tanh_test1_pca<-geom_line(data =ann_tanh_test_pca$plotdata, aes(x = FP, y = TP, color = 'PCA'))

#Learning Curves
l=1
m=1
tanh_Lc_ann<-list()
tanh_error_test_ann<-c()
tanh_error_train_ann<-c()
tanh_predict_test_ann<-list()
ann_time_Train<-c()
ann_time_Test<-c()
range_ann<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l in range_ann){
  tanh_Lc_ann[[m]]<-ann_classifier_pca(train = training_data_pca[1:(l*nrow(training_data_pca)),],test = training_data_pca[1:(l*nrow(training_data_pca)),],act = "Tanh",hidden_layer = c(13))
  tanh_predict_test_ann[[m]]<-ann_classifier_pca(train= training_data_pca[1:(l*nrow(training_data_pca)),],test = testing_data_pca[1:(l*nrow(testing_data_pca)),],act = "Tanh",hidden_layer = c(13))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train set Size

PCA_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='PCA_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='PCA_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='PCA_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='PCA_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Sample size-ANN with PCA")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(PCA_Train="red",PCA_Test="blue"))
PCA_plot

#Error vs Time

PCA_plot2<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='PCA_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='PCA_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='PCA_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='PCA_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Clock time- ANN with PCA")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(PCA_Train="red",PCA_Test="blue"))


PCA_plot2
#ICA
set.seed(1)
# verify by plotting mean of columns
barplot(sapply(data_scale_dr, mean), horiz=T, las=1, cex.names=0.8, main = "Mean")
# verify by plotting variance of columns
barplot(sapply(data_scale_dr, var), horiz=T, las=1, cex.names=0.8, main = "Variance")
# verify by plotting skewness of columns
barplot(sapply(data_scale_dr, skewness), horiz=T, las=1, cex.names=0.8, main = "Skewness")
# verify by plotting kurtosis of columns
barplot(sapply(data_scale_dr, kurtosis), horiz=T, las=1, cex.names=0.5, main = "Kurtosis")



ica.5 <- fastICA(data_scale_dr,n.comp=5)
ica.10 <- fastICA(data_scale_dr,n.comp=10)
ica.15 <- fastICA(data_scale_dr,n.comp=15)
ica.20 <- fastICA(data_scale_dr,n.comp=20)
ica.50<-fastICA(data_scale_dr,n.comp = 50)



barplot(sapply(data.frame(ica.5$S), kurtosis), horiz=T, las=1, cex.names=0.5, main = "k = 5")
barplot(sapply(data.frame(ica.10$S), kurtosis), horiz=T, las=1, cex.names=0.5, main = "k = 10")
barplot(sapply(data.frame(ica.15$S), kurtosis), horiz=T, las=1, cex.names=0.5, main = "k = 15")
barplot(sapply(data.frame(ica.20$S), kurtosis), horiz=T, las=1, cex.names=0.5, main = "k = 20")
barplot(sapply(data.frame(ica.50$S), kurtosis), horiz=T, las=1, cex.names=0.5, main = "k = 50")
barplot(sapply(data.frame(ica.60$S), kurtosis), horiz=T, las=1, cex.names=0.5, main = "k = 60")
barplot(sapply(data.frame(ica.65$S), kurtosis), horiz=T, las=1, cex.names=0.5, main = "k = 65")
barplot(sapply(data.frame(ica.96$S), kurtosis), horiz=T, las=1, cex.names=0.5, main = "k = 96")











ica<-fastICA(data_scale_dr,42)
plot(ica$S, main = "ICA components")
s<-data.frame(ica$S)
ggplot()+geom_point(aes(x=s[1:45211,1],y=s[1:45211,2]))+xlab("First Component")+ylab("Second Component")
library(plyr)
data_kica<-s[,numcolwise(kurtosis)(s) > 10]



#Neural net after ica
data_ica<-data_kica
data_ica$class<-data$class
set.seed(1)
split_ica = sample.split(data_ica$class, SplitRatio = 0.70)
training_data_ica = subset(data_ica, split_ica == TRUE)
testing_data_ica = subset(data_ica, split_ica == FALSE)
library(h2o)
h2o.init(nthreads = -1)
ann_classifier_ica<-function(train,test,act,hidden_layer){
  
  train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[c(1:25)]),y ='class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
  test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test[-26])))
  performance<-h2o.performance(ann,as.h2o(test))
  confusion_tree<-h2o.confusionMatrix(performance)
  AUC<-performance@metrics$AUC
  Accuracy<-1-confusion_tree$Error[3]
  dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
  results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
  return(results_ann)
}

#Activation Function

ann_rectifier_test_ica<-ann_classifier_ica(train = training_data_ica,test = testing_data_ica,act = "Rectifier",hidden_layer = c(8))
ann_tanh_test_ica<-ann_classifier_ica(train = training_data_ica,test = testing_data_ica,act = "Tanh",hidden_layer = c(14))
ann_max_test_ica<-ann_classifier_ica(train = training_data_ica,test = testing_data_ica,act = "Maxout",hidden_layer = c(8))


roc_tanh_test_ICA<-geom_line(data =ann_tanh_test_ica$plotdata, aes(x = FP, y = TP, color = 'ICA'))

#Learning Curves
l=1
m=1
tanh_Lc_ann<-list()
tanh_error_test_ann<-c()
tanh_error_train_ann<-c()
tanh_predict_test_ann<-list()
ann_time_Train<-c()
ann_time_Test<-c()
range_ann<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l in range_ann){
  tanh_Lc_ann[[m]]<-ann_classifier_ica(train = training_data_ica[1:(l*nrow(training_data_ica)),],test = training_data_ica[1:(l*nrow(training_data_ica)),],act = "Tanh",hidden_layer = c(14))
  tanh_predict_test_ann[[m]]<-ann_classifier_ica(train= training_data_ica[1:(l*nrow(training_data_ica)),],test = testing_data_ica[1:(l*nrow(testing_data_ica)),],act = "Tanh",hidden_layer = c(14))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train set Size

ICA_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='ICA_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='ICA_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='ICA_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='ICA_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Sample size-ANN with ICA")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(ICA_Train="red",ICA_Test="blue"))
ICA_plot

#Error vs Time

ICA_plot2<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='ICA_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='ICA_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='ICA_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='ICA_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Clock time- ANN with ICA")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(ICA_Train="red",ICA_Test="blue"))


ICA_plot2


#KMEANS on ICA

#Elbow method
set.seed(1)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(data_ica[-26],i,nstart = 25)$withinss)

ggplot()+geom_point(aes(x=1:10,y=wcss),color="red")+geom_line(aes(x=1:10,y=wcss),color="spring green")+
  ggtitle("Elbow Method")+xlab("Number of Clusters")+ylab("WCSS")





kmeans_opt_ica <- kmeans(x=data_ica[-26], centers = 9, nstart = 25)
fviz_cluster(kmeans_opt_ica, data = data_ica[-26],
             ellipse = FALSE,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE
             
             
)



#Expectation-maximization

library(mclust)


BIC_ica <- mclustBIC(data_ica[-26],initialization = list(subset=sample(1:nrow(data_ica[-26]), size=100)),modelNames=mclust.options("emModelNames"),control=emControl())

plot(BIC_ica)

em_ica<-Mclust(data_ica[-26],x = BIC_ica,modelNames=mclust.options("emModelNames"))

fviz_cluster(em_ica, data = data_ica[-26],
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE,
             ellipse = FALSE)

summary(em_ica)


#Overlap with Labels
# kmeans
table(data$class, kmeans_opt_ica$cluster)

# EM
table(data$class, em_ica$classification)


#Random projections
library(RandPro)

# SRP Random Projection

dim_10<-find_dim_JL(10,epsilon = c(0.01,0.1,0.5,0.8,1))
dim_100<-find_dim_JL(100,epsilon = c(0.01,0.1,0.5,0.8,1))
dim_1000<-find_dim_JL(1000,epsilon = c(0.01,0.1,0.5,0.8,1))
dim_42<-find_dim_JL(42,epsilon = c(0.01,0.1,0.5,0.8,0.9,1))
dim_42



srp<-form_sparse_matrix(n_rows=42,n_cols = 4,JLT = TRUE,method = "achlioptas",eps = 1)
rp <- data.frame(as.matrix(data_scale_dr)%*%srp)


#KMEANS on Random projections

#Elbow method
set.seed(1)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(rp,i,nstart = 25)$withinss)

ggplot()+geom_point(aes(x=1:10,y=wcss),color="red")+geom_line(aes(x=1:10,y=wcss),color="spring green")+
  ggtitle("Elbow Method")+xlab("Number of Clusters")+ylab("WCSS")





kmeans_opt_rp <- kmeans(x=rp, centers = 9, nstart = 25)
fviz_cluster(kmeans_opt_rp, data = rp,
             ellipse = FALSE,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE
             
             
)



#Expectation-maximization

library(mclust)


BIC_rp <- mclustBIC(rp,initialization = list(subset=sample(1:nrow(rp), size=100)),modelNames=mclust.options("emModelNames"),control=emControl())

plot(BIC_rp)

em_rp<-Mclust(rp,x = BIC_rp,modelNames=mclust.options("emModelNames"))

fviz_cluster(em_rp, data = rp,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             labelsize = 0,
             show.clust.cent=TRUE,
             ellipse = FALSE)

summary(em_rp)
#Overlap with Labels
# kmeans
table(data$class, kmeans_opt_rp$cluster)

# EM
table(data$class, em_rp$classification)


#Neural net after RP
data_rp<-rp
data_rp$class<-data$class
set.seed(1)
split_rp = sample.split(data_rp$class, SplitRatio = 0.70)
training_data_rp = subset(data_rp, split_rp == TRUE)
testing_data_rp = subset(data_rp, split_rp == FALSE)
library(h2o)
h2o.init(nthreads = -1)
ann_classifier_rp<-function(train,test,act,hidden_layer){
  
  train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[c(1:33)]),y ='class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
  test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test[-34])))
  performance<-h2o.performance(ann,as.h2o(test))
  confusion_tree<-h2o.confusionMatrix(performance)
  AUC<-performance@metrics$AUC
  Accuracy<-1-confusion_tree$Error[3]
  dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
  results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
  return(results_ann)
}

#Activation Function

ann_rectifier_test_rp<-ann_classifier_rp(train = training_data_rp,test = testing_data_rp,act = "Rectifier",hidden_layer = c(8))
ann_tanh_test_rp<-ann_classifier_rp(train = training_data_rp,test = testing_data_rp,act = "Tanh",hidden_layer = c(18))
ann_max_test_rp<-ann_classifier_rp(train = training_data_rp,test = testing_data_rp,act = "Maxout",hidden_layer = c(8))

roc_tanh_test_rp<-geom_line(data =ann_tanh_test_rp$plotdata, aes(x = FP, y = TP, color = 'RCA'))


#Learning Curves
l=1
m=1
tanh_Lc_ann<-list()
tanh_error_test_ann<-c()
tanh_error_train_ann<-c()
tanh_predict_test_ann<-list()
ann_time_Train<-c()
ann_time_Test<-c()
range_ann<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l in range_ann){
  tanh_Lc_ann[[m]]<-ann_classifier_rp(train = training_data_rp[1:(l*nrow(training_data_rp)),],test = training_data_rp[1:(l*nrow(training_data_rp)),],act = "Tanh",hidden_layer = c(18))
  tanh_predict_test_ann[[m]]<-ann_classifier_rp(train= training_data_rp[1:(l*nrow(training_data_rp)),],test = testing_data_rp[1:(l*nrow(testing_data_rp)),],act = "Tanh",hidden_layer = c(18))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train set Size

RP_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='RP_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='RP_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='RP_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='RP_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Sample size-ANN with Random Projection")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(RP_Train="red",RP_Test="blue"))
RP_plot

#Error vs Time

RP_plot2<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='RP_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='RP_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='RP_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='RP_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Clock time- ANN with Random Projection")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(RP_Train="red",RP_Test="blue"))


RP_plot2

#plotting ROC for PCA,ICA,RP  and feature selection
ggplot()+roc_tanh_test1_fs+roc_tanh_test1_pca+roc_tanh_test_ICA+roc_tanh_test_rp+
  xlab("False Positive")+ylab("True Positive")+ggtitle("ROC [Adult]-ANN with different Dimensionality Reduction")+
  scale_colour_manual(name="Legend",
                      values=c(featureSelection="red",PCA="blue",ICA="cyan",RCA="black"))



