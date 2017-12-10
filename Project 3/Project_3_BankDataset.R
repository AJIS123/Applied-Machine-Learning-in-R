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
library(xgboost)
library(kknn)
library(h2o)

#Data import

data<-read.csv('bank-full.csv',sep = ";")

#Data preprocessing

nrow(data) - sum(complete.cases(data))
dim(data)
data<-na.omit(data)
sum(is.na(data))
data$Class<-factor(data$y,levels=c("no","yes"),labels = c("0","1"))
data$y = NULL

#Train and Test set preparation for KNN

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

train_test$Class<-factor(train_test$Class)
library(caTools)
set.seed(1)
split1 = sample.split(train_test$Class, SplitRatio = 0.70)
training_set = subset(train_test, split1 == TRUE)
testing_set = subset(train_test, split1 == FALSE)
training_set[,-17]<-scale(training_set[,-17])
testing_set[,-17]<-scale(testing_set[,-17])

# Train and Test Dataset for ANN

library(caTools)
set.seed(1)
split1 = sample.split(data$Class, SplitRatio = 0.70)
training_data = subset(data, split1 == TRUE)
testing_data = subset(data, split1 == FALSE)

# Feature Scaling
training_data[,c(1,6,10,12,13,14,15)] = scale(training_data[,c(1,6,10,12,13,14,15)])
testing_data[,c(1,6,10,12,13,14,15)] = scale(testing_data[,c(1,6,10,12,13,14,15)])

#Neural network: 

library(h2o)
h2o.init(nthreads = -1)
ann_classifier<-function(train,test,act,hidden_layer){
  
train.time<-system.time(ann<- h2o.deeplearning(x=colnames(train[c(1:16)]),y ='Class',
                                                 training_frame = as.h2o(train),
                                                 activation = act,
                                                 hidden = hidden_layer,
                                                 epochs = 100,
                                                 train_samples_per_iteration = -2,
                                                 balance_classes = T,
                                                 seed = 1))
test.time<-system.time(ann_pred <- h2o.predict(ann, newdata = as.h2o(test[-15])))
performance<-h2o.performance(ann,as.h2o(test))
confusion_tree<-h2o.confusionMatrix(performance)
AUC<-performance@metrics$AUC
Accuracy<-1-confusion_tree$Error[3]
dd<-data.frame(FP = performance@metrics$thresholds_and_metric_scores$fpr, TP = performance@metrics$thresholds_and_metric_scores$tpr)
results_ann<-list(Confustion_Matrix=confusion_tree,AUC=AUC,plotdata=dd,Accuracy=Accuracy,perf=performance,train.time=train.time,test.time=test.time)
return(results_ann)
}

#Activation Function

ann_rectifier_test<-ann_classifier(train = training_data,test = testing_data,act = "Rectifier",hidden_layer = c(9))
ann_tanh_test<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(9))
ann_max_test<-ann_classifier(train = training_data,test = testing_data,act = "Maxout",hidden_layer = c(9))

ann_rectifier_train<-ann_classifier(train = training_data,test = training_data,act = "Rectifier",hidden_layer = c(9))
ann_tanh_train<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(9))
ann_max_train<-ann_classifier(train = training_data,test = training_data,act = "Maxout",hidden_layer = c(9))

roc_rectifier_test<-geom_line(data =ann_rectifier_test$plotdata, aes(x = FP, y = TP, color = 'Ann_Rectifier'))
roc_tanh_test<-geom_line(data =ann_tanh_test$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh'))
roc_max_test<-geom_line(data =ann_max_test$plotdata, aes(x = FP, y = TP, color = 'Ann_Maxout'))

ann_rectifier_test$Confustion_Matrix

#plotting ROC for activation function

ggplot()+roc_rectifier_test+roc_tanh_test+roc_max_test+
  xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC [Bank]-ANN for different activation functions")+
  scale_colour_manual(name="Legend",
                      values=c(Ann_Rectifier="red",Ann_Tanh="blue",Ann_Maxout="cyan"))

#Changing number of layers
ann_tanh_test_one<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(9))
ann_tanh_train_one<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(9))
ann_tanh_test_two<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(9,9))
ann_tanh_train_two<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(9,9))
ann_tanh_test_three<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(9,9,9))
ann_tanh_train_three<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(9,9,9))


roc_tanh_test_one<-geom_line(data =ann_tanh_test_one$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_OneLayer'))
roc_tanh_test_two<-geom_line(data =ann_tanh_test_two$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_TwoLayer'))
roc_tanh_test_three<-geom_line(data =ann_tanh_test_three$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_ThreeLayer'))


#plotting ROC for number of layers

ggplot()+roc_tanh_test_one+roc_tanh_test_two+roc_tanh_test_three+
  xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC [Bank]-ANN for different number of layers")+
  scale_colour_manual(name="Legend",
                      values=c(Ann_Tanh_OneLayer="red",Ann_Tanh_TwoLayer="blue",Ann_Tanh_ThreeLayer="cyan"))


#Changing the number of nodes and layers

ann_tanh_test_nine_nodes_one<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(9))
ann_tanh_train_nine_nodes_one<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(9))
ann_tanh_test_18_nodes_one<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(18))
ann_tanh_train_18_nodes_one<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(18))
ann_tanh_test_nine_nodes_two<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(9,9))
ann_tanh_train_nine_nodes_two<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(9,9))
ann_tanh_test_18_nodes_two<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(18,18))
ann_tanh_train_18_nodes_two<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(18,18))
ann_tanh_test_100_nodes_one<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(100))
ann_tanh_train_100_nodes_one<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(100))
ann_tanh_test_100_nodes_two<-ann_classifier(train = training_data,test = testing_data,act = "Tanh",hidden_layer = c(100,100))
ann_tanh_train_100_nodes_two<-ann_classifier(train = training_data,test = training_data,act = "Tanh",hidden_layer = c(100,100))



#Plotting 

roc_tanh_test_nine_nodes_one<-geom_line(data =ann_tanh_test_nine_nodes_one$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_OneLayer_9nodes'))
roc_tanh_test_18_nodes_one<-geom_line(data =ann_tanh_test_18_nodes_one$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_OneLayer_18nodes'))
roc_tanh_test_100_nodes_one<-geom_line(data =ann_tanh_test_100_nodes_one$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_OneLayer_100nodes'))
roc_tanh_test_nine_nodes_two<-geom_line(data =ann_tanh_test_nine_nodes_two$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_TwoLayer_9nodes'))
roc_tanh_test_18_nodes_two<-geom_line(data =ann_tanh_test_18_nodes_two$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_TwoLayer_18nodes'))
roc_tanh_test_100_nodes_two<-geom_line(data =ann_tanh_test_100_nodes_two$plotdata, aes(x = FP, y = TP, color = 'Ann_Tanh_TwoLayer_100nodes'))

ggplot()+roc_tanh_test_nine_nodes_one+roc_tanh_test_nine_nodes_two+roc_tanh_test_18_nodes_one+roc_tanh_test_18_nodes_two+roc_tanh_test_100_nodes_one+roc_tanh_test_100_nodes_two+
  xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC [Bank]- ANN for different number of layers and nodes")+
  scale_colour_manual(name="Legend",
                      values=c(Ann_Tanh_OneLayer_9nodes="red",Ann_Tanh_OneLayer_18nodes="blue",Ann_Tanh_OneLayer_100nodes="cyan",Ann_Tanh_TwoLayer_9nodes="black",Ann_Tanh_TwoLayer_18nodes="pink",Ann_Tanh_TwoLayer_100nodes="yellow"))


#Learning Curve  #each time h2o produces different results

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
  tanh_Lc_ann[[m]]<-ann_classifier(train = training_data[1:(l*nrow(training_data)),],test = training_data[1:(l*nrow(training_data)),],act = "Tanh",hidden_layer = c(9))
  tanh_predict_test_ann[[m]]<-ann_classifier(train= training_data[1:(l*nrow(training_data)),],test = testing_data[1:(l*nrow(testing_data)),],act = "Tanh",hidden_layer = c(9))
  tanh_error_train_ann[m]<-1-as.numeric(tanh_Lc_ann[[m]][4])
  tanh_error_test_ann[m]<-1-as.numeric(tanh_predict_test_ann[[m]][4])
  ann_time_Train[m]<-as.numeric(tanh_Lc_ann[[m]][[6]][3])
  ann_time_Test[m]<-as.numeric(tanh_predict_test_ann[[m]][[6]][3])
  m=m+1
}

#Error Vs Train Set size

tanh_plot<-ggplot()+geom_point(aes(x=range_ann,y=tanh_error_train_ann,color='Tanh_ann_Train'))+
  geom_line(aes(x=range_ann,y=tanh_error_train_ann,color='Tanh_ann_Train'))+geom_line(aes(x=range_ann,y=tanh_error_test_ann,color='Tanh_ann_Test'))+
  geom_point(aes(x=range_ann,y=tanh_error_test_ann,color='Tanh_ann_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Sample size-ANN with Tanh")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Tanh_ann_Train="red",Tanh_ann_Test="blue"))

tanh_plot

#Error vs Time

tanh_plot1<-ggplot()+geom_point(aes(x=ann_time_Train,y=tanh_error_train_ann,color='Tanh_ann_Train'))+
  geom_line(aes(x=ann_time_Train,y=tanh_error_train_ann,color='Tanh_ann_Train'))+geom_line(aes(x=ann_time_Test,y=tanh_error_test_ann,color='Tanh_ann_Test'))+
  geom_point(aes(x=ann_time_Test,y=tanh_error_test_ann,color='Tanh_ann_Test'))+
  ggtitle("Learning Curve[Bank]: Error vs Clock time-ANN with Tanh")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(Tanh_ann_Train="red",Tanh_ann_Test="blue"))

tanh_plot1


#grid search Parameter Tuning - computional intense

activation_opt <- c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout") 
hidden_opt <- list(c(9),c(18),c(36),c(100),c(9,9),c(18,18),c(18,18,18),c(36,36),c(9,18,36),c(100,100))
balance_classes = c(TRUE, FALSE)
hyper_params <- list(activation = activation_opt,
                     hidden = hidden_opt,
                     balance_classes=balance_classes)
grid <- h2o.grid("deeplearning", x=colnames(training_data[c(1:16)]),y ='Class',
                 training_frame = as.h2o(training_data),
                 epochs = 100,
                 nfolds = 5,
                 fold_assignment = "Modulo",
                 keep_cross_validation_predictions = TRUE,                    
                 hyper_params = hyper_params)
summary(grid)
h2o.shutdown()

#KNN

#Cross validation to find the optimum k
opt_k<-train.kknn(Class~.,data = training_set,kmax = 100,kernel = "rectangular",kcv=10)
opt_k$best.parameters
plot(opt_k)

#Knn Function
knn_classifier<-function(train,test,k){
  train_time<-system.time(knn<-kknn(Class~.,train=train,test = test,k=k,kernel = "rectangular"))
  knn_confusionMatrix<-confusionMatrix(knn$fitted.values,test$Class)
  knn_prediction<-prediction(knn$prob[,2],test$Class)
  knn_performance<-performance(knn_prediction,"tpr","fpr")
  dd_knn<-data.frame(FP = knn_performance@x.values[[1]], TP = knn_performance@y.values[[1]])
  auc_knn<-performance(knn_prediction, measure = 'auc')@y.values[[1]]
  
  knn_results<-list(Confustion_Matrix=knn_confusionMatrix,AUC=auc_knn,plotdata=dd_knn,Accuracy=knn_confusionMatrix$overall[1],time=train_time)
  return(knn_results)
}

#Experimenting with various values of k

knn_k1_test<-knn_classifier(train = training_set,test = testing_set,k=1)
knn_k5_test<-knn_classifier(train = training_set,test = testing_set,k=5)
knn_k10_test<-knn_classifier(train = training_set,test = testing_set,k=10)
knn_k11_test<-knn_classifier(train = training_set,test = testing_set,k=11)
knn_k15_test<-knn_classifier(train = training_set,test = testing_set,k=15)
knn_k21_test<-knn_classifier(train = training_set,test = testing_set,k=21)
knn_k25_test<-knn_classifier(train = training_set,test = testing_set,k=25)
knn_k30_test<-knn_classifier(train = training_set,test = testing_set,k=30)
knn_k50_test<-knn_classifier(train = training_set,test = testing_set,k=50)


roc_knn_k1 <-geom_line(data =knn_k1_test$plotdata, aes(x = FP, y = TP, color = 'knn_k1'))
roc_knn_k5 <-geom_line(data =knn_k5_test$plotdata, aes(x = FP, y = TP, color = 'knn_k5'))
roc_knn_k10 <-geom_line(data =knn_k10_test$plotdata, aes(x = FP, y = TP, color = 'knn_k10'))
roc_knn_k11 <-geom_line(data =knn_k11_test$plotdata, aes(x = FP, y = TP, color = 'knn_k11'))
roc_knn_k15 <-geom_line(data =knn_k15_test$plotdata, aes(x = FP, y = TP, color = 'knn_k15'))
roc_knn_k21 <-geom_line(data =knn_k21_test$plotdata, aes(x = FP, y = TP, color = 'knn_k21'))
roc_knn_k25 <-geom_line(data =knn_k25_test$plotdata, aes(x = FP, y = TP, color = 'knn_k25'))
roc_knn_k30 <-geom_line(data =knn_k30_test$plotdata, aes(x = FP, y = TP, color = 'knn_k30'))
roc_knn_k50 <-geom_line(data =knn_k50_test$plotdata, aes(x = FP, y = TP, color = 'knn_k50'))

#ROC curve for knn

ggplot()+roc_knn_k1+roc_knn_k5+roc_knn_k10+roc_knn_k11+roc_knn_k15+roc_knn_k21+roc_knn_k25+roc_knn_k30+roc_knn_k50+
  xlab("False Positive")+ylab("True Positive")+ggtitle("ROC [Bank]- KNN for different number of neighbours")+
  scale_colour_manual(name="Legend",
                      values=c(knn_k1="red",knn_k5="blue",knn_k10="cyan",knn_k11="black",knn_k15="pink",knn_k21="springgreen",knn_k25="brown",knn_k30="palegreen",knn_k50="yellow"))


#Learning Curve 

l1=1
m1=1
Lc_knn<-list()
error_test_knn<-c()
error_train_knn<-c()
predict_test_knn<-list()
knn_time_Train<-c()
knn_time_Test<-c()
range_knn<-c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for(l1 in range_knn){
  Lc_knn[[m1]]<-knn_classifier(train = training_set[1:(l1*nrow(training_set)),],test = training_set[1:(l1*nrow(training_set)),],k = 19)  
  predict_test_knn[[m1]]<-knn_classifier(train= training_set[1:(l1*nrow(training_set)),],test = testing_set[1:(l1*nrow(testing_set)),],k=19)
  error_train_knn[m1]<-1-as.numeric(Lc_knn[[m1]][4])
  error_test_knn[m1]<-1-as.numeric(predict_test_knn[[m1]][4])
  knn_time_Train[m1]<-as.numeric(Lc_knn[[m1]][[5]][3])
  knn_time_Test[m1]<-as.numeric(predict_test_knn[[m1]][[5]][3])
  m1=m1+1
}

#Error vs Train Set Size

knn_plot<-ggplot()+geom_point(aes(x=range_knn,y=error_train_knn,color='knn_Train'))+
  geom_line(aes(x=range_knn,y=error_train_knn,color='knn_Train'))+geom_line(aes(x=range_knn,y=error_test_knn,color='knn_Test'))+
  geom_point(aes(x=range_knn,y=error_test_knn,color='knn_Test'))+
  ggtitle("Learning Curve [Bank]: Error vs Sample Size- KNN with k=21")+
  xlab('Train Set Size')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(knn_Train="red",knn_Test="blue"))

knn_plot

#Error vs Time

knn_plot1<-ggplot()+geom_point(aes(x=knn_time_Train,y=error_train_knn,color='knn_Train'))+
  geom_line(aes(x=knn_time_Train,y=error_train_knn,color='knn_Train'))+geom_line(aes(x=knn_time_Test,y=error_test_knn,color='knn_Test'))+
  geom_point(aes(x=knn_time_Test,y=error_test_knn,color='knn_Test'))+
  ggtitle("Learning Curve [Bank]:Error vs Clock time- KNN with k=21")+
  xlab('Time')+
  ylab('Error')+
  scale_colour_manual(name="Legend",
                      values=c(knn_Train="red",knn_Test="blue"))

knn_plot1

#Comparison of 2 algorithms

ggplot()+roc_knn_k21+roc_tanh_test+
  xlab("False Positive")+ylab("True Positive")+ggtitle("ROC [Bank] for KNN and ANN")+
  scale_colour_manual(name="Legend",
                      values=c(knn_k21="springgreen",Ann_Tanh="red"))
