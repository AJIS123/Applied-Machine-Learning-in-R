##############################################################
##                                                      ##
##        BUAN 6341:: Applied Machine Learning          ##  
##                                                      ##
##                    PROJECT 1                         ##
##                                                      ##
##      Linear Regression on the Bike Sharing Dataset   ##
##           Using Gradient Descent Algorithm           ##
##                                                      ##
##              Name:  Aji Somaraj                      ##
##              NetID: axs161031                        ##
##                                                      ##
##                                                      ##
##############################################################


#Setting the working directory

setwd("E:/AML - BUAN 6341")
getwd()

#Importing the Dataset

data<-read.csv('hour.csv',header = TRUE)
data<-data[c('season','mnth','hr','holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed','cnt')]

#Basic Data Exploration

str(data)
summary(data)

#Libraries

library(pracma)#To Create random matrices
library(ggplot2)#Visualisation
library(corrplot)#Correlation
library(caTools)#Split Data into Test and Train Set

#Correlation

cr<-cor(data)
corrplot(cr,type="lower")
corrplot(cr,method = "number")

#splitting the dataset

set.seed(1)
split <- sample.split(data$cnt, SplitRatio = 0.7)
training_set <-subset(data, split == TRUE)
test_set <- subset(data, split == FALSE)

#Feature Scaling

training_set[,1:11]<-scale(training_set[,1:11])
test_set[,1:11]<-scale(test_set[,1:11])

#Design Matrix X is created

x_Train<-as.matrix(training_set[,-12])
y_Train<-training_set[,12]
x_Test<-as.matrix(test_set[,-12])
y_Test<-test_set[,12]
n<-ncol(x_Train)
x_Train<-cbind(X0=1,x_Train)
x_Test<-cbind(X0=1,x_Test)

#Checking if x_Train and x_Test are matrices

is.matrix(x_Train)
is.matrix(x_Test)

#intialising beta with random numbers

beta<-rand(n+1,1)
is.matrix(beta)

#Intialising vectors and lists

Alpha_Hist<-c()
threshold_Hist<-double()
alpha_Train_results<-list()
alpha_Test_results<-list()
Th_Train_results<-list()
Th_Test_results<-list()
cost_Hist<-double()

#Cost Function

J<- function(x,y,beta){
  sum((x %*% beta-y)^2)/(2*length(y))
}

i=3
j=1
k=1

#Gradient Descent Algorithm with Batch Update Rule

G<-function(x,y,beta,th,alpha){
  m <- length(y)
  cost_Hist[1]<-100000 #Intial Cost
  cost_Hist[2]<-J(x,y,beta)
  Alpha_Hist[j]<-alpha
  
  change<-(cost_Hist[1]-cost_Hist[2])/cost_Hist[1]#Computing intial change
  while(change>th){
    beta <- beta - alpha*(1/m)*(t(x)%*%(x%*%beta - y))#updating beta
    cost_Hist[i]<-J(x,y,beta)
    change<-(cost_Hist[i-1]-cost_Hist[i])/cost_Hist[i-1]#Computing the change
    i=i+1
    j=j+1
    Alpha_Hist[j]<-alpha
  }
  threshold_Hist[j]<-th
  results<-list(Alpha=Alpha_Hist,Beta=beta,num_iter=j,cost=cost_Hist,threshold=threshold_Hist)
  return(results)
}
##------------------------------------------------------------------------------------------------------------------
#Experiment 1:: Experimenting with various values of alpha using a Threshold,th= 0.01
##------------------------------------------------------------------------------------------------------------------

for (alpha in c(0.001,0.01,0.1,0.3,0.4,0.5,0.6,0.648,0.65,0.68,0.7,0.8,0.9,1)){
  alpha_Train_results[[k]]<-G(x_Train,y_Train,beta,th=0.01,alpha)
  alpha_Test_results[[k]]<-G(x_Test,y_Test,beta,th=0.01,alpha)
  k<-k+1
}

#Retriving the minimum  cost for each alpha

alphalist_Train<-unlist(lapply(sapply(alpha_Train_results,'[[',1),tail,n=1L))
costlist_Train<-unlist(lapply(sapply(alpha_Train_results,'[[',4),tail,n=1L))
alphalist_Test<-unlist(lapply(sapply(alpha_Test_results,'[[',1),tail,n=1L))
costlist_Test<-unlist(lapply(sapply(alpha_Test_results,'[[',4),tail,n=1L))

#Visualisation
#Cost Vs Learning Rate

ggplot()+
  geom_point(aes(x=alphalist_Train,y=costlist_Train,color="Training_Set"))+
  geom_point(aes(x=alphalist_Test,y=costlist_Test,color="Test_Set"))+
  geom_line(aes(x=alphalist_Train,y=costlist_Train,color="Training_Set"))+
  geom_line(aes(x=alphalist_Test,y=costlist_Test,color="Test_Set"))+
  ggtitle('Cost vs Learning rate(Threshold = 0.01)')+
  xlab('Learning rate')+
  ylab('Cost')+
  scale_colour_manual(name="Legend",
                      values=c(Training_Set="red", Test_Set="blue"))+
  theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6),
    legend.key = element_rect(colour = "black"))## Legend is added and repositioned
##------------------------------------------------------------------------------------------------------------------
#Experiment 2:: Experimenting with various threshold values using alpha=0.648
##------------------------------------------------------------------------------------------------------------------

t<-1
for(th in c(0.000001,0.00001,0.0001,0.001,0.003,0.01,0.03,0.1)){
  Th_Train_results[[t]]<-G(x_Train,y_Train,beta,th,alpha=0.648)
  Th_Test_results[[t]]<-G(x_Test,y_Test,beta,th,alpha=0.648)
  t<-t+1
}

#Retriving the minimum  cost,number of iterations for each threshold

threshold_Train_th<-unlist(lapply(sapply(Th_Train_results,'[[',5),tail,n=1L))
costlist_Train_th<-unlist(lapply(sapply(Th_Train_results,'[[',4),tail,n=1L))
threshold_Test_th<-unlist(lapply(sapply(Th_Test_results,'[[',5),tail,n=1L))
costlist_Test_th<-unlist(lapply(sapply(Th_Test_results,'[[',4),tail,n=1L))
num_iter_th_train<-unlist(lapply(sapply(Th_Train_results,'[[',3),tail,n=1L))
num_iter_th_test<-unlist(lapply(sapply(Th_Test_results,'[[',3),tail,n=1L))


#Visualisation
#Cost Vs threshold

ggplot()+
  geom_point(aes(x=threshold_Train_th,y=costlist_Train_th,color="Training_Set"))+
  geom_point(aes(x=threshold_Test_th,y=costlist_Test_th,color="Test_Set"))+
  geom_line(aes(x=threshold_Train_th,y=costlist_Train_th,color="Training_Set"))+
  geom_line(aes(x=threshold_Test_th,y=costlist_Test_th,color="Test_Set"))+
  ggtitle('Cost vs Threshold(Learning Rate=0.648)')+
  xlab('Threshold')+
  ylab('Cost')+
  scale_colour_manual(name="Legend",
                      values=c(Training_Set="red", Test_Set="blue"))## Legend is added

##------------------------------------------------------------------------------------------------------------------
#Experiment 3:: Randomly Selecting 3 features
##------------------------------------------------------------------------------------------------------------------
s<-sample(1:11,3)
s
rand_x_Train<-as.matrix(training_set[,s])
colnames(rand_x_Train)
y_Train<-training_set[,12]
rand_x_Test<-as.matrix(test_set[,s])
colnames(rand_x_Test)
y_Test<-test_set[,12]
l<-ncol(rand_x_Train)

# Design Matrix creation

rand_x_Train<-cbind(X0=1,rand_x_Train)
rand_x_Test<-cbind(X0=1,rand_x_Test)
rand_beta<-rand(l+1,1)

#Applying Gradient Descent Algorithm using the randomly selected features with alpha =0.648 and threshold=0.00001 

rand_Train_results<-G(rand_x_Train,y_Train,rand_beta,th=0.00001,alpha=0.648)
rand_Test_results<-G(rand_x_Test,y_Test,rand_beta,th=0.00001,alpha=0.648)
Train_results<-G(x_Train,y_Train,beta,th=0.00001,alpha=0.648)
Test_results<-G(x_Test,y_Test,beta,th=0.00001,alpha=0.648)

#Visualisation 
#Plotting random features and all features

ggplot()+
  geom_point(aes(x=1:rand_Train_results$num_iter,y=rand_Train_results$cost[-1],color="Random_Features_Training_Set"))+
  geom_point(aes(x=1:rand_Test_results$num_iter,y=rand_Test_results$cost[-1],color="Random_Features_Test_Set"))+
  geom_point(aes(x=1:Train_results$num_iter,y=Train_results$cost[-1],color="All_Features_Training_Set"))+
  geom_point(aes(x=1:Test_results$num_iter,y=Test_results$cost[-1],color="All_Features_Test_Set"))+
  geom_line(aes(x=1:rand_Train_results$num_iter,y=rand_Train_results$cost[-1],color="Random_Features_Training_Set"))+
  geom_line(aes(x=1:rand_Test_results$num_iter,y=rand_Test_results$cost[-1],color="Random_Features_Test_Set"))+
  geom_line(aes(x=1:Train_results$num_iter,y=Train_results$cost[-1],color="All_Features_Training_Set"))+
  geom_line(aes(x=1:Test_results$num_iter,y=Test_results$cost[-1],color="All_Features_Test_Set"))+
  ggtitle('Comparing Random Features and All Features(alpha = 0.648 & Threshold = 0.00001)')+
  xlab('Number of iterations')+
  ylab('Cost')+
  scale_colour_manual(name="Legend",
                      values=c(Random_Features_Training_Set="Red",Random_Features_Test_Set="Blue",
                               All_Features_Training_Set="Orange",All_Features_Test_Set="Green"))+
  theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6),
    legend.key = element_rect(colour = "black"))## Legend is added and repositioned
##------------------------------------------------------------------------------------------------------------------
#Experiment 4:: The 3 best suited features
##------------------------------------------------------------------------------------------------------------------
best_x_Train<-as.matrix(training_set[,c('hr','atemp','hum')])
colnames(best_x_Train)
y_Train<-training_set[,12]
best_x_Test<-as.matrix(test_set[,c('hr','atemp','hum')])
colnames(best_x_Test)
y_Test<-test_set[,12]
a<-ncol(best_x_Train)

# Design Matrix creation and intial beta creation

best_x_Train<-cbind(X0=1,best_x_Train)
best_x_Test<-cbind(X0=1,best_x_Test)
best_beta<-rand(l+1,1)

#Applying Gradient Descent Algorithm using the selected 3 best features with alpha =0.648 and threshold=0.00001 

best_Train_results<-G(best_x_Train,y_Train,best_beta,th=0.00001,alpha=0.648)
best_Test_results<-G(best_x_Test,y_Test,best_beta,th=0.00001,alpha=0.648)
Train_results<-G(x_Train,y_Train,beta,th=0.00001,alpha=0.648)
Test_results<-G(x_Test,y_Test,beta,th=0.00001,alpha=0.648)

#Plotting best(p), random features(q) and all features(r)

p<-ggplot()+
  geom_point(aes(x=1:best_Train_results$num_iter,y=best_Train_results$cost[-1],color="Best_Features_TrainingSet"))+
  geom_point(aes(x=1:best_Test_results$num_iter,y=best_Test_results$cost[-1],color="Best_Features_TestSet"))+
  geom_line(aes(x=1:best_Train_results$num_iter,y=best_Train_results$cost[-1],color="Best_Features_TrainingSet"))+
  geom_line(aes(x=1:best_Test_results$num_iter,y=best_Test_results$cost[-1],color="Best_Features_TestSet"))+
  ggtitle('Selected Best Features(alpha = 0.648 & Threshold = 0.00001)')+
  xlab('Number of iterations')+
  ylab('Cost')+
  scale_colour_manual(name="Legend",
                      values=c(Best_Features_TrainingSet="Red",Best_Features_TestSet="Blue"))+
  theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6),
    legend.key = element_rect(colour = "black"))## Legend is added and repositioned


q<-ggplot()+
  geom_point(aes(x=1:rand_Train_results$num_iter,y=rand_Train_results$cost[-1],color="Random_Features_Training_Set"))+
  geom_point(aes(x=1:rand_Test_results$num_iter,y=rand_Test_results$cost[-1],color="Random_Features_Test_Set"))+
  geom_line(aes(x=1:rand_Train_results$num_iter,y=rand_Train_results$cost[-1],color="Random_Features_Training_Set"))+
  geom_line(aes(x=1:rand_Test_results$num_iter,y=rand_Test_results$cost[-1],color="Random_Features_Test_Set"))+
  ggtitle('Randomly Selected Features(alpha = 0.648 & Threshold = 0.00001)')+
  xlab('Number of iterations')+
  ylab('Cost')+
  scale_colour_manual(name="Legend",
                      values=c(Random_Features_Training_Set="Black",Random_Features_Test_Set="Cyan"))+
  theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6),
    legend.key = element_rect(colour = "black"))## Legend is added and repositioned

r<-ggplot()+geom_point(aes(x=1:Train_results$num_iter,y=Train_results$cost[-1],color="All_Features_Training_Set"))+
  geom_point(aes(x=1:Test_results$num_iter,y=Test_results$cost[-1],color="All_Features_Test_Set"))+
  geom_line(aes(x=1:Train_results$num_iter,y=Train_results$cost[-1],color="All_Features_Training_Set"))+
  geom_line(aes(x=1:Test_results$num_iter,y=Test_results$cost[-1],color="All_Features_Test_Set"))+
  ggtitle('All Features(alpha = 0.648 & Threshold = 0.00001)')+
  xlab('Number of iterations')+
  ylab('Cost')+
  scale_colour_manual(name="Legend",
                      values=c(All_Features_Training_Set="Purple",All_Features_Test_Set="Green"))+
  theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6),
    legend.key = element_rect(colour = "black"))## Legend is added and repositioned

#Plotting best, random features, and all features with alpha =0.648 and threshold=0.00001 

ggplot()+
  geom_point(aes(x=1:rand_Train_results$num_iter,y=rand_Train_results$cost[-1],color="Random_Features_Training_Set"))+
  geom_point(aes(x=1:rand_Test_results$num_iter,y=rand_Test_results$cost[-1],color="Random_Features_Test_Set"))+
  geom_point(aes(x=1:Train_results$num_iter,y=Train_results$cost[-1],color="All_Features_Training_Set"))+
  geom_point(aes(x=1:Test_results$num_iter,y=Test_results$cost[-1],color="All_Features_Test_Set"))+
  geom_line(aes(x=1:rand_Train_results$num_iter,y=rand_Train_results$cost[-1],color="Random_Features_Training_Set"))+
  geom_line(aes(x=1:rand_Test_results$num_iter,y=rand_Test_results$cost[-1],color="Random_Features_Test_Set"))+
  geom_line(aes(x=1:Train_results$num_iter,y=Train_results$cost[-1],color="All_Features_Training_Set"))+
  geom_line(aes(x=1:Test_results$num_iter,y=Test_results$cost[-1],color="All_Features_Test_Set"))+
  geom_point(aes(x=1:best_Train_results$num_iter,y=best_Train_results$cost[-1],color="Best_Features_TrainingSet"))+
  geom_point(aes(x=1:best_Test_results$num_iter,y=best_Test_results$cost[-1],color="Best_Features_TestSet"))+
  geom_line(aes(x=1:best_Train_results$num_iter,y=best_Train_results$cost[-1],color="Best_Features_TrainingSet"))+
  geom_line(aes(x=1:best_Test_results$num_iter,y=best_Test_results$cost[-1],color="Best_Features_TestSet"))+
  ggtitle('Comparing Best, Random Features and All Features(alpha = 0.648 & Threshold = 0.00001)')+
  xlab('Number of iterations')+
  ylab('Cost')+
  scale_colour_manual(name="Legend",
                      values=c(Random_Features_Training_Set="Red",Random_Features_Test_Set="Blue",
                               All_Features_Training_Set="Orange",All_Features_Test_Set="Green",
                               Best_Features_TrainingSet="Cyan",Best_Features_TestSet="Black"))+
  theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6),
    legend.key = element_rect(colour = "black"))## Legend is added and repositioned
##------------------------------------------------------------------------------------------------------------------
#Final Discussion:: Suggestion
##------------------------------------------------------------------------------------------------------------------
final_x_Train<-as.matrix(training_set[,c('hr','atemp','hum','season','holiday','weekday','windspeed')])
colnames(final_x_Train)
y_Train<-training_set[,12]
final_x_Test<-as.matrix(test_set[,c('hr','atemp','hum','season','holiday','weekday',"windspeed")])
colnames(final_x_Test)
y_Test<-test_set[,12]
z<-ncol(final_x_Train)
final_x_Train<-cbind(X0=1,final_x_Train)
final_x_Test<-cbind(X0=1,final_x_Test)
final_beta<-rand(z+1,1)
final_Train_results<-G(final_x_Train,y_Train,final_beta,th=0.00001,alpha=0.648)
final_Test_results<-G(final_x_Test,y_Test,final_beta,th=0.00001,alpha=0.648)
Train_results<-G(x_Train,y_Train,beta,th=0.00001,alpha=0.648)
Test_results<-G(x_Test,y_Test,beta,th=0.00001,alpha=0.648)
final_Train_results$cost
Train_results$cost

