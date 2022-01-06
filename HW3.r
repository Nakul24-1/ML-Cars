library(tidyverse)
install.packages('caret')
install.packages("glmnet", repos = "https://cran.us.r-project.org")
library(caret)
require(gh)
library(stringr)
tmp = tempfile()
qurl = 'https://raw.githubusercontent.com/Nakul24-1/ML-Cars/main/mushrooms.csv'
gh(paste0('GET ', qurl), .destfile = tmp, .overwrite = TRUE)

library(rpart)

mush = read.csv(tmp,stringsAsFactors = T)
head(mush)
mush$veil.type

set.seed(121)

mush = mush %>% select(-veil.type)
mush_x = mush %>% select(-class)
mush_y = mush %>% select(class)

size<- floor(0.7*nrow(mush))
train_ind <- sample(seq_len(nrow(mush)), size = size)
train<-mush[train_ind,]
test<-mush[-train_ind,]
train_y <- as.data.frame(mush_y[train_ind,])
test_y<-as.data.frame(mush_y[-train_ind,])
names(train_y) = 'class'
names(test_y) = 'class'
true_test_y = 1*(test$class == 'p')
true_test_y

install.packages('doParallel')
install.packages('randomForest')
library(doParallel)
library(future)


library(randomForest)
	classifier_rf = randomForest(x = train[-1], y = train$class, 
	                             data = train,ntree = 100)
	y_pred_rf = predict(classifier_rf, newdata = test[-1])
	confusionMatrix(test$class, y_pred_rf)

install.packages('PRROC')
library(PRROC)



PRROC_obj <- roc.curve(scores.class0 = y_pred_rf, weights.class0 = true_test_y,
                       curve=TRUE)
plot(PRROC_obj) 

install.packages('fastAdaboost')
library(fastAdaboost)



ad <- adaboost(class ~., data = train, tree_depth = 5, n_rounds = 5,10)


y_pred_ada = predict(ad, newdata = test[-1])
y_pred_ada$class

	confusionMatrix(y_pred_ada$class,test$class)


PRROC_obj <- roc.curve(scores.class0 = as.factor(y_pred_ada$class) , weights.class0 = true_test_y,
                       curve=TRUE)
plot(PRROC_obj) 

bag <- bagging(class ~., data = train,30)

y_pred_bag = predict(bag, newdata = test[-1])
y_pred_bag$class

confusionMatrix(as.factor(y_pred_bag$class),test$class)

PRROC_obj <- roc.curve(scores.class0 = as.factor(y_pred_bag$class) , weights.class0 = true_test_y,
                       curve=TRUE)
plot(PRROC_obj)

cl <- makeCluster(3)


setDefaultCluster(cl)
registerDoParallel(cl)

system.time({
classifier_rf = randomForest(x = train[-1], y = train$class, 
	                             data = train,ntree = 250)
bag <- bagging(class ~., data = train,35)
ad <- adaboost(class ~., data = train, tree_depth = 10, n_rounds = 5,10)
})



stopCluster(cl) # close multi-core cluster
rm(cl)


system.time({
classifier_rf = randomForest(x = train[-1], y = train$class, 
	                             data = train,ntree = 250)
bag <- bagging(class ~., data = train,35)
ad <- adaboost(class ~., data = train, tree_depth = 10, n_rounds = 5,10)
  })


