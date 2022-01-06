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

mush = mush %>% select(-veil.type)
mush_x = mush %>% select(-class)
mush_y = mush %>% select(class)

set.seed(121)
size<- floor(0.7*nrow(mush))
train_ind <- sample(seq_len(nrow(mush)), size = size)
train<-mush[train_ind,]
test<-mush[-train_ind,]
train_y <- as.data.frame(mush_y[train_ind,])
test_y<-as.data.frame(mush_y[-train_ind,])
names(train_y) = 'class'
names(test_y) = 'class'

tree.mush <-rpart(class~.,data=train)
tree.mush$variable.importance

g = glm(class~  odor + gill.color + stalk.surface.below.ring + stalk.surface.above.ring + ring.type  , data = train ,family = 'binomial')
g2 = glm(class~ . ,data = train ,family = 'binomial')


summary(g2)

summary(g)

test5 <- test
test5$model_prob <- predict(g2, newdata = test5, type = "response")
library(dplyr)
test5 <- test5 %>% mutate(model_pred = 1*(model_prob > .51) + 0,
                          target_binary = 1*(class == 'p') + 0)

head(test5)


test5 <- test5 %>% mutate(accurate = 1*(model_pred == target_binary))
sum(test5$accurate)/nrow(test5)


coef(g2)


library(glmnet)

X = model.matrix(class ~ ., train)[, -1]
X2 = model.matrix(class ~ ., test)[, -1]

Y = train_y %>% 
  mutate(Edible = ifelse(class=='e', 1, 0),
          Poison = ifelse(class=='p', 1, 0))

Y = model.matrix(class~. ,Y)[,-1]
Y2 = test_y %>% 
  mutate(Edible = ifelse(class=='e', 1, 0),
          Poison = ifelse(class=='p', 1, 0))
Y2 = model.matrix(class~. ,Y2)[,-1]

fit_lasso = cv.glmnet(X, Y, alpha = 1,family = 'binomial')

true_test_y = 1*(test$class == 'p')
true_test_y

summary(fit_lasso)
coef(fit_lasso, s = "lambda.min")

plot(fit_lasso)


y_prob = predict(fit_lasso, X2, s = "lambda.min",type = "response")
y_pred = 1*(y_prob > .50)
Y_df = as.data.frame(Y)
head(y_pred)

dim(y_pred)
head(true_test_y)

cm <- confusionMatrix(factor(y_pred), reference = factor(true_test_y))
cm

install.packages('PRROC')
library(PRROC)

PRROC_obj <- roc.curve(scores.class0 = y_pred, weights.class0=true_test_y,
                       curve=TRUE)
plot(PRROC_obj)

names(y_pred) = 'y_pred_lasso_bin'

library(e1071)
	classifier_svm = svm(formula = class ~ .,
	                 data = train,
	                 type = 'C-classification',
	                 kernel = 'linear')
	y_pred_svm = predict(classifier_svm, newdata = test[-1])
	confusionMatrix(test$class, y_pred_svm)

true_test_y = 1*(test$class == 'p')
y_pred_svm_bin = 1*(y_pred_svm == 'p')
PRROC_obj <- roc.curve(scores.class0 = y_pred_svm_bin, weights.class0 = true_test_y,
                       curve=TRUE)
plot(PRROC_obj)



install.packages("rpart.plot", repos = "https://cran.us.r-project.org")
library(rpart.plot)
tree.mush <-rpart(class~.,data=train)
rpart.plot(tree.mush,extra= 106)

y_pred_dt = predict(tree.mush, newdata = test[-1],type = 'class')
y_pred_dt

confusionMatrix(y_pred_dt,test$class)

y_pred_tree_bin = 1*(y_pred_dt == 'p')
PRROC_obj <- roc.curve(scores.class0 = y_pred_tree_bin, weights.class0=true_test_y,
                       curve=TRUE)
plot(PRROC_obj)

library(class)
set.seed(121)
size<- floor(0.7*nrow(mush))
train_ind <- sample(seq_len(nrow(mush)), size = size)
train<-mush[train_ind,]
test<-mush[-train_ind,]
y2 <- train %>% select(class)
Xtr = as.data.frame( model.matrix(class ~ ., train)[, -1])
Xte = as.data.frame( model.matrix(class ~ ., test)[, -1])

Ytr = y2 %>% 
  mutate(Poison = ifelse(class=='p', 1, 0))


pr <- knn(Xtr,Xte,cl=(as.factor(train$class)),k=5)


tb <- confusionMatrix(pr,test$class)
tb

y_pred_knn_bin = 1*(pr == 'p')
PRROC_obj <- roc.curve(scores.class0 = pr, weights.class0=true_test_y,
                       curve=TRUE)
plot(PRROC_obj)

new_df = data.frame(true_test_y,y_pred_knn_bin,y_pred_tree_bin,y_pred_svm_bin,y_pred)
new_df


