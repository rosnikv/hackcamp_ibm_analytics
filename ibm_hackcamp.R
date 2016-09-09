# fb : DataScienceGroup
#srinivasgnv
#adityajoshi5
#bluemix docs/tutorial
#shailrishabh@gmail.com

train <- read.csv("/home/freestyler/hackcamp/titanic_train.csv")
test <- read.csv("/home/freestyler/hackcamp/titanic_test.csv")


#summary(train)# NA in age
#summary(test) # NA in fare

str(train$Survived)
# 0->1,1->2
train$Survived<-as.factor(train$Survived)

test$Survived <- 1
#combine both data
combine <- rbind(train,test)
set.seed(415)
summary(combine)

## Missing data
combine$Embarked[c(62,830)]="S"
combine$Fare[1044]<-median(combine$Fare,na.rm=T)

## Interpolate missing data via tree
library(rpart)
Agefit <- rpart(Age ~ Pclass + Sex + Fare + Embarked + SibSp + Parch, data=combine[!is.na(combine$Age),],method="anova")
combine$Age[is.na(combine$Age)] <- predict(Agefit,combine[is.na(combine$Age),])

###  Make variables

# child variable
combine$child <- 0
combine$child[combine$Age < 18] <- 1

# name information
combine$Name= as.character(combine$Name)
split_Title <- function(x){
  strsplit(x,split='[.,]')[[1]][2] #split for title from name
}
# Apply split to all rows
combine$Title <- sapply(combine$Name,FUN=split_Title)
# remove start whitespace
combine$Title <- sub(' ','',combine$Title)
#summary(as.factor(combine$Title))
combine$Title[combine$Title %in% c("Mme","Mlle")]<- "Mlle"
combine$Title[combine$Title %in% c("Capt","Don","Major","Sir")]<- "Sir"
combine$Title[combine$Title %in% c("Dona","Lady","the Countess","Jonkheer")] <- "Lady"

combine$Title <- factor(combine$Title)

# family size
combine$Family_size <- combine$SibSp + combine$Parch + 1
# parentless child
combine$Parentless <- 0
combine$Parentless[combine$child==1 & combine$Parch==0] <- 1

## cabin 
combine$Cabin = as.character(combine$Cabin)
combine$CabinL = sapply(combine$Cabin,FUN=function(x){strsplit(x,split='[,.]')[[1]][1]})
combine$CabinL <- factor(combine$CabinL)

#Family ID for large families
combine$Surname <- sapply(combine$Name,FUN=function(x){ strsplit(x,split='[,.]')[[1]][1]})
combine$FamilyID <- paste(as.character(combine$Family_size),combine$Surname,sep="")
combine$FamilyID[combine$Family_size <= 2] <- 'Small'
famIDs <- data.frame(table(combine$FamilyID))
famIDs <- famIDs[famIDs$Freq <=2,]
combine$FamilyID[combine$FamilyID %in% famIDs$Var1] <- 'Small'
combine$FamilyID <- factor(combine$FamilyID)

combine$FamilyID2 <- combine$FamilyID
combine$FamilyID2 <- as.character(combine$FamilyID2)
combine$FamilyID2[combine$Family_size <= 3] <- "Small"
combine$FamilyID2 <- factor(combine$FamilyID2)


######  MODELS   #############

train_new <- combine[1:891,]
test_new <- combine[892:1309,]
train_new$Survived <- as.numeric(train_new$Survived)
#! test_new$Sex <- as.numeric(test_new$Sex)

## 1. women and children first
fit <- lm(Survived ~ Sex + child, data=train_new)
summary(fit)

## 2.  1 + Class Model
fit2 <- lm(Survived ~ Sex + child + Pclass, data=train_new)
summary(fit2)

## Decision tree model

library(rpart)
fit3 <- rpart(Survived ~ Pclass + Sex + Age + Family_size + Fare + Embarked + Title + FamilyID, data=train_new,method="class")
pred3 <- predict(fit3,test_new,type="class")

## Random forest model
library(randomForest)
fit4 <- randomForest(as.factor(Survived)~  Pclass + Sex + Age + Fare + Embarked + Title + FamilyID2 + Family_size + SibSp + Parch,data=train_new,method="class")
pred4 <- predict(fit4,test_new,type="class")

## SVM model
library(class)
library(e1071)
fit5 <- svm(as.factor(Survived)~ Pclass + Sex + Age + Fare + Embarked + Title + FamilyID2 + Family_size + SibSp + Parch, data=train_new)
pred5 <- predict(fit5,test_new,type="class")
pred5b <- predict(fit5,train_new,type="class")
table(train_new$Survived,pred5b)

## ensemble model rpart, rf, svm

pred_svm <- pred5
pred_svm[pred5 != pred4 & pred5 != pred3 & pred_svm==1] <-2
pred_svm[pred5 != pred4 & pred5 != pred3 & pred_svm==2] <-1

predictions <- as.numeric(pred_svm)
predictions[pred_svm==1] <- 0
predictions[pred_svm==2] <- 1


## save/submission file

submit <- data.frame(PassengerId=test_new$PassengerId,Survived=predictions)
write.csv(submit,file="/home/freestyler/hackcamp/titanic_sample_submission2.csv",row.names = FALSE)
