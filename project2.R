library(tibble)
library(ggplot2)
library(dplyr)
library(tidyr)

df1 <- read.table('~/stats/154/project2/image_data/image1.txt')
df2 <- read.table('~/stats/154/project2/image_data/image2.txt')
df3 <- read.table('~/stats/154/project2/image_data/image3.txt')
columns <- c('y','x','expert','NDAI','SD','CORR','DF','CF','BF','AF','AN')
features <- c('expert','NDAI','SD','CORR','DF','CF','BF','AF','AN')
colnames(df1) <- columns
colnames(df2) <- columns
colnames(df3) <- columns
df <- rbind(df1,df2,df3)

#1b
perc1 <- df1 %>% group_by(expert) %>% summarise(round(n()/nrow(df1),2))
perc2 <- df2 %>% group_by(expert) %>% summarise(round(n()/nrow(df2),2))
perc3 <- df3 %>% group_by(expert) %>% summarise(round(n()/nrow(df3),2))
perc <- df %>% group_by(expert) %>% summarise(round(n()/nrow(df),2))

ggplot(data=df1,aes(x=x,y=y)) + geom_point(aes(color=expert)) + ggtitle('Expert Label Plot of Image1')
ggplot(data=df2,aes(x=x,y=y)) + geom_point(aes(color=expert)) + ggtitle('Expert Label Plot of Image2')
ggplot(data=df3,aes(x=x,y=y)) + geom_point(aes(color=expert)) + ggtitle('Expert Label Plot of Image3')

#1c
library(lattice)
corrDf <- function(data) {
  corr <- data %>% as.matrix %>% cor %>% as.data.frame %>% rownames_to_column(var = 'var1') %>% gather(var2, correlation, -var1)
  corr
}

filteredCorrDf <- function(corr) {
  corr1 <- corr %>% filter((correlation<1.0 & correlation>=0.6) | (correlation>-1.0 & correlation<=-.6))
  corr1$correlation <- round(corr1$correlation,4)
  corr1 <- corr1 %>% spread(var1,correlation) %>% arrange(order(features)) %>% column_to_rownames(var='var2')
  corr1[,features]
}

expertCorr <- function(corr) {
  corr %>% filter(var1 == 'expert')
}

corr1 <- filteredCorrDf(corrDf(df))
corr1

levelplot(as.matrix(corr1),main='matrix of significant correlations among features',xlab='features',ylab='features')

corr2 <- expertCorr(corrDf(df))
corr2
ggplot(corr2,aes(x=var2,y=correlation)) + geom_bar(stat='identity') + ggtitle('correlations of expert with features') + xlab('features')

mean <- df %>% group_by(expert) %>% summarise_all(funs(mean))
mean

#2a
TRAINING <- 0.60
VALIDATION <- 0.20
TEST <- 0.20

trainset <- df[which(is.na(df$text)), ]
validset <- df[which(is.na(df$text)), ]
testset <- df[which(is.na(df$text)), ]

for (i in c(-1,0,1)) {
  data <- df %>% filter(expert==i)
  trainSize <- floor(TRAINING*nrow(data))
  validSize <- floor(VALIDATION*nrow(data))
  testSize <- floor(TEST*nrow(data))
  set.seed(7)
  indicesTraining <- sort(sample(seq_len(nrow(data)), size=trainSize,replace=FALSE))
  indicesNotTraining <- setdiff(seq_len(nrow(data)), indicesTraining)
  indicesValidation <- sort(sample(indicesNotTraining, size=validSize,replace=FALSE))
  indicesTest <- setdiff(indicesNotTraining, indicesValidation)
  
  dfTraining <- df[indicesTraining, ]
  trainset <- rbind(trainset,dfTraining)
  dfValidation <- df[indicesValidation, ]
  validset <- rbind(validset,dfValidation)
  dfTest <- df[indicesTest, ]
  testset <- rbind(testset,dfTest)
}

#2b
(nrow(validset %>% filter(expert==-1)) + nrow(testset %>% filter(expert==-1))) / (nrow(validset) + nrow(testset))

#3a
#Method 1: Logistic Regression
columns <- c('V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11')
colnames(trainset) <- columns
colnames(validset) <- columns
colnames(testset) <- columns
cloud = rbind(trainset, validset)
##testing data here
library(scales)
library(caret)
#Cutting into k-fold
cloud<-cloud[sample(nrow(cloud)),]
folds <- cut(seq(1,nrow(cloud)),breaks=10,labels=FALSE)
#Linear Regression Model
logistic_regression_results = c(1:10)
for(i in 1:10) {
  index <- which(folds==i,arr.ind=TRUE)
  train <- cloud[index, ]
  test <- cloud[-index, ]
  glm.fits = glm( rescale(V3, to = c(0,1)) ~ V1+V2+V4+V5+V6+V7+V8+V9+V10+V11,data=train,family=binomial )
  glm.probs=predict (glm.fits,subset(test,select=c(1,2,4,5,6,7,8,9,10,11)),type="response")
  glm.pred=rep(-1 , nrow(test))
  glm.pred[glm.probs > 1/3]=0
  glm.pred[glm.probs > 2/3]=1
  table(glm.pred, test$V3)
  error = mean(glm.pred != test$V3)
  logistic_regression_results[i] = 1 - error
}
glm.fits = glm( rescale(V3, to = c(0,1)) ~ V1+V2+V4+V5+V6+V7+V8+V9+V10+V11, data=cloud,family=binomial)
glm.probs= predict (glm.fits,subset(testset,select=c(1,2,4,5,6,7,8,9,10,11)),type="response")
glm.pred=rep(-1 , nrow(testset))
glm.pred[glm.probs > 1/3]=0
glm.pred[glm.probs > 2/3]=1
table(glm.pred, testset$V3)
error = mean(glm.pred != testset$V3)
test_accuracy = 1 - error

logistic_regression_results

mean(logistic_regression_results)

test_accuracy

#Method 2: Linear Discriminant Analysis
library(MASS)
lda_results = c(1:10)
#Cutting into k-fold
cloud<-cloud[sample(nrow(cloud)),]
folds <- cut(seq(1,nrow(cloud)),breaks=10,labels=FALSE)
for(i in 1:10) {
  index <- which(folds==i,arr.ind=TRUE)
  train <- cloud[index, ]
  test <- cloud[-index, ]
  lda.fit = lda( V3 ~ V1+V2+V4+V5+V6+V7+V8+V9+V10+V11,data=train)
  lda.pred = predict(lda.fit, subset(test,select=c(1,2,4,5,6,7,8,9,10,11)))
  lda.class = lda.pred$class
  table(lda.class, test$V3)
  error = mean(lda.class != test$V3)
  lda_results[i] = 1 - error
}
lda.fit = lda( V3 ~ V1+V2+V4+V5+V6+V7+V8+V9+V10+V11,data=cloud)
lda.pred = predict(lda.fit, subset(testset,select=c(1,2,4,5,6,7,8,9,10,11)))
lda.class = lda.pred$class
table(lda.class, testset$V3)
error = mean(lda.class != testset$V3)
test_accuracy = 1 - error

lda_results

mean(lda_results)

test_accuracy

#Method 3: QDA
library(MASS)
qda_results = c(1:10)
#Cutting into k-fold
cloud<-cloud[sample(nrow(cloud)),]
folds <- cut(seq(1,nrow(cloud)),breaks=10,labels=FALSE)
for(i in 1:10) {
  index <- which(folds==i,arr.ind=TRUE)
  train <- cloud[index, ]
  test <- cloud[-index, ]
  qda.fit = qda( V3 ~ V1+V2+V4+V5+V6+V7+V8+V9+V10+V11,data=train)
  qda.pred = predict(qda.fit, subset(test,select=c(1,2,4,5,6,7,8,9,10,11)))
  qda.class = qda.pred$class
  table(qda.class, test$V3)
  error = mean(qda.class != test$V3)
  qda_results[i] = 1 - error
}
qda.fit = qda( V3 ~ V1+V2+V4+V5+V6+V7+V8+V9+V10+V11,data=cloud)
qda.pred = predict(qda.fit, subset(testset,select=c(1,2,4,5,6,7,8,9,10,11)))
qda.class = qda.pred$class
error = mean(qda.class != testset$V3)
test_accuracy = 1 - error

qda_results

mean(qda_results)

test_accuracy

#3b

#ROC for Logistic
library(pROC)
pROC_obj <- roc(testset$V3, glm.pred,
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
abline(v=0.5)

#ROC for LDA
library(pROC)
pROC_obj <- roc(testset$V3, as.numeric(lda.class) - 2,
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
abline(v=0.5)

#ROC for QDA
library(pROC)
pROC_obj <- roc(testset$V3, as.numeric(qda.class) - 2,
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
abline(v=0.5)





