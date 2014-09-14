## Libraries
library(caret)
library(rpart)
library(rpart.plot)
library(RWeka)

## Set working directory
setwd("~/nba/")

## Open csv
nba  <- read.csv("nba_stats.csv")

## Open Team %
teamPer  <- read.csv("teamStats.csv")

## Team Classification
nba$team  <- gsub(".*\\(|\\)","", nba$STATS)

## Merge Data
nba  <- merge(nba, teamPer, by="team", all.x = TRUE)

## Clean names
nba$STATS  <- gsub("\\\n.*","", nba$STATS)

## Custom Classification
nba$custom  <- nba$APG * 1.1 + nba$ORPG * 1.2 + nba$DRPG + nba$PPG + nba$TPG + nba$winning
nba$custom  <- ifelse(is.na(nba$custom), nba$FPPG,  nba$custom)

## Correlation
cor(nba[, c("PPG", "APG", "BPG", "SPG", "DRPG", "ORPG","TPG","custom")])

# Analysis for Fantasy Points Per Minute ----

## Create data Partition
set.seed(999)
nbaPred <- nba[,-c(1,2,3,4,6,7,8,10,20,21,22,24,25,27)]
inTraining  <- createDataPartition(nbaPred$FPPM, p = .75, list = FALSE)
training  <- nbaPred[inTraining, ]; testing  <- nbaPred[-inTraining,]

## Train Model - glm
glmModel  <- train(FPPM ~ . ,
                   data = training, method="glm", preProcess="scale")
summary(glmModel)

## Predict
predictGlm  <- predict(glmModel, newdata = testing)
mean(sqrt((testing$FPPM - predictGlm)^2), na.rm = TRUE)
summary(testing$FPPM)

## Classification and Regression Trees
# CART algorithm 
mpartModel  <- rpart(FPPM ~ . , data = training)
mpartPred  <- predict(mpartModel, newdata = testing)
mean(sqrt((testing$FPPM - mpartPred)^2))

summary(mpartModel)
rpart.plot(mpartModel, type =3, fallen.leaves = TRUE)

## M5P - Weka
m5pModel  <- M5P(FPPM ~ . , data = training)
m5pPred  <- predict(m5pModel, newdata = testing)
summary(m5pModel)
m5pModel
mean(sqrt((testing$FPPM - m5pPred)^2))

## GBM - Boost
boostModel <- train(FPPM ~ ., data = training, method="gbm")
boostPred  <- predict(boostModel, newdata = testing)
mean(sqrt((testing$FPPM - boostPred)^2))

## Comparing Models
summary(predictGlm)
summary(mpartPred)
summary(m5pPred)
summary(boostPred)
summary(testing$FPPM)

## Exploratory Analysis ----
nba[ order(-nba[,23]), c(1,2,4,21,22,23,26,27)]

teste  <- nba$FPPG / nba$MPG
round(teste - nba$FPPM, 2)
