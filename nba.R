## Libraries
library(caret)
library(rpart)
library(rpart.plot)
library(RWeka)
library(tm)
library(plyr)


## Set working directory
setwd("~/nba-predictions/")

## Open csv
nba  <- read.csv("nba_stats_13-14.csv")

## Open Team %
teamPer  <- read.csv("teamStats.csv")

## Team Classification
nba$team  <- gsub(".*\\(|\\)","", nba$STATS)

## Merge Data
nba  <- merge(nba, teamPer, by="team", all.x = TRUE)

## Clean names
nba$STATS  <- gsub("\\\n.*","", nba$STATS)

## Custom Classification
nba$custom  <- (nba$APG * 1.1 + nba$ORPG * 1.2 + nba$DRPG + nba$PPG - nba$TPG + nba$winning)/nba$MPG
#nba$custom  <- ifelse(is.na(nba$custom), nba$FPPM,  nba$custom)

## Correlation
cor(nba[, c("PPG", "APG", "BPG", "SPG", "DRPG", "ORPG","TPG","custom")])

# Analysis for Fantasy Points Per Minute ----

## Create data Partition
set.seed(999)
nbaPred  <- subset(nba, nba$team!="*FA")
nbaPred <- nbaPred[,-c(1,2,3,4,6,7,8,10,20,21,22,23,24,25)]
inTraining  <- createDataPartition(nbaPred$custom, p = .75, list = FALSE)
training  <- nbaPred[inTraining, ]; testing  <- nbaPred[-inTraining,]

## Train Model - glm
glmModel  <- train(custom ~ . ,
                   data = training, method="glm", preProcess="scale")
summary(glmModel)

## Predict
predictGlm  <- predict(custom, newdata = testing)
mean(sqrt((testing$custom - predictGlm)^2), na.rm = TRUE)
summary(testing$custom)

## Classification and Regression Trees
# CART algorithm 
mpartModel  <- rpart(custom ~ . , data = training)
mpartPred  <- predict(mpartModel, newdata = testing)
mean(sqrt((testing$custom - mpartPred)^2))

summary(mpartModel)
rpart.plot(mpartModel, type =3, fallen.leaves = TRUE)

## M5P - Weka
m5pModel  <- M5P(custom ~ . , data = training)
m5pPred  <- predict(m5pModel, newdata = testing)
summary(m5pModel)
m5pModel
mean(sqrt((testing$custom - m5pPred)^2))

## GBM - Boost
boostModel <- train(custom ~ ., data = training, method="gbm")
boostPred  <- predict(boostModel, newdata = testing)
boostModel
mean(sqrt((testing$custom - boostPred)^2))

## Comparing Models
summary(predictGlm)
summary(mpartPred)
summary(m5pPred)
summary(boostPred)
summary(testing$custom)

boxplot(testing$custom, boostPred, m5pPred, mpartPred,predictGlm)

## Quick Analysis ----
nba[ order(-nba[,27]), c(1,2,4,23,27)]

nbaFantasy  <- function(csv, position) {
  nba  <- read.csv(csv)
  teamPer  <- read.csv("teamStats.csv")
  nba$team  <- gsub(".*\\(|\\)","", nba$STATS)
  nba  <- merge(nba, teamPer, by="team", all.x = TRUE)
  nba$STATS  <- gsub("\\\n.*","", nba$STATS)
  nba$custom  <- (nba$APG * 1.1 + nba$ORPG * 1.2 + nba$DRPG + nba$PPG - nba$TPG + nba$winning)/nba$MPG
  
  position <- as.character(position)
  subsetPos  <- subset(nba, nba=="position")
  #print
  subsetPos[ order(-subsetPos[,27]), c(1,2,4,23,26,27)]
}

## Three year analysis
nba2k4 <- read.csv("nba_stats_13-14.csv")
nba2k3 <- read.csv("nba_stats_12-13.csv")
nba2k2 <- read.csv("nba_stats_11-12.csv")

nba2k4$team  <- gsub(".*\\(|\\)","", nba2k4$STATS)
nba2k4$STATS  <- gsub("\\\n.*","", nba2k4$STATS)
nba2k4$STATS  <-  removePunctuation(nba2k4$STATS)
nba2k4$custom  <- (nba2k4$APG * 1.1 + nba2k4$ORPG * 1.2 + nba2k4$DRPG + nba2k4$PPG - nba2k4$TPG)/nba2k4$MPG

nba2k3$team  <- gsub(".*\\(|\\)","", nba2k3$STATS)
nba2k3$STATS  <- gsub("\\\n.*","", nba2k3$STATS)
nba2k3$STATS  <-  removePunctuation(nba2k3$STATS)
nba2k3$custom  <- (nba2k3$APG * 1.1 + nba2k3$ORPG * 1.2 + nba2k3$DRPG + nba2k3$PPG - nba2k3$TPG)/nba2k3$MPG

nba2k2$team  <- gsub(".*\\(|\\)","", nba2k2$STATS)
nba2k2$STATS  <- gsub("\\\n.*","", nba2k2$STATS)
nba2k2$STATS  <-  removePunctuation(nba2k2$STATS)
nba2k2$custom  <- (nba2k2$APG * 1.1 + nba2k2$ORPG * 1.2 + nba2k2$DRPG + nba2k2$PPG - nba2k2$TPG)/nba2k2$MPG

# Merge DataFrames
nbaAll  <- merge(nba2k2, nba2k3, by="STATS", suffixes=c("2012","2013"), all = TRUE)
nbaAll  <- merge(nbaAll, nba2k4, by="STATS", all = TRUE)

# Eliminate vars
nba  <- nbaAll[, -c(2,3,4,5,6,7,20,21,26,27,28,29,31,44,45,51,54,55,68,69)]

# Estimate means
nba$meanCustom  <- (nba$custom2012 + nba$custom2013 + nba$custom)/3
nba$meanFPPM <- (nba$FPPM2012 + nba$FPPM2013 + nba$FPPM)/3

# Convert to character
nba$FAN  <- as.character(nba$FAN)

# Explore
#print
nbaList <- nba[ order(-nba[,51]), c(1,35,36,52,53,54,51,55)]
nbaList <- subset(nbaList, nbaList$P=="G" | nbaList$P=="GF")
nbaList


# Prediction
prediction <- tapply(nba2k4$FPPM, nba2k4$FAN, mean)
prediction
pred <- data.frame(prediction)

cbind(sort(pred$prediction, decreasing = TRUE))
boxplot(pred$prediction)
