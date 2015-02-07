############################################
##              Coded By                  ##
##           ANDRES FISCHLER              ## 
##    INSTITUTE ADVANCE ANALYTICS 2015    ## 
##       ANDRES_FISCHLER@NCSU.EDU         ##
##                                        ##
############################################

#Import Libraries
library(RSQLite)
library(survival)
library(splines)
library(doBy)
library(plyr)
library(caret)
library(pROC)
library(car)
library(lsmeans)
library(party)


#################################################################################
############        Connect to Database and Flatten Data                #########
#################################################################################

con <- dbConnect(dbDriver("SQLite"), dbname="C:/Users/AFischler/Desktop/Job Applications/RTI/exercises/exercise01/exercise01.sqlite")

#Retrieve list of databases
dbListTables(con)

#Inspect database elements
head(dbReadTable(con,"countries"))
head(dbReadTable(con,"education_levels"))
head(dbReadTable(con,"marital_statuses"))
head(dbReadTable(con,"occupations"))
head(dbReadTable(con,"races"))
head(dbReadTable(con,"records"))
head(dbReadTable(con,"relationships"))
head(dbReadTable(con,"sexes"))
head(dbReadTable(con,"workclasses"))

#Flatten Data
flat <- dbGetQuery(con,'SELECT a.id, a.age, a.education_num, a.capital_gain, a.capital_loss, 
                        a.hours_week, a.over_50K, b.name as education_level, c.name as marital_status,
                        d.name as occupation, e.name as race, f.name as relationship, 
                        g.name as sex, h.name as workclass, i.name as country
                        FROM records AS a
                        LEFT OUTER JOIN education_levels AS b on a.education_level_id=b.id
                        LEFT OUTER JOIN marital_statuses AS c on a.marital_status_id=c.id
                        LEFT OUTER JOIN occupations AS d on a.occupation_id=d.id
                        LEFT OUTER JOIN races AS e on a.race_id=e.id
                        LEFT OUTER JOIN relationships AS f on a.relationship_id=f.id
                        LEFT OUTER JOIN sexes AS g on a.sex_id=g.id                   
                        LEFT OUTER JOIN workclasses AS h on a.workclass_id=h.id                    
                        LEFT OUTER JOIN countries AS i on a.country_id=i.id                    
                        ')

## Write CSV
write.csv(flat, file=paste(getwd(),"/RTI.csv", sep=""), row.names=FALSE)


rm(con)

#################################################################################
############                       Explore the Data                     #########
#################################################################################

summary(flat)
str(flat)

#Convert "?" into NA
flat[ flat == "?" ] = NA

#Convert 99999 into NA
flat[ flat == 99999 ] = NA

#Convert education_num to categorical
flat$education_num2 <- as.character(flat$education_num) 


#Combine levels 1 and 2 into level 2 due to Quasiseparation later
flat$education_num2[flat$education_num == "1"] <- "2"


# Charts

a<- table(flat$sex, flat$over_50k)
barplot(a, main="Proportion Making Over 50K by Gender", xlab="Gender", names.arg=c("Males", "Females"), ylim=c(0,35000), col=c("darkblue","lightblue"))

b<- table(flat$over_50k, flat$education_level )
barplot(b, main="Proportion Making Over 50K by Gender", col=c("darkblue","lightblue"))

c<- table(flat$over_50k, flat$age)
barplot(rbind(c[2,],c[1,]), main="Population by Earnings and Age", legend = c("Over 50K","Under 50k"), ylim=c(0,1500), col=c("darkblue","lightblue"))

rm (a, b, c)

#################################################################################
############                       Create Data Partition                #########
#################################################################################


#Creates partition keeping approximately same proportion of events to non-events
set.seed(3456)

Split1 <- createDataPartition(flat$over_50k, p = 0.8, list = F)
interm <- flat[Split1,] #will become training/validation
dfTest <- flat[-Split1,] #Test data set

Split2 <- createDataPartition(interm$over_50k, p = 0.8, list = F)
dfTrain <- interm[Split2,] #Train data set
dfValid <- interm[-Split2,] #Validation data set

rm(interm,Split1,Split2)


#################################################################################
############                       Logistic Model                       #########
#################################################################################

#Remove incomplete cases
dfTrain2 <- dfTrain[complete.cases(dfTrain),] #28789 obs remain in training dataset

#####   Education_num  ######
table(dfTrain2$education_num, dfTrain2$over_50k) # quasi-separation, fixed above at flat level

### Setting Alpha at 0.001 ###


########################### Build Model #########################
attach(dfTrain2)

Over50 <- factor(over_50k, levels=c(0,1), labels=c("No", "Yes"))


############ Variable Selection Techniques ##############
#Note to Self --> Look into LASSO / Ridge techniques when time-permitting

nullmodel <- glm(Over50 ~ 1, data=dfTrain2, family=binomial(link="logit"))
fullmodel <- glm(Over50 ~ age + country + education_num2 + marital_status + occupation + race + relationship
                 + sex + workclass + capital_gain + capital_loss + hours_week, data=dfTrain2,
                 family=binomial(link="logit"))

#Stepwise Selection
Stepwise <- step(nullmodel,scope=list(lower=nullmodel, upper=fullmodel), direction="both")
Stepwise
#Backward Selection
back <- step(fullmodel, direction="backward")
back
#Forward Selection
forth <- step(fullmodel, direction="forward")
forth
##### All Variable selection Techniques select the full model

#Examine full model
Model1 <- glm(Over50 ~ age + country + education_num2 + marital_status + occupation + race + relationship
              + sex + workclass + capital_gain + capital_loss + hours_week, data=dfTrain2,
              family=binomial(link="logit"))
summary(Model1) # AIC 18729, Country and Race not significant at 0.001 level

Model2 <- glm(over_50k ~ age + education_num2 + marital_status + occupation + relationship
              + sex + workclass +  capital_gain + capital_loss + hours_week, data=dfTrain2,
              family=binomial(link="logit"))
summary(Model2) # AIC 18728 -- Excludes Country and Race


# Comparing ROC Curves #
Model1.ROC <- roc(Model1$y, Model1$fitted)
print(Model1.ROC)
plot(Model1.ROC)

Model2.ROC <- roc(Model2$y, Model2$fitted)
print(Model2.ROC)
plot(Model2.ROC)

roc.test(Model1.ROC, Model2.ROC) ## Reject Null, Model1 and Model2 are different

#Plot both Models ROC Curves
plot(x=1-Model1.ROC$specificities, y=Model1.ROC$sensitivities, type="l", col="red", xlab="1 - Specificity", ylab="Sensitivity", main="Comparing 2 ROC Curves")
lines(x=1-Model2.ROC$specificities, y=Model2.ROC$sensitivities, col="blue")
abline(a=0, b=1, col="gray", lty="dashed")

####### Going forward with Model 2 ########

# Checking Linearity of Age #
Assump.test <- data.frame(dfTrain2, Age.LogAge = age*log(age))
attach(Assump.test)
names(Assump.test)


Assump.Check <- glm(over_50k ~ age + education_level + marital_status + occupation + relationship
              + sex + workclass +  capital_gain + capital_loss + hours_week + Age.LogAge, data=Assump.test,
              family=binomial(link="logit"))
summary(Assump.Check)

## Age fails test, need Box-Tidwell or categorical transformation --> Decide to create bins

#Create Binned Age Vatiable
dfTrain2$Age_bin <- ifelse(dfTrain2$age < 25, "17-24", 
                           ifelse(dfTrain2$age >= 25 & dfTrain2$age < 35, "25-34", 
                                  ifelse(dfTrain2$age >= 35 & dfTrain2$age < 45, "35-44",
                                         ifelse(dfTrain2$age >=45 & dfTrain2$age < 55, "45-54",
                                                ifelse(dfTrain2$age >=55 & dfTrain2$age < 65, "55-64",
                                                       ifelse(dfTrain2$age >= 65 & dfTrain2$age < 75, "65-74",
                                                              ifelse(dfTrain2$age >= 75 & dfTrain2$age < 85, "75-84", "85+")))))))


Model2<- glm(over_50k ~ Age_bin + education_level + marital_status + occupation + relationship +
                   sex + workclass +  capital_gain + capital_loss + hours_week, 
                   data=dfTrain2, family=binomial(link="logit"))
summary(Model2)

attach(dfTrain2)
####### Checking Linearity of capital_gain #
Assump.test <- data.frame(dfTrain2, gain2 = capital_gain+1, gain.LogGain = (capital_gain+1)*log((capital_gain+1)))
attach(Assump.test)
names(Assump.test)


Assump.Check <- glm(over_50k ~ Age_bin + education_level + marital_status + occupation + relationship
                    + sex + workclass +  capital_gain + capital_loss + hours_week + gain.LogGain, 
                    data=Assump.test, family=binomial(link="logit"))
summary(Assump.Check) 

attach(dfTrain2)

#Capital_gain fails test, need Box-Tidwell or categorical transformation --> 
# Decided to create bins using splits on tree

gain.splits <- ctree(over_50k ~ capital_gain, data=dfTrain2, controls=ctree_control(maxdepth=5))       
plot(gain.splits)

#splits <5000, 5000 - 7000, >7001

#Create bin
dfTrain2$gain_bin <- ifelse(dfTrain2$capital_gain < 5000, "0-4999", 
                           ifelse(dfTrain2$capital_gain >= 5000 & dfTrain2$capital_gain < 7000, "5000-6999","7000+")) 
                                  

attach(dfTrain2)

####### Checking Linearity of capital_loss #
Assump.test <- data.frame(dfTrain2, loss2 = capital_loss + 1, loss.Logloss = (capital_loss+1)*log((capital_loss+1)))
attach(Assump.test)
names(Assump.test)


Assump.Check <- glm(over_50k ~ Age_bin + education_level + marital_status + occupation + relationship
                    + sex + workclass +  gain_bin + loss2 + hours_week + loss.Logloss, 
                    data=Assump.test)
summary(Assump.Check) 

attach(dfTrain2)

#Capital_loss fails test, need Box-Tidwell or categorical transformation --> 
# Decided to create bins using splits on tree

loss.splits <- ctree(over_50k ~ capital_loss, data=dfTrain2, controls=ctree_control(maxdepth=3))       
plot(loss.splits)

#splits <1600, 1600-1800, 1800 - 2000, >2001

#Create bin
dfTrain2$loss_bin  <- ifelse(dfTrain2$capital_loss < 1800, "0-1799", 
                        ifelse(dfTrain2$capital_loss >= 1800 & dfTrain2$capital_loss < 2000, "1800-1999","2000+"))


####### Checking Linearity of hours_week #
Assump.test <- data.frame(dfTrain2, week.Logweek = hours_week*log(hours_week))
attach(Assump.test)
names(Assump.test)


Assump.Check <- glm(over_50k ~ Age_bin + education_level + marital_status + occupation + relationship
                    + sex + workclass +  gain_bin + loss_bin + hours_week + week.Logweek, 
                    data=Assump.test)
summary(Assump.Check) # No adjustment needed

attach(dfTrain2)


#### Rerun Model2 with new variables

Model2<- glm(over_50k ~ Age_bin + education_num2 + marital_status + occupation + relationship +
                     sex + workclass +  gain_bin + loss_bin + hours_week, 
             data=dfTrain2, family=binomial(link="logit"))
summary(Model2)

################### Set Coding for Categorical Variables ##########

#### Set coding to Reference

#Education - Ref level HS grad
dfTrain2$education_num2 <- factor(dfTrain2$education_num2 )
contrasts(dfTrain2$education_num2 ) <- contr.treatment(nlevels(dfTrain2$education_num2), base = 9)

#Sex- Ref level Female
dfTrain2$sex <- factor(dfTrain2$sex )
contrasts(dfTrain2$sex ) <- contr.treatment(nlevels(dfTrain2$sex), base = 1)


#### Set coding to Effects

dfTrain2$Age_bin <- factor(dfTrain2$Age_bin)
contrasts(dfTrain2$Age_bin) <- contr.sum(nlevels(dfTrain2$Age_bin))

dfTrain2$marital_status <- factor(dfTrain2$marital_status )
contrasts(dfTrain2$marital_status ) <- contr.sum(nlevels(dfTrain2$marital_status ))

dfTrain2$occupation <- factor(dfTrain2$occupation )
contrasts(dfTrain2$occupation ) <- contr.sum(nlevels(dfTrain2$occupation ))

dfTrain2$relationship <- factor(dfTrain2$relationship)
contrasts(dfTrain2$relationship) <- contr.sum(nlevels(dfTrain2$relationship ))

dfTrain2$workclass <- factor(dfTrain2$workclass )
contrasts(dfTrain2$workclass) <- contr.sum(nlevels(dfTrain2$workclass ))

dfTrain2$gain_bin <- factor(dfTrain2$gain_bin )
contrasts(dfTrain2$gain_bin) <- contr.sum(nlevels(dfTrain2$gain_bin ))

dfTrain2$loss_bin <- factor(dfTrain2$loss_bin)
contrasts(dfTrain2$loss_bin) <- contr.sum(nlevels(dfTrain2$loss_bin ))


#### Rerun Model2 with new coding

Model2<- glm(over_50k ~ Age_bin + education_num2 + marital_status + occupation + relationship +
                     sex + workclass +  gain_bin + loss_bin + hours_week, 
             data=dfTrain2, family=binomial(link="logit"))
summary(Model2)

#####Diagnostics

#Cooks D
cutoff <- 4/(nrow(dfTrain2)-length(Model2$coefficients)-2)
plot(Model2, which=4)
abline(h=cutoff, lty="dashed", col="red")

#DFBetas
dfbetaPlots(Model2)

#DFFits
round(dffits(Model2),3)[dffits(Model2)>.5]

### All diagnostics look good



# ROC Curves #

Model2.ROC <- roc(Model2$y, Model2$fitted)
print(Model2.ROC)
plot(Model2.ROC) #AUC .9109




######Classification Tables##

#Create Youden J Statistic
Youden.J <-  Model2.ROC$sensitivities + Model2.ROC$specificities -1

#Create Classification Table
Class.Table <- cbind(Model2.ROC$thresholds, Model2.ROC$sensitivities, Model2.ROC$specificities, Youden.J)
colnames(Class.Table) <- c("Probability", "Sensitivity", "Specificity", "Youden.J")

#Find Optimal Cutoff
head(Class.Table[order(-Youden.J),]) # <-- Optimal cutoff is 0.2616013 

#Create Accuracy Table
classDF <- data.frame(response = Model2$y, predicted = (Model2$fitted.values > 0.2616013 ))
xtabs(~ predicted + response, data= classDF)

# Model has 82% accuracy


########## Odds Ratios ############
OR <- exp(coef(Model2)[-1])
OR
OR.CI <- exp(cbind(OR = coef(Model2), confint(Model2, level=0.999)))[-1,]
OR.CI
OR.CI2 <- exp(cbind(OR = coef(Model2), confint(Model2, level=0.999)))[-1,]
OR.CI

#################################################################################
############                Checking with Validation Data               #########
#################################################################################

dfValid2 <- dfValid

#Create Binned Age Variable
dfValid2$Age_bin <- ifelse(dfValid2$age < 25, "17-24", 
                           ifelse(dfValid2$age >= 25 & dfValid2$age < 35, "25-34", 
                                  ifelse(dfValid2$age >= 35 & dfValid2$age < 45, "35-44",
                                         ifelse(dfValid2$age >=45 & dfValid2$age < 55, "45-54",
                                                ifelse(dfValid2$age >=55 & dfValid2$age < 65, "55-64",
                                                       ifelse(dfValid2$age >= 65 & dfValid2$age < 75, "65-74",
                                                              ifelse(dfValid2$age >= 75 & dfValid2$age < 85, "75-84", "85+")))))))

#Create gain bin
dfValid2$gain_bin <- ifelse(dfValid2$capital_gain < 5000, "0-4999", 
                            ifelse(dfValid2$capital_gain >= 5000 & dfValid2$capital_gain < 7000, "5000-6999","7000+")) 

#Create loss bin
dfValid2$loss_bin  <- ifelse(dfValid2$capital_loss < 1800, "0-1799", 
                             ifelse(dfValid2$capital_loss >= 1800 & dfValid2$capital_loss < 2000, "1800-1999","2000+"))


#Remove incomplete cases
dfValid2 <- dfValid2[complete.cases(dfValid2),] #7238 obs remain in training dataset


##plot predictions
plot(data.frame('Predicted'=predict(Model2, dfValid2, type="response"), 'Observed'=dfValid2$over_50k))

#Create Accuracy Table
classDF2 <- data.frame(response = dfValid2$over_50k, predicted = (predict(Model2, dfValid2, type="response") > 0.2616013 ))
xtabs(~ predicted + response, data= classDF2)

#### Accuracy = 81%, Sensitivity = 82%, Specificity = 80%



#################################################################################
############                Checking with Validation Data               #########
#################################################################################

dfTest2 <- dfTest

#Create Binned Age Variable
dfTest2$Age_bin <- ifelse(dfTest2$age < 25, "17-24", 
                           ifelse(dfTest2$age >= 25 & dfTest2$age < 35, "25-34", 
                                  ifelse(dfTest2$age >= 35 & dfTest2$age < 45, "35-44",
                                         ifelse(dfTest2$age >=45 & dfTest2$age < 55, "45-54",
                                                ifelse(dfTest2$age >=55 & dfTest2$age < 65, "55-64",
                                                       ifelse(dfTest2$age >= 65 & dfTest2$age < 75, "65-74",
                                                              ifelse(dfTest2$age >= 75 & dfTest2$age < 85, "75-84", "85+")))))))

#Create gain bin
dfTest2$gain_bin <- ifelse(dfTest2$capital_gain < 5000, "0-4999", 
                            ifelse(dfTest2$capital_gain >= 5000 & dfTest2$capital_gain < 7000, "5000-6999","7000+")) 

#Create loss bin
dfTest2$loss_bin  <- ifelse(dfTest2$capital_loss < 1800, "0-1799", 
                             ifelse(dfTest2$capital_loss >= 1800 & dfTest2$capital_loss < 2000, "1800-1999","2000+"))


#Remove incomplete cases
dfTest2 <- dfTest2[complete.cases(dfTest2),] #8966 obs remain in training dataset


##plot predictions
plot(data.frame('Predicted'=predict(Model2, dfTest2, type="response"), 'Observed'=dfTest2$over_50k))

#Create Accuracy Table
classDF3 <- data.frame(response = dfTest2$over_50k, predicted = (predict(Model2, dfTest2, type="response") > 0.2616013 ))
xtabs(~ predicted + response, data= classDF3)

#### Accuracy = 81%, Sensitivity = 83%, Specificity = 81%

