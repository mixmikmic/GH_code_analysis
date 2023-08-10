import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas

get_ipython().magic('load_ext rpy2.ipython')
get_ipython().magic('R library(lme4)')

get_ipython().magic("R data <- read.csv('http://www.stat.wisc.edu/~ane/st572/data/floweringTime.txt', sep='\\t')")
get_ipython().magic('R data <- subset(data, !is.na(dtf))')
get_ipython().magic('R data$logdtf <- log(data$dtf)')
get_ipython().magic('R fit <- lmer(logdtf ~ (1|subspecies)+(1|inventoryID), data=data, REML=F)')
get_ipython().magic('R print(summary(fit))')

data = pandas.read_csv('http://www.stat.wisc.edu/~ane/st572/data/floweringTime.txt', delimiter='\t', index_col=0)
data = data.dropna()
data['logdtf'] = np.log(data['dtf'])
vcf = {"subspecies" : "0 + C(subspecies)", "inventoryID" : "0 + C(inventoryID)"}

model = sm.MixedLM.from_formula('logdtf ~ 1', groups="subspecies", vc_formula=vcf,  data=data)
result = model.fit(reml=False)
result.summary()

get_ipython().magic('R fit.noinventory <- update(fit, .~.- (1|inventoryID))')
get_ipython().magic('R print(summary(fit.noinventory))')

#data = pandas.read_csv('http://www.stat.wisc.edu/~ane/st572/data/floweringTime.txt', delimiter='\t', index_col=0)
#data = data.dropna()
#data['logdtf'] = np.log(data['dtf'])
vcf = {"subspecies" : "1 + C(subspecies)"}
model_noinventory = sm.MixedLM.from_formula('logdtf ~ 1', groups="subspecies",  vc_formula=vcf, data=data)
result_noinventory = model_noinventory.fit(reml=False)
result_noinventory.summary()

get_ipython().magic('R print(anova(fit, fit.noinventory))')

result.compare_lr_test(result_noinventory)

get_ipython().magic("R data <- read.table('http://www.stat.wisc.edu/~ane/st572/data/bilingual.txt', header=T)")
get_ipython().magic('R fit = lmer(score ~ phase * time * language + (1|pair) + (1|child), data=data)')
get_ipython().magic('R print(summary(fit))')

data = pandas.read_table('bilingual.txt', skipinitialspace=True)
vcf = {"pair" : "0 + C(pair)", "child" : "0 + C(child)"}
model = sm.MixedLM.from_formula('score ~  phase*time*language', groups='pair', vc_formula=vcf, data=data)
result = model.fit(reml=False)
print (result.summary())

#%R fitno3way = lmer(score ~ (1|pair) + (1|child), data=data)
get_ipython().magic('R fitno3way <- update(fit, .~. - phase:time:language)')
get_ipython().magic('R print(summary(fitno3way))')

modelno3way = sm.MixedLM.from_formula('score ~  phase*time*language - phase:language:time', groups='pair', vc_formula=vcf, data=data)
resultno3way = modelno3way.fit(reml=False)
print (resultno3way.summary())

result.compare_lr_test(resultno3way)

get_ipython().magic('R print(anova(fit, fitno3way))')

get_ipython().magic("R data <- read.csv('http://www-personal.umich.edu/~bwest/classroom.csv')")

get_ipython().magic('R model <- lmer(mathgain ~ 1 + (1|schoolid) + (1|classid), REML=F, data=data)')
get_ipython().magic('R print(summary(model))')

data = pandas.read_csv('http://www-personal.umich.edu/~bwest/classroom.csv')
vcf = {"schoolid" : "1 + C(schoolid)", "classid" : "1 + C(classid)"}

model = sm.MixedLM.from_formula('mathgain ~ 1', groups="schoolid", vc_formula=vcf,  data=data)
result = model.fit(reml=False)
result.summary()

get_ipython().magic('R model2 <- lmer(mathgain ~ mathkind + sex + minority + ses +(1|schoolid) + (1|classid), data=data, REML = F)')
get_ipython().magic('R print(summary(model2))')

data = pandas.read_csv('http://www-personal.umich.edu/~bwest/classroom.csv')
vcf = {"schoolid" : "1 + C(schoolid)", "classid" : "1 + C(classid)"}

model2 = sm.MixedLM.from_formula('mathgain ~ mathkind + sex + minority + ses', groups="schoolid", vc_formula=vcf,  data=data)
result2 = model2.fit(reml=False)
result2.summary()

get_ipython().magic('R print(anova(model,model2))')

result2.compare_lr_test(result)

get_ipython().magic('R model3 <- lmer(mathgain ~   minority +(mathkind|schoolid) + (sex|schoolid) + (ses|schoolid) , data=data, REML = F)')
get_ipython().magic('R print(summary(model3))')

get_ipython().magic('R print(anova(model2, model3))')

get_ipython().magic('R model3 <- lmer(mathgain ~  mathkind  + minority + ses +(1|schoolid) + (1|classid), data=data, REML = F)')
get_ipython().magic('R print(summary(model3))')

model3 = sm.MixedLM.from_formula('mathgain ~ mathkind  + minority + ses*ses', groups="schoolid", vc_formula=vcf,  data=data)
result3 = model3.fit(reml=False, maxiter=10000, full_output=True)
result3.summary()

get_ipython().magic('R print(anova(model2,model3))')

result2.compare_lr_test(result3)

model4 = sm.MixedLM.from_formula('mathgain ~ mathkind  + minority + ses*ses', groups="schoolid", vc_formula=vcf,  data=data)

#REML[Should raise warning]
result4 = model4.fit(maxiter=10000)
result2.compare_lr_test(result4)

