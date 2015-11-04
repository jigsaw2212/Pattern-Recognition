import pandas
import numpy as np 
import csv as csv

titanic = pandas.read_csv("train.csv")

gender = np.array(titanic["Sex"])
gender = gender.astype('unicode')

total =  len(gender) #prints the total number of passenger

sur = np.array(titanic["Survived"])
sur = sur.astype(np.int)


#n1 - survived
#n2 - not survived
n1 = 0
n2 = 0

for item in sur:
	if item==0:
		n2 = n2 + 1
	else:
		n1 = n1 + 1


#prior probabilities
pS = float(n1)/total
pNS = float(n2)/total


MgS = 0
MgNS = 0

FMgS = 0
FMgNS = 0


for i in range(0,len(gender)):
	if gender[i]=='male': 
		if sur[i]==1:
			MgS = MgS + 1
		else:
			MgNS = MgNS + 1
	elif gender[i] =='female':
		if sur[i]==0:
			FMgNS = FMgNS + 1
		else:
			FMgS = FMgS + 1


#Likelihoods
pMgS = float(MgS) / n1
pFMgS = float(FMgS)/ n1

pMgNS = float(MgNS) / n2
pFMgNS = float(FMgNS)/ n2 

print
print "The likelihoods are as follows:-"
print "pMgS",pMgS
print "pMgNS", pMgNS

print "pFMgS",pFMgS
print "pFMgNS", pFMgNS
print

#Discriminant Function, gi(x) = ln(likelihood) + ln(prior)

query = raw_input("Enter gender (male/female): ")

if query=='male':
	g1 = np.log(pMgS) + np.log(pS)
	g2 = np.log(pMgNS) + np.log(pNS)
else:
	g1 = np.log(pFMgS) + np.log(pS)
	g2 = np.log(pFMgNS) + np.log(pNS)

G = g1-g2

print "The Discriminant functions are:-"
print "g1: ", g1
print "g2: ", g2
print

if G > 0:
	print "Result: Survived"
else:
	print "Result: Not Survived"

