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

'''
print "pMgS",pMgS
print "pMgNS", pMgNS

print "pFMgS",pFMgS
print "pFMgNS", pFMgNS
'''

#For classifying a male
evidence = pMgS*pS + pMgNS*pNS
pSgM = pMgS*pS/(evidence)
pNSgM = pMgNS*pNS/(evidence)


#For classifying a female
evidence = pFMgS*pS + pFMgNS*pNS
pSgFM = pFMgS*pS/(evidence)
pNSgFM = pFMgNS*pNS/(evidence)

print 
#Posterior probablities
print "The Posterior probablities are:-"
print "pSgM",pSgM
print "pNSgM", pNSgM

print "pSgFM",pSgFM
print "pNSgFM", pNSgFM

#Losses incurred
#We use a zero-one loss function for min error rate classification
lambda11 = 0
lambda12 = 1
lambda21 = 1
lambda22 = 0

print
query = raw_input("Enter query (male/female): ")

if query=='male':
	Rs = lambda11*pSgM + lambda12*pNSgM
	Rns = lambda21*pSgM + lambda22*pNSgM
else:
	Rs = lambda11*pSgFM + lambda12*pNSgFM
	Rns = lambda21*pSgFM + lambda22*pNSgFM

print
print "The risks are as follows:-"
print "Risk(survived/X) = ", Rs 
print "Risk(Not survived/X) = ", Rns
print

if Rs < Rns:
	print "Result: Survived"
else:
	print "Result: Not Survived" 



