import pandas
import numpy as np 
import csv as csv

titanic = pandas.read_csv("train.csv")

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

age = np.array(titanic["Age"])
age = age.astype(np.int)

total =  len(age) #prints the total number of passenger

#Numpy array of labels
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

query = input("Enter age: ")
print

#likelihoods
XgS = 0
XgNS = 0

for i in range(0,len(sur)):
	if age[i]==query & sur[i]==1:
		XgS = XgS + 1
	elif age[i]==query & sur[i]==0:
		XgNS = XgNS + 1


pXgS = float(XgS)/n1
pXgNS = float(XgNS)/n2


print "Likelihood pXgS:",pXgS
print "Likelihood pXgNS:", pXgNS
print
evidence = pXgNS*pNS + pXgS*pS


pSgX = float(pXgS*pS)/evidence
pNSgX = float(pXgNS*pNS)/evidence

print "The posterior probabilities are: ", pSgX, pNSgX


