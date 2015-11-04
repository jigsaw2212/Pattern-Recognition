import pandas
import numpy as np 
import csv as csv
import pylab as pl

titanic = pandas.read_csv("train.csv")

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
age = np.array(titanic["Age"])
age = age.astype(np.int)

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1 
gender = np.array(titanic["Sex"])
gender = gender.astype(np.int)
#print gender

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

'''
C1 = Convariance of age_survived, gender_survived
C2 = Convariance of age_Notsurvived, gender_Notsurvived

Similarly for the mean
'''

age_survived = age[sur==1]
gender_survived = gender[sur==1]

age_Notsurvived = age[sur==0]
gender_Notsurvived = gender[sur==0]

data = np.vstack((age_survived,gender_survived))
C1 = np.array(np.cov(data))

data = np.vstack((age_Notsurvived,gender_Notsurvived))
C2 = np.array(np.cov(data))

'''
print 
print C1
print
print C2
'''

M1 = np.array([np.mean(age_survived), np.mean(gender_survived)])
M2 = np.array([np.mean(age_Notsurvived), np.mean(gender_Notsurvived)])

'''
print
print M1
print 
print M2
'''

D=2 #No of dimensions

input_age = input("Enter age: ")
input_sex = input("Enter sex (male->0, female->1): ")

'''
Using the formula, gi(x) = -1/2*(x-u).T*inverse(cov)*(x-u) - d/2*ln(2*pi) -1/2*ln(cov) + ln(P(W))
'''

points = [input_age, input_sex]

invC = np.linalg.inv(C1)
v = points - M1
g1 = -0.5*np.sum(np.dot(v, invC) * v) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C1)) + np.log(pS)


invC = np.linalg.inv(C2)
v = points - M2
g2 = -0.5*np.sum(np.dot(v, invC) * v) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C2)) + np.log(pNS)

print
print "Discriminant Function1, g1(survived): ", g1
print
print "Discriminant Function2, g2(not survived) :", g2
print

if(g1 > g2):
	print "Result: Survived"
else:
	print "Result: Not Survived"
