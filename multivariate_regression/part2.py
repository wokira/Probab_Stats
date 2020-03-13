import math
import numpy as np
import pandas as pd
from itertools import combinations

def compute_coeff(x,y):
	# B = ((Xt * X) ^-1) *Xt * Y
    xt = np.transpose(x)
    prd1 = np.dot(xt,x)
    prd2 = np.dot(xt,y)
    prd1_inv = np.linalg.pinv(prd1)
    b = np.dot(prd1_inv,prd2)
    return b

def valueY(x,b):
	# Y = X * B
    return np.dot(x,b)

def error(actual, predicted):
    diff = math.pow((predicted - actual), 2)
    return diff

def map_list(headerList,valueList):
	hList = []
	for i in valueList:
		hList.append(headerList[i])
	return hList

myData = pd.read_csv("~C:/Users/Ira/Documents/Probablity-and-Statisctics-master/Probablity-and-Statisctics-master/multivariate_regression/inputFinal.csv")
headerList = list(myData)
column = [1,2,3,4,5,6,7,8,9]
comb = [] # generating all possible combinations
for i in range(1,10):
	tup = list(combinations(column, i))
	tupList = []
	for elt in tup:
		element = []
		for val in elt:
			element.append(val)
		tupList.append(element)
	comb.append(tupList)

for element in comb:
	for column_list in element:
		n = len(myData)
		m = len(column_list)
		totalSquareError = 0
		totalAbsError = 0
		for i in range(0, n):
			# print(column_list)
			a = myData.iloc[0:i,column_list].append(myData.iloc[i+1:n,column_list]).as_matrix() # leave one out cross validation 
			x = np.ones((len(a),m+1))
			x[:,1:m+1] = a
			y = myData.iloc[0:i,10].append(myData.iloc[i+1:n,10]).as_matrix() # last column has impact factor
			b = compute_coeff(x,y)
			Xin = np.ones((1,m+1))
			sampleX = myData.iloc[i,column_list].as_matrix()
			Xin[:,1:m+1] = sampleX
			actual = myData.iloc[i,10]
			predicted = valueY(Xin,b)[0]
			err_val = error(actual, predicted)
			totalAbsError += (actual - predicted) if (actual - predicted)>0 else -(actual - predicted)
			totalSquareError += err_val

		meanSquareError = (float) (totalSquareError/n)
		print('{}:{:.6f}:{:.6f}'.format(map_list(headerList, column_list), meanSquareError, totalAbsError/n))

