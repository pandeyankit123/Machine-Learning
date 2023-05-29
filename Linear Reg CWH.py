#importing libraries 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# #loading dataset from scikit learn's inbuilt data repositories 
diabetes = datasets.load_diabetes()
#print(diabetes)

#print(diabetes.keys())
# #['data','target','frame','DESCR','feature_names','data_filename','target_filename','data_module']

#print(diabetes.data)                  #gives numpy arrays of the data
#print(diabetes.DESCR)                 #complete info about data
print(diabetes.target)                #numpy array of result coulumn
#print(diabetes.frame)
#print(diabetes.feature_names)         
#print(diabetes.data_filename)
#print(diabetes.target_filename)
#print(diabetes.data_module)


# #for single feature model
#print("For single feature model")
# #taking only 2nd feature
diabetes_X=diabetes.data[:, np.newaxis, 2]
#print(diabetes_x)        

diabetes_X_train = diabetes_X[:-30]  #taking 1st to last 30 for training 
diabetes_X_test = diabetes_X[-30:]   #taking last 30 for testing purpose
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

# #x-axis for feature(input)
# #y-axis for label(target/output)

model = linear_model.LinearRegression()           #making model(machine)
model.fit(diabetes_X_train, diabetes_Y_train)     #Giving data to machine
diabetes_Y_predicted = model.predict(diabetes_X_test) #obtaning result form machine/model

print("Mean squared error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))     #testing model by getting mse
print("Weights: ", model.coef_)    # m in LR(y=mx+c) and Wo,W1,W2...in MR(tan0)
print("Intercept: ", model.intercept_)  #c in LR(y=mx+c)

plt.scatter(diabetes_X_test, diabetes_Y_test) 
plt.plot(diabetes_X_test, diabetes_Y_predicted)
plt.show()  

# #Mean squared error is:  3035.060115291269
# #Weights:  [941.43097333]
# #Intercept:  153.39713623331644


# # #for multiple feature
# diabetes_X=diabetes.data

# diabetes_X_train = diabetes_X[:-30]   
# diabetes_X_test = diabetes_X[-30:]
# diabetes_Y_train = diabetes.target[:-30]
# diabetes_Y_test = diabetes.target[-30:]

# model = linear_model.LinearRegression()       
# model.fit(diabetes_X_train, diabetes_Y_train)   
# diabetes_Y_predicted = model.predict(diabetes_X_test)

# print("Mean squared error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))   
# print("Weights: ", model.coef_)  
# print("Intercept: ", model.intercept_)

# # #Mean squared error is:  1826.4841712795046
# # #Weights:  [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067 458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
# # #Intercept:  153.05824267739402
