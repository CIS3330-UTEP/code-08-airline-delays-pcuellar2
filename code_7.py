import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
#If any of this libraries is missing from your computer. Please install them using pip.

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename, low_memory=False)

#ARR_DELAY is the column name that will be used as dependent variable (Y)

#DESCRIPTIVE STATISTICS

#prints descriptive statistics on arrival delays, grouping by airline name to see data
print("Descriptive statistics on arrival delays, grouping by airline name")
print (df.groupby(['OP_CARRIER_NAME'])['ARR_DELAY'].describe())

#Potential independent variables: DEP_DELAY, DISTANCE,

#prints descriptive statistics on departure delays, grouping by airline name to determine predictor variable
print("Descriptive statistics on departure delays, grouping by airline name")
print (df.groupby(['OP_CARRIER_NAME'])['DEP_DELAY'].describe())

#prints descriptive statistics on distance, grouping by airline name to determine predictor variable
print("Descriptive statistics on distance traveled, grouping by airline name")
print (df.groupby(['OP_CARRIER_NAME'])['DISTANCE'].describe())

#VISUAL REPRESENTATION
filter_query= "OP_CARRIER_NAME == 'Delta Air Lines Inc.' or OP_CARRIER_NAME == 'American Airlines Inc.' or OP_CARRIER_NAME == 'United Air Lines Inc.'"
smaller_df= df.query(filter_query)
smaller_df.boxplot(column='ARR_DELAY', by='OP_CARRIER_NAME')
plt.show()


#VISUAL REPRESENTATION - Scatter Plots between ARR_DELAY and DEP_DELAY
df.plot.scatter(x='DEP_DELAY', y='ARR_DELAY')
plt.show()


#PREDICTIVE STATISTICS
#Run OLS
#define predictor and dependent variables
y = df['ARR_DELAY']
x = df['DEP_DELAY']
#add constant to predictor variables
x = sm.add_constant(x)
#fit linear regression model
model = sm.OLS(y, x).fit()
#view model summary
print(model.summary())

