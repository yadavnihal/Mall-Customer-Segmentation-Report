# Importing the Data
# import pandas as pd
# Data=pd.read_csv("Mall_Customers.csv")
# print(Data)


# ######################################
# Data cleaning parts starts from here #
# ######################################
import pandas as pd

Data=pd.read_csv("Mall_Customers.csv")
# print(Data)
# renaming the Genre to Gender
Data.rename(columns={"Genre":"Gender"},inplace=True)
# print(Data)
# checking the missing values
missing_values=Data.isnull().sum()
# print(missing_values)
# hence the data have not any missing values
# then we can go to the next move

# now we are handling the outliers

import matplotlib.pyplot as plt
plt.figure(figsize=(14,7))
# plotting a box plot for the Age and Annual Income (k$) coloumn
plt.subplot(1,2,1)
plt.boxplot(Data["Age"])
plt.title("Age Outliers using boxplot")

plt.subplot(1,2,2)
plt.boxplot(Data["Annual Income (k$)"])
plt.title("Income Outliers using boxplot")
plt.show()
# by the help of boxplot we can see there is an outlier in Income which is above significantly from the other observationns
# Now we check the outliers with the IQR method
for col in ["Age","Annual Income (k$)"]:
    # Calculate IQR
    Q1 = Data[col].quantile(0.25)
    Q3 = Data[col].quantile(0.75)
    IQR = Q3 - Q1
    IQR=Q3-Q1
    lower_bound=Q1-(1.5*IQR)
    upper_bound=Q3+(1.5*IQR)
    outliers = ((Data[col] < lower_bound) | (Data[col] > upper_bound))
# print(Data[col][outliers])

# now capping the outliers using IQR method
for col in ["Age","Annual Income (k$)"]:
    # Calculate IQR
    Q1 = Data[col].quantile(0.25)
    Q3 = Data[col].quantile(0.75)
    IQR = Q3 - Q1
    IQR=Q3-Q1
    lower_bound=Q1-(1.5*IQR)
    upper_bound=Q3+(1.5*IQR)
    Data[col]=Data[col].apply(lambda x: upper_bound if(x>upper_bound) else(lower_bound if(x<lower_bound) else x) )
# print(Data)

# we handled the outliers by capping with IQR method
# now we are removing the decimal points from the Income column
Data["Annual Income (k$)"]=Data["Annual Income (k$)"].astype(int)
# print(Data["Annual Income (k$)"])

################################
# Data cleaning part ends here #
################################

##############################################
#Exploratory Data Analysis (EDA) starts here #
##############################################

# Summary
Summary=Data.describe()
# print(Summary)

#visualization using Matplotlib
# ploting a bar_graph of income_by_gender
income_by_gender=Data.groupby("Gender")["Annual Income (k$)"].mean()
# reindexing as Male,Female
income_by_gender=income_by_gender.reindex(["Male","Female"])

income_by_gender.plot(kind="bar",color=["green","pink"])
plt.xlabel("Gender")
plt.ylabel("Average Income")
plt.title("Average Income by Gender")
plt.show()

# creating a histogram of Spending by Age below 40 and above 40
Data_below_40=Data[Data["Age"]<40]
Data_above_40=Data[Data["Age"]>=40]
plt.figure(figsize=(12, 6))
# Histogram for ages below 40
plt.subplot(1, 2, 1)
plt.hist(Data_below_40['Spending Score (1-100)'], bins=5, color='blue', alpha=0.7)
plt.title('Spending Score of people having Ages Below 40')
plt.xlabel('Spending Score')
plt.ylabel('Frequency')
# Histogram for ages 40 and above
plt.subplot(1, 2, 2)
plt.hist(Data_above_40['Spending Score (1-100)'], bins=5, color='green')
plt.title('Spending Score of people having Ages 40 and Above')
plt.xlabel('Spending Score')
plt.ylabel('Frequency')
plt.show()

# Insights from the visualization
#--------------------------------------
# 1)The probability of customers who are below the age of 40 and have a spending score between 60 and 80 is high.
# 2)The probability of customers who are above the age of 40 and have a spending score between 40 and 60 is high.
# 3)The Males have high income(mean) than the Female so, we can give more recommendations to the Male customers.


############################################
#Exploratory Data Analysis (EDA) ends here #
############################################

# Making a copy of cleaned data for visualization in powerbi
Data.to_csv("Data_cleaned.csv",index=False)

#####################################
# Customer Segmentation starts here #
#####################################

### Feature Selection ###
#Variance Thresholding
# removing the columns which have less variance
df=Data.copy()
# changing male to 0 and female to 1
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
from sklearn.feature_selection import VarianceThreshold
selector=VarianceThreshold(0.1)
Data_Vfiltered=pd.DataFrame(selector.fit_transform(df),columns=df.columns[selector.get_support()])
if(Data_Vfiltered.shape==df.shape):
    print("No columns having less variance")
else:
    print("the data have less variance features")

# the features we are selecting are All features except the customer_id because the feature don't give any valuable insights for the improvement

### K-Means Clustering ###
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#selecting features for the clustering
features=Data[["Age","Annual Income (k$)",'Spending Score (1-100)']]

#standarizing the features
scaler=StandardScaler()
scaled_features=scaler.fit_transform(features)
# print(scaled_features)

# Elbow method to find optimal k
inertia=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
# plot the elbow curve
plt.figure(figsize=(10,5))
plt.plot(range(1,11),inertia,marker="o")
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
# plt.xticks(range(1, 11))
plt.show()

# Based on the elbow plot, choose the optimal k ( k=5)
optimal_k = 5  # Adjust this based on your elbow plot

###################################
# Customer Segmentation ends here #
###################################

#############################
# Visualization Starts here #
#############################

# Apply k-means with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
Data["cluster"] = kmeans.fit_predict(scaled_features)

# Visualize the clusters (2D plots)
import seaborn as sns

plt.figure(figsize=(10,5))
sns.scatterplot(data=Data, x=Data["Age"], y=Data["Annual Income (k$)"], hue='cluster', palette='viridis')
plt.title("Cluster based on Age and Annual Income")
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(data=Data, x=Data["Age"], y=Data["Spending Score (1-100)"], hue='cluster', palette='viridis')
plt.title("Cluster based on Age and Spending Score")
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(data=Data, x=Data["Annual Income (k$)"], y=Data["Spending Score (1-100)"], hue='cluster', palette='viridis')
plt.title("Cluster based on Annual Income and Spending Score")
plt.show()

###########################
# Visualization ends here #
###########################
