# Step 1: Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Calling a dataset
# Use pandas to call a dataset
df = pd.read_csv('C:/Users/Hp/Desktop/python(datascience)/Python Project/Data Analysis Project (Student Result)/Student Data.csv')


# Step 3: Exploring the dataset
print("First 5 rows of the dataset:")
print(df.head())      #print(df) Display the entire DataFrame (It's a convenient way to quickly inspect the structure and content of the DataFrame.)
print("\nInformation about the data types of each column:")
print(df.info())


# Step 4: Data Analysis
# Exclude the 'Name' column before calculating statistics
numeric_columns = df.columns[1:]  # Assuming 'Name' is the first column and extracting all column names from the DataFrame
cv_students = df[numeric_columns].std(axis=1) / df[numeric_columns].mean(axis=1)
df['Coefficient of Variation'] = cv_students      #adds a new column called 'Coefficient of Variation' to the DataFrame

# Display the coefficient of variation for each student
print("\nCoefficient of Variation for Each Student:")
print(cv_students)

# Find the best student based on the lowest coefficient of variation
best_student = df.loc[df['Coefficient of Variation'].idxmin()]
                           #selecting column CV      selecting row where min value occur
                               
print("\nBest Student based on Coefficient of Variation:")
print(best_student)


# Step 5: Visualization
# Reshape the data for visualization
df_long = df.melt('Name', var_name='Subject', value_name='Score')
# pd.melt: This is a Pandas function used to reshape a DataFrame.

# Create a bar plot
plt.figure(figsize=(9, 5))
sns.barplot(x='Name', y='Score', hue='Subject', data=df_long)
plt.title('Student Scores in Different Subjects')
plt.xlabel('Students')
plt.ylabel('Scores')
plt.legend(title='Subject', bbox_to_anchor=(1, 1))
#The legend helps identify which color corresponds to each subject.
# bbox_to_anchor=(1, 1):adjusts the position of the legend to be outside the plot.
plt.show()
