import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 


data = pd.read_csv('C:/Users/wasay/Desktop/data.csv')


#count of graduate vs dropout 
"""
sns.histplot(data['Class'], bins=20, kde=True)  # Replace 'column_name' with your column
plt.title('Histogram of Column')
plt.show()
"""
'''
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Class', y='Age at enrollment', palette='Blues')

# Customize labels and title
plt.xlabel('Dropout Status', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.title('Distribution of Age by Dropout Status', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()



'''

column_name = 'Course'

# Create the count plot
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x=column_name, color = 'lightblue')

# Customize labels and title
plt.xticks(fontsize=8)
plt.xlabel('Student Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'Original Target Variables (Multi-class)', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()



sns.histplot(data['Course'], bins=17, kde=True)  # Replace 'column_name' with your column
plt.title('Histogram of Column')
plt.show()

sns.boxplot(x='Class', y='Curricular units 1st sem (grade)', data=data)
plt.title("Boxplot of Feature1 vs Class")
plt.show()

