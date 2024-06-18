import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import sys

# Redirect standard output to a file
sys.stdout = open('output.txt', 'w')


# Load the dataset
data = pd.read_csv('banking_data.csv')

# Handle missing values using mean imputation
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns

numeric_imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

# Convert necessary columns to appropriate data types
data['age'] = data['age'].astype(int)
data['balance'] = data['balance'].astype(int)

# Identify outliers using z-score method for 'age' and 'balance'
z_scores_age = np.abs((data['age'] - data['age'].mean()) / data['age'].std())
data = data[z_scores_age <= 3]

z_scores_balance = np.abs((data['balance'] - data['balance'].mean()) / data['balance'].std())
data = data[z_scores_balance <= 3]

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
mapping_dict = {}
for col in categorical_cols:
    data[col + '_encoded'] = label_encoder.fit_transform(data[col])
    mapping_dict[col] = dict(enumerate(label_encoder.classes_, 0))

# 1. Distribution of age
plt.figure(figsize=(8, 6))
sns.histplot(data['age'], kde=True, bins=20)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png', bbox_inches='tight')
plt.show()
print("What is the distribution of age among the clients?")
print(data['age'].describe())
print()

# 2. Job type distribution
plt.figure(figsize=(8, 6))
sns.countplot(y='job', data=data, order=data['job'].value_counts().index, orient='h')
plt.title('Job Type Distribution')
plt.xlabel('Count')
plt.ylabel('Job Type')
plt.tight_layout()
plt.savefig('job_distribution.png', bbox_inches='tight')
plt.show()
print("How does the job type vary among the clients?")
print(data['job'].value_counts())
print()

# 3. Marital status distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='marital', data=data)
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.savefig('marital_distribution.png', bbox_inches='tight')
plt.show()
print("What is the marital status distribution of the clients?")
print(data['marital'].value_counts())
print()

# 4. Education level distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='education', data=data)
plt.title('Education Level Distribution')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.savefig('education_distribution.png', bbox_inches='tight')
plt.show()
print("What is the level of education among the clients?")
print(data['education'].value_counts())
print()

# 5. Proportion of clients with credit in default
plt.figure(figsize=(8, 6))
sns.countplot(x='default', data=data)
plt.title('Credit Default Distribution')
plt.xlabel('Credit Default')
plt.ylabel('Count')
plt.savefig('default_distribution.png', bbox_inches='tight')
plt.show()
print("What proportion of clients have credit in default?")
print(data['default'].value_counts(normalize=True))
print()

# 6. Distribution of average yearly balance
plt.figure(figsize=(8, 6))
sns.histplot(data['balance'], kde=True)
plt.title('Distribution of Average Yearly Balance')
plt.xlabel('Balance (euros)')
plt.ylabel('Count')
plt.savefig('balance_distribution.png', bbox_inches='tight')
plt.show()
print("What is the distribution of average yearly balance among the clients?")
print(data['balance'].describe())
print()

# 7. Number of clients with housing loans
plt.figure(figsize=(8, 6))
sns.countplot(x='housing', data=data)
plt.title('Housing Loan Distribution')
plt.xlabel('Housing Loan')
plt.ylabel('Count')
plt.savefig('housing_distribution.png', bbox_inches='tight')
plt.show()
print("How many clients have housing loans?")
print(data['housing'].value_counts())
print()

# 8. Number of clients with personal loans
plt.figure(figsize=(8, 6))
sns.countplot(x='loan', data=data)
plt.title('Personal Loan Distribution')
plt.xlabel('Personal Loan')
plt.ylabel('Count')
plt.savefig('loan_distribution.png', bbox_inches='tight')
plt.show()
print("How many clients have personal loans?")
print(data['loan'].value_counts())
print()

# 9. Communication types used for contacting clients
plt.figure(figsize=(8, 6))
sns.countplot(x='contact', data=data)
plt.title('Contact Type Distribution')
plt.xlabel('Contact Type')
plt.ylabel('Count')
plt.savefig('contact_distribution.png', bbox_inches='tight')
plt.show()
print("What are the communication types used for contacting clients during the campaign?")
print(data['contact'].value_counts())
print()

# 10. Distribution of the last contact day of the month
plt.figure(figsize=(8, 6))
sns.histplot(data['day'], kde=True, bins=30)
plt.title('Last Contact Day Distribution')
plt.xlabel('Day')
plt.ylabel('Count')
plt.savefig('day_distribution.png', bbox_inches='tight')
plt.show()
print("What is the distribution of the last contact day of the month?")
print(data['day'].describe())
print()

# 11. Last contact month distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='month', data=data, order=data['month'].value_counts().index)
plt.title('Last Contact Month Distribution')
plt.xlabel('Month')
plt.ylabel('Count')
plt.savefig('month_distribution.png', bbox_inches='tight')
plt.show()
print("How does the last contact month vary among the clients?")
print(data['month'].value_counts())
print()

# 12. Distribution of the duration of the last contact
plt.figure(figsize=(8, 6))
sns.histplot(data['duration'], kde=True)
plt.title('Last Contact Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.savefig('duration_distribution.png', bbox_inches='tight')
plt.show()
print("What is the distribution of the duration of the last contact?")
print(data['duration'].describe())
print()

# 13. Number of contacts performed during the campaign
plt.figure(figsize=(8, 6))
sns.countplot(x='campaign', data=data)
plt.title('Campaign Contact Count Distribution')
plt.xlabel('Number of Contacts')
plt.ylabel('Count')
plt.savefig('campaign_distribution.png', bbox_inches='tight')
plt.show()
print("How many contacts were performed during the campaign for each client?")
print(data['campaign'].value_counts())
print()

# 14. Distribution of the number of days since the last contact
plt.figure(figsize=(8, 6))
sns.histplot(data[data['pdays'] != -1]['pdays'], kde=True)
plt.title('Days Passed Since Last Contact Distribution')
plt.xlabel('Days')
plt.ylabel('Count')
plt.savefig('pdays_distribution.png', bbox_inches='tight')
plt.show()
print("What is the distribution of the number of days passed since the client was last contacted from a previous campaign?")
print(data[data['pdays'] != -1]['pdays'].describe())
print()

# 15. Number of contacts performed before the current campaign
plt.figure(figsize=(8, 6))
sns.countplot(x='previous', data=data)
plt.title('Previous Contact Count Distribution')
plt.xlabel('Number of Contacts')
plt.ylabel('Count')
plt.savefig('previous_distribution.png', bbox_inches='tight')
plt.show()
print("How many contacts were performed before the current campaign for each client?")
print(data['previous'].value_counts())
print()

# 16. Outcomes of the previous marketing campaigns
plt.figure(figsize=(8, 6))
sns.countplot(x='poutcome', data=data)
plt.title('Previous Campaign Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.savefig('poutcome_distribution.png', bbox_inches='tight')
plt.show()
print("What were the outcomes of the previous marketing campaigns?")
print(data['poutcome'].value_counts())
print()

# 17. Distribution of clients who subscribed to a term deposit
plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=data)
plt.title('Term Deposit Subscription Distribution')
plt.xlabel('Subscription')
plt.ylabel('Count')
plt.savefig('subscription_distribution.png', bbox_inches='tight')
plt.show()
print("What is the distribution of clients who subscribed to a term deposit vs. those who did not?")
print(data['y'].value_counts())
print()

# 18. Correlation between attributes and term deposit subscription
numeric_cols = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(method='spearman'), annot=True, cmap='YlOrRd')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png', bbox_inches='tight')
plt.show()

# 19. Are there any correlations between different attributes and the likelihood of subscribing to a term deposit?
corr_matrix = numeric_cols.corr(method='spearman')
target_corr = corr_matrix['y_encoded'].drop('y_encoded')
print("Are there any correlations between different attributes and the likelihood of subscribing to a term deposit?")
print(target_corr.sort_values(ascending=False))


# Close the file at the end
sys.stdout.close()