import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import os
from scipy.stats import pearsonr
from setuptools.sandbox import save_path
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

''' Load data '''


current_dir = os.getcwd()
stock_file_name = "tesla_stock_new.xlsx"
interest_file_name = "tesla_interest.xlsx"
stock_file_path = os.path.join(current_dir, stock_file_name)
interest_file_path = os.path.join(current_dir, interest_file_name)
stock_df = pd.read_excel(stock_file_path)
interest_df = pd.read_excel(interest_file_path)

pd.set_option('display.max_columns', None)

''' Data Preprocessing '''

# Transfer string data Date to DateTime
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
interest_df['Date'] = pd.to_datetime(interest_df['Date'])

# Merge two dataframes
df = pd.merge(stock_df, interest_df, on='Date', how='left')

# Get US holiday
us_holidays = holidays.US()

# Add a feature: whether trading day
df['trading_day'] = df['Date'].apply(lambda x: 'No' if x in us_holidays or x.weekday() >= 5 else 'Yes')

# Convert category data to numerical data
df['converted_trading_day'] = df['trading_day'].apply(lambda x: 1 if x == 'Yes' else 0)
df.set_index('Date', inplace=True)

# Data cleaning
df_trading = df[df['converted_trading_day'] != 0]
df_trading.loc[df_trading['Opening price'] == 0, 'converted_trading_day'] = 0
df_trading = df_trading[df_trading['converted_trading_day'] != 0]
df_trading = df_trading[df_trading['Opening price'] != 0]
print(df_trading)

''' Data preprocessing: deal with stock splits '''

stock_split_index = df_trading.index.get_loc('2022-08-25')
opening_price_index = df_trading.columns.get_loc('Opening price')
lowest_price_index = df_trading.columns.get_loc('Lowest price')+1
# Preprocess Opening price, Closing price, Highest price, Lowest price
df_trading.iloc[stock_split_index:, opening_price_index:lowest_price_index] = df_trading.iloc[stock_split_index:, opening_price_index:lowest_price_index] * 3

# Preprocess Percentage change
percentage_change = (
    (df_trading['Closing price'].iloc[stock_split_index:] - df_trading['Closing price'].shift(1).iloc[stock_split_index:])
    / df_trading['Closing price'].shift(1).iloc[stock_split_index:]
) * 100
df_trading.iloc[stock_split_index:, df_trading.columns.get_loc('Percentage change')] = percentage_change.values

# Preprocess MA10
ma10_values = df_trading['Closing price'].iloc[400:].rolling(window=10).mean().iloc[10:]
df_trading.iloc[410:, df_trading.columns.get_loc('MA10')] = ma10_values.values

#Preprocess RSI
rsi_period = 14
delta = df_trading['Closing price'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
rs = gain / loss
new_rsi = 100 - (100 / (1 + rs))
df_trading.loc[df_trading.index[410]:, 'RSI'] = new_rsi.iloc[410:]

# Preprocess BOLL
boll_period = 20
df_trading['Middle Band'] = df_trading['Closing price'].rolling(window=boll_period).mean()
df_trading['StdDev'] = df_trading['Closing price'].rolling(window=boll_period).std()
new_upper_band = df_trading['Middle Band'] + 2 * df_trading['StdDev']
new_lower_band = df_trading['Middle Band'] - 2 * df_trading['StdDev']
df_trading.loc[df_trading.index[410]:, 'Upper Band'] = new_upper_band.iloc[410:]
df_trading.loc[df_trading.index[410]:, 'Middle Band'] = df_trading['Middle Band'].iloc[410:]
df_trading.loc[df_trading.index[410]:, 'Lower Band'] = new_lower_band.iloc[410:]
df_trading.drop(columns=['StdDev'], inplace=True)

# Save numerical data
df_preprocessed = df_trading[['Opening price', 'Closing price', 'Highest price', 'Lowest price', 'Volum (10,000)', 'Percentage change', 'MA10', 'KDJ', 'RSI', 'BOLL', 'Public interest']]
df_preprocessed_file_path = os.path.join(current_dir, 'df_preprocessed.xlsx')
df_preprocessed.to_excel(df_preprocessed_file_path, index=True)

''' EDA '''

# Show data
print('Data after preprocessing:\n')
print(df_preprocessed)
print('\n')

''' Visualization '''

# Line charts
plt.plot(df_preprocessed.index, df_preprocessed['Opening price'], color='g')
plt.title('Line charts of Opening price')
opening_file_path = os.path.join(current_dir, 'Line charts of Opening price.png')
plt.savefig(opening_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['Closing price'], color='g')
plt.title('Line charts of Closing price')
closing_file_path = os.path.join(current_dir, 'Line charts of Closing price.png')
plt.savefig(closing_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['Highest price'], color='g')
plt.title('Line charts of Highest price')
highest_file_path = os.path.join(current_dir, 'Line charts of Highest price.png')
plt.savefig(highest_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['Lowest price'], color='g')
plt.title('Line charts of Lowest price')
lowest_file_path = os.path.join(current_dir, 'Line charts of Lowest price.png')
plt.savefig(lowest_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['Volum (10,000)'], color='g')
plt.title('Line charts of Volum (10,000)')
volum_file_path = os.path.join(current_dir, 'Line charts of Volum (10,000).png')
plt.savefig(volum_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['Percentage change'], color='g')
plt.title('Line charts of Percentage change')
percentage_change_file_path = os.path.join(current_dir, 'Line charts of Percentage change.png')
plt.savefig(percentage_change_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['MA10'], color='g')
plt.title('Line charts of MA10')
MA10_file_path = os.path.join(current_dir, 'Line charts of MA10.png')
plt.savefig(MA10_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['KDJ'], color='g')
plt.title('Line charts of KDJ')
KDJ_file_path = os.path.join(current_dir, 'Line charts of KDJ.png')
plt.savefig(KDJ_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['RSI'], color='g')
plt.title('Line charts of RSI')
RSI_file_path = os.path.join(current_dir, 'Line charts of RSI.png')
plt.savefig(RSI_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['BOLL'], color='g')
plt.title('Line charts of BOLL')
BOLL_file_path = os.path.join(current_dir, 'Line charts of BOLL.png')
plt.savefig(BOLL_file_path)
plt.show()
plt.plot(df_preprocessed.index, df_preprocessed['Public interest'], color='g')
plt.title('Line charts of Public interest')
interest_file_path = os.path.join(current_dir, 'Line charts of Public interest.png')
plt.savefig(interest_file_path )
plt.show()

''' Scatter plot '''

# Set a dataframe 12 numerical variables

# Create pairwise scatter plots
pairplot = sns.pairplot(df_preprocessed, plot_kws={'s': 2, 'alpha': 0.3})
# Set title and adjust layout
pairplot.fig.suptitle('Pairwise Scatter Plots of 12 Variables', y=1.02, fontsize=16)
# Save the figure and display it
pair_scatter_file_path = os.path.join(current_dir, 'Pairwise Scatter Plots of 12 Variables.png')
pairplot.fig.savefig(pair_scatter_file_path, bbox_inches='tight')  # Use bbox_inches='tight' to ensure edges are not clipped
plt.show()

''' Correlation analysis: Pearson correlation analysis '''

# Calculate Pearson correlation matrix
pearson_corr_matrix = df_preprocessed.corr(method='pearson')
print('Pearson correlation matrix of 11 variables:\n')
print(pearson_corr_matrix)
print('\n')
# Set figure size and plot the heatmap
plt.figure(figsize=(12, 10))  # Increase figure size
sns.heatmap(pearson_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
# Set title and rotate axis labels
plt.title('Pearson Correlation Matrix Heatmap of 11 Variables', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels 45 degrees and align right
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
# Save the figure and display it
pearson_file_path = os.path.join(current_dir, 'Pearson Correlation Matrix Heatmap of 11 Variables.png')
plt.savefig(pearson_file_path, bbox_inches='tight')  # Use bbox_inches='tight' to prevent edge clipping
plt.show()


''' Correlation analysis: Spearman correlation analysis '''

# Calculate Spearman correlation matrix
spearman_corr_matrix = df_preprocessed.corr(method='spearman')
print('Spearman correlation matrix of 11 variables:\n')
print(spearman_corr_matrix)
print('\n')
# Set figure size and plot the heatmap
plt.figure(figsize=(12, 10))  # Increase figure size
sns.heatmap(spearman_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
# Set title and rotate axis labels
plt.title('Spearman Correlation Matrix Heatmap of 11 Variables', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels 45 degrees and align right
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
# Save the figure and display it
spearman_file_path = os.path.join(current_dir, 'Spearman Correlation Matrix Heatmap of 11 Variables.png')
plt.savefig(spearman_file_path, bbox_inches='tight')  # Use bbox_inches='tight' to prevent edge clipping
plt.show()


''' PCA and Factor analysis '''

# 1. Separate 'Closing price' and standardize the remaining features
closing_price = df_preprocessed['Closing price']  # Separate the target variable
X = df_preprocessed.drop('Closing price', axis=1)  # Remaining features as input variables

# 2. Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize the feature data

# 3. Perform PCA for dimensionality reduction
n_components_pca = 3  # Set the number of principal components (based on cumulative explained variance ratio)
pca = PCA(n_components=n_components_pca)
X_pca = pca.fit_transform(X_scaled)

# 4. Print the explained variance ratio of each principal component
print("Explanation Variance Ratio of Each Principal Component in PCA:\n")
print(pca.explained_variance_ratio_)
print('\n')

# 5. Print Eigenvalues of each principal component in PCA
eigenvalues = pca.explained_variance_
print("Eigenvalues of Each Principal Component:\n", eigenvalues, '\n')

# 6. Print Principal Component Loadings Matrix
# Each element represents the loading of a feature on the corresponding principal component
pca_loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f'PC{i+1}' for i in range(n_components_pca)])
print("Principal Component Loadings Matrix:\n")
print(pca_loadings, '\n')

# 7. Get the explained variance ratio of each principal component in PCA
explained_variance = pca.explained_variance_ratio_

# 8. Calculate the cumulative variance ratio (optional)
cumulative_variance = np.cumsum(explained_variance)

# 9. Plot the Scree Plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b', label='Explained Variance')
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='r', label='Cumulative Variance')
plt.title('Scree Plot with Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained Ratio')
plt.xticks(np.arange(1, len(explained_variance) + 1))  # Set x-axis ticks
plt.grid(True)
plt.legend()
scree_plot_file_path = os.path.join(current_dir, 'Scree Plot with Cumulative Explained Variance.png')
plt.savefig(scree_plot_file_path)
plt.show()

# 10. Kaiser-Meyer-Olkin (KMO) test
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all, kmo_model = calculate_kmo(df_preprocessed)
print("KMO values for each variable:\n", kmo_all, '\n')
print("Overall KMO score:\n", kmo_model, '\n')

# 11. Perform Factor Analysis on the principal components extracted from PCA
n_factors = 3  # Extract 3 factors
fa = FactorAnalysis(n_components=n_factors)
X_fa = fa.fit_transform(X_pca)

# 12. Factor Loadings
factor_loadings = pd.DataFrame(fa.components_.T, index=[f'PC{i+1}' for i in range(n_components_pca)],
                               columns=[f'Factor{i+1}' for i in range(n_factors)])
print("Factor Loadings:\n")
print(factor_loadings)
print('\n')

# 13. 3D Scatter Plot of Factors
factor_data = pd.DataFrame(X_fa, columns=['Factor1', 'Factor2', 'Factor3'])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(factor_data['Factor1'], factor_data['Factor2'], factor_data['Factor3'], c='blue', marker='o')
ax.set_xlabel('Factor1')
ax.set_ylabel('Factor2')
ax.set_zlabel('Factor3')
plt.title('3D Scatter Plot of Factors')
scatter_factors_file_path = os.path.join(current_dir, '3D Scatter Plot of Factors.png')
plt.savefig(scatter_factors_file_path)
plt.show()

# 14. Factors pair plot
sns.pairplot(factor_data, kind='scatter', diag_kind='kde', plot_kws={'s': 2, 'alpha': 0.3})
plt.suptitle('Pair Plot of Factors', y=1.02)
pair_plot_factors_file_path = os.path.join(current_dir, 'Pair Plot of Factors.png')
plt.savefig(pair_plot_factors_file_path)
plt.show()

# 15. Find best feature for PC1, PC2, PC3
max_index_pc1 = pca_loadings['PC1'].idxmax()
second_max_index_pc1 = pca_loadings['PC1'].nlargest(2).index[-1]
max_index_pc2 = pca_loadings['PC2'].idxmax()
max_index_pc3 = pca_loadings['PC3'].idxmax()
print('The best feature for Principle Component 1 (PC1):', max_index_pc1, '\n')
print('The second best feature for Principle Component 1 (PC1):', second_max_index_pc1, '\n')
print('The best feature for Principle Component 2 (PC2):', max_index_pc2, '\n')
print('The best feature for Principle Component 3 (PC3):', max_index_pc3, '\n')