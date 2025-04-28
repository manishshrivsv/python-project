import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
# Load CSV
file_path = r"C:\Users\asus\OneDrive\Attachments\python project\Electric_Vehicle_Population_Data.csv"
df = pd.read_csv(file_path)
#
# Auto-detect columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

sns.set(style="whitegrid")

# 1️⃣ Scatter Plot
if len(numerical_cols) >= 2:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=numerical_cols[0], y=numerical_cols[1])
    plt.title(f"Scatter Plot: {numerical_cols[0]} vs {numerical_cols[1]}")
    plt.tight_layout()
    plt.show()

# 2️⃣ Area Plot
if 'Model Year' in df.columns:
    year_data = df['Model Year'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.fill_between(year_data.index, year_data.values, color="skyblue", alpha=0.4)
    plt.plot(year_data.index, year_data.values, color="Slateblue", alpha=0.6)
    plt.title("EV Registration Over Years")
    plt.xlabel("Year")
    plt.ylabel("Number of EVs")
    plt.tight_layout()
    plt.show()

# 3️⃣ Heatmap
if len(numerical_cols) >= 2:
    plt.figure(figsize=(10, 6))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# 4️⃣ Donut Chart
if 'Make' in df.columns:
    top_makes = df['Make'].value_counts().nlargest(5)
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(top_makes, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
    plt.title("Top 5 Vehicle Makes - Donut Chart")
    plt.legend(top_makes.index, title="Make", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.show()

# 5️⃣ Violin Plot
if len(numerical_cols) >= 1 and 'Electric Vehicle Type' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x='Electric Vehicle Type', y=numerical_cols[0])
    plt.title(f"Violin Plot of {numerical_cols[0]} by Electric Vehicle Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 6️⃣ Countplot with hue
if 'Electric Vehicle Type' in df.columns and 'Make' in df.columns:
    top_5_makes = df['Make'].value_counts().nlargest(5).index
    filtered_df = df[df['Make'].isin(top_5_makes)]
    plt.figure(figsize=(10, 5))
    sns.countplot(data=filtered_df, x='Make', hue='Electric Vehicle Type')
    plt.title("Countplot with Hue (Make vs EV Type)")
    plt.tight_layout()
    plt.show()

# 7️⃣ Pairplot
if len(numerical_cols) >= 2:
    sns.pairplot(df[numerical_cols].dropna().sample(n=min(100, len(df))), diag_kind='kde')
    plt.suptitle("Pairplot of Numerical Features", y=1.02)
    plt.show()

# 8️⃣ Swarmplot
if len(numerical_cols) >= 1 and 'Electric Vehicle Type' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.swarmplot(data=df, x='Electric Vehicle Type', y=numerical_cols[0])
    plt.title(f"Swarmplot of {numerical_cols[0]}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 9️⃣ Stripplot
if len(numerical_cols) >= 1 and 'Electric Vehicle Type' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.stripplot(data=df, x='Electric Vehicle Type', y=numerical_cols[0], jitter=True)
    plt.title(f"Stripplot of {numerical_cols[0]}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 🔟 FINAL DASHBOARD (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
sns.histplot(df[numerical_cols[0]], kde=True, bins=30, ax=axes[0, 0])
axes[0, 0].set_title(f"Histogram of {numerical_cols[0]}")

# Pie Chart
if 'Make' in df.columns:
    top_makes = df['Make'].value_counts().nlargest(5)
    top_makes.plot.pie(autopct='%1.1f%%', ax=axes[0, 1], startangle=90)
    axes[0, 1].set_title("Top 5 Vehicle Makes - Pie Chart")
    axes[0, 1].set_ylabel("")

# Bar Chart
if 'Electric Vehicle Type' in df.columns:
    sns.countplot(data=df, x='Electric Vehicle Type', ax=axes[1, 0], order=df['Electric Vehicle Type'].value_counts().index)
    axes[1, 0].set_title("Bar Chart of Electric Vehicle Types")
    axes[1, 0].tick_params(axis='x', rotation=45)

# Boxplot
sns.boxplot(y=df[numerical_cols[1]], ax=axes[1, 1])
axes[1, 1].set_title(f"Boxplot of {numerical_cols[1]}")

plt.tight_layout()
plt.show()
