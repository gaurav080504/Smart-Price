import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# 1. Load and Clean (Quick version of what we just did)
df = pd.read_csv('laptop_data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# 2. ENCODING: Turn text categories into numbers
encoder = LabelEncoder()
df['Company'] = encoder.fit_transform(df['Company'])
df['TypeName'] = encoder.fit_transform(df['TypeName'])

# 3. SPLITTING: Separate Features (X) from Target Price (y)
X = df[['Company', 'TypeName', 'Ram', 'Weight']]
y = df['Price']

# Split data into Training set (80%) and Testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAINING: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. EVALUATION: See how accurate we are
accuracy = model.score(X_test, y_test)
print(f"Model Training Complete!")
print(f"Model Accuracy (R^2 Score): {accuracy * 100:.2f}%")

# 6. PREDICTION: Let's test it with a fake laptop!
# Features: Company(Encoded), TypeName(Encoded), Ram(8), Weight(1.3)
test_laptop = [[1, 1, 8, 1.3]] 
predicted_price = model.predict(test_laptop)
print(f"\nPredicted Price for test laptop: {predicted_price[0]:.2f} INR")