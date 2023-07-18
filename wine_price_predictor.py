import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv('red.csv')

# Extract the relevant features
features = ['Rating', 'Region', 'Winery', 'Year']
target = 'Price'
wine_data = data[features + [target]]

# Handle missing values (if any)
wine_data = wine_data.dropna()

# Convert categorical variables to numerical using one-hot encoding
wine_data = pd.get_dummies(wine_data)

# Split the data into training and testing sets
X = wine_data.drop(target, axis=1)
y = wine_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('Wine Price Predictions')
plt.show()
