import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42) 
square_footage = np.random.randint(500, 3500, 100)  
prices = square_footage * 300 + np.random.normal(0, 20000, 100) 

data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Price': prices
})
X = data[['SquareFootage']]  
y = data['Price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted', linewidth=2)
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Linear Regression: House Prices')
plt.legend()
plt.show()
