import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

''' Bulid LSTM model '''

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))  # Output for 1-day prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

''' Build evaluation function '''

def evaluate_model(y_true, y_pred):
    # Calculate evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAPE, avoiding division by zero
    y_true_no_zero = y_true.copy()
    y_true_no_zero[y_true_no_zero == 0] = 1e-10
    mape = (np.abs((y_true_no_zero - y_pred) / y_true_no_zero).mean()) * 100

    # Print evaluation results
    output = (
        "LSTM model evaluation values:\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Coefficient of Determination (R²): {r2:.4f}"
    )

    print(output)

''' Load data '''

current_dir = os.getcwd()
stock_file_path = os.path.join(current_dir, 'df_preprocessed.xlsx')
StockData = pd.read_excel(stock_file_path, index_col=0)
data = StockData['Closing price']
data = data.to_frame()
# print(data.shape) #=> (933, 1)
print('Stock data DataFrame head:\n\n', data.head(), '\n\n')
print('Stock data DataFrame tail:\n\n', data.tail())

''' Normalization '''

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.iloc[:, -1].values.reshape(-1, 1))
# print(scaled_data.shape) #=> (933, 1)
# print(type(scaled_data)) #=> <class 'numpy.ndarray'>

''' Set Train set (80%) and Validation Set (20%) '''

timestep = 50
X, y = [], []

# Modification: Single-step prediction
for i in range(len(scaled_data) - timestep):
    X.append(scaled_data[i:(i + timestep), -1])
    y.append(scaled_data[i + timestep, -1])
    end_time = i

# print(end_time) #=> 882

# Convert to NumPy arrays
X, y = np.array(X), np.array(y)

# Reshape X to ensure it's 3D
if len(X.shape) == 2:
    X = X.reshape((X.shape[0], timestep, 1))

# Check shapes
# print(X.shape) #=> (883, 50, 1)
# print(y.shape) #=> (883,)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Check train and test shapes
# print(X_train.shape) #=> (706, 50, 1)
# print(X_test.shape) #=> (177, 50, 1)
# print(y_train.shape) #=> (706,)
# print(y_test.shape) #=> (177,)

''' Record the time points '''

future_days = 15 # the number of days we plan to predict using LSTM model

train_start_date_index = 0
train_start_date = data.iloc[train_start_date_index].name
# print(train_start_date) #=> 2021-01-04 00:00:00

train_end_date_index = X_train.shape[0]+timestep-1
train_end_date = data.iloc[train_end_date_index].name
# print(train_end_date) #=> 2024-01-17 00:00:00

test_start_date_index = X_train.shape[0]+timestep
test_start_date = data.iloc[test_start_date_index].name
# print(test_start_date) #=> 2024-01-18 00:00:00

test_end_date_index = end_time+timestep
test_end_date = data.iloc[test_end_date_index].name
# print(test_end_date) #=> 2024-09-30 00:00:00

train_dates = data.index[:test_start_date_index]
test_dates = data.index[test_start_date_index: test_end_date_index+1]
predict_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')


''' Train the LSTM model for single-step prediction '''

stock_price_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
stock_price_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Perform predictions on training and testing sets
train_predict = stock_price_model.predict(X_train)
test_predict = stock_price_model.predict(X_test)

# print('train_predict.shape:', train_predict.shape)  #=> (706, 1)
# print('test_predict.shape:', test_predict.shape)    #=> (177, 1)


''' Denormalize and convert to a one-dimensional array '''

test_predict_original_scale = scaler.inverse_transform(test_predict).flatten()


''' Convert prediction results to a Pandas Series and set dates index '''

# Convert prediction results to a Pandas Series and set the date index
test_predict_series = pd.Series(test_predict_original_scale.flatten(), index=test_dates)

# Check the shape of the output
print(test_predict_series.head(3))
print(test_predict_series.tail(3))


''' Evaluate the LSTM model '''

evaluate_model(y_test, test_predict.flatten())

''' Recursive Multi-step Closing Price Prediction '''


def iterative_forecast(model, last_window, num_days, scaler):
    """
    Generate multiple future time-step predictions using iterative forecasting.

    Parameters：
        - model: Trained LSTM model.
        - last_window: The last window of data required by the model as an initial input for prediction.
        - num_days: Number of future days to predict.
        - scaler: MinMaxScaler instance used to inverse scale the predictions.

    Return：
        - predictions: List of future predicted prices.
    """

    predictions = []
    # Reshape the last input window to match model input format (1, window_size, 1)
    current_input = last_window.reshape(1, last_window.shape[0], 1)
    print(current_input.shape)

    for _ in range(num_days):
        # Predict the next time step
        next_pred = model.predict(current_input)

        # Inverse scale the prediction and add to the list
        next_pred_rescaled = scaler.inverse_transform(next_pred)
        predictions.append(next_pred_rescaled.flatten()[0])

        # Reshape next_pred to 3D to match the dimensions of current_input
        next_pred_reshaped = np.reshape(next_pred, (1, 1, 1))

        # Use concatenate to append the new prediction to the end of current_input's last part
        current_input = np.concatenate((current_input[:, 1:, :], next_pred_reshaped), axis=1)

    return predictions


''' Preditc the future stock price '''

# Retrieve the last input window
print(X_test[-1].shape)
last_window = scaled_data[-timestep:, -1]
print(last_window.shape)

# Generate future predictions
future_predictions = iterative_forecast(stock_price_model, last_window, future_days, scaler)

# Convert predictions to a Pandas Series and set the index to predict dates
future_predictions_series = pd.Series(future_predictions, index=predict_dates)

print(future_predictions_series)

''' Plot the comparison charts '''

plt.figure(figsize=(12, 6))

plt.plot(train_dates, data.iloc[train_start_date_index: train_end_date_index + 1, :], label="Train Set Closing Price",
         color="blue", linestyle="-")
plt.plot(test_predict_series.index, data.iloc[test_start_date_index: test_end_date_index + 1, :],
         label="Test Set Closing Price", color="green", linestyle="-", alpha=0.5)
plt.plot(test_predict_series.index, test_predict_series.values, label="Test Set LSTM Predicted Closing Price",
         color="red", linestyle="--", alpha=0.5)
plt.plot(future_predictions_series.index, future_predictions_series.values, label="LSTM Predicted Future Closing Price",
         color="orange", linestyle="-")

# Set labels and title
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title('Comparison chart of original closing prices, test closing prices, and predicted closing prices.')
plt.legend()
plt.grid(True)

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=10, maxticks=15))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.tight_layout()

comparison_file_path = os.path.join(current_dir, 'Comparison chart of original closing prices, test closing prices, and predicted closing prices.png')
plt.savefig(comparison_file_path)
plt.show()