
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --------------------------------
# App Title
# --------------------------------
st.title("📈 Advanced Forecasting Platform")
st.markdown("Upload your data or use sample datasets, choose a model, and run forecasts!")

# --------------------------------
# Sidebar: Data Upload and Sample Data
# --------------------------------
st.sidebar.header("Data Input")
sample_option = st.sidebar.selectbox("Choose Sample Data", ["Sales Data", "Stock Data", "Revenue Data"])

# Load sample data
if sample_option == "Sales Data":
    data = pd.DataFrame([
        ['2023-01', 1000, 5000, 1],
        ['2023-02', 1200, 5500, 1],
        ['2023-03', 1100, 5200, 2],
        ['2023-04', 1400, 6000, 2],
        ['2023-05', 1600, 6500, 2],
        ['2023-06', 1800, 7000, 3],
        ['2023-07', 2000, 7500, 3],
        ['2023-08', 2200, 8000, 3],
        ['2023-09', 1900, 7200, 4],
        ['2023-10', 1700, 6800, 4],
        ['2023-11', 1500, 6200, 4],
        ['2023-12', 1300, 5800, 1]
    ], columns=['date', 'sales', 'marketing', 'season'])
elif sample_option == "Stock Data":
    data = pd.DataFrame({
        'date': [f"2023-{str(i%12+1).zfill(2)}" for i in range(24)],
        'price': 100 + np.random.rand(24)*50 + np.sin(np.arange(24)*0.5)*20,
        'volume': 1000000 + np.random.rand(24)*500000,
        'volatility': 0.1 + np.random.rand(24)*0.3
    })
else:
    data = pd.DataFrame({
        'date': [f"2023-{str(i%12+1).zfill(2)}" for i in range(18)],
        'revenue': 50000 + np.arange(18)*2000 + np.random.rand(18)*10000,
        'customers': 500 + np.arange(18)*25 + np.random.rand(18)*100,
        'avgOrder': 100 + np.random.rand(18)*50
    })

# Upload file option
uploaded_file = st.sidebar.file_uploader("Or upload CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

st.write("### Current Dataset")
st.dataframe(data.head())

# --------------------------------
# Model Settings
# --------------------------------
st.sidebar.header("Model Settings")
dependent_var = st.sidebar.selectbox("Dependent Variable", [col for col in data.columns if col != 'date'])
independent_vars = st.sidebar.multiselect(
    "Independent Variables", [col for col in data.columns if col not in ['date', dependent_var]]
)
model_choice = st.sidebar.selectbox("Forecasting Model", ["Linear Regression", "Multiple Regression", "Random Forest", "Moving Average", "Polynomial Regression"])
forecast_periods = st.sidebar.slider("Forecast Periods", 1, 36, 12)

# --------------------------------
# Run Forecast Button
# --------------------------------
if st.sidebar.button("Run Forecast"):
    y = data[dependent_var].values
    x = np.arange(len(y)).reshape(-1, 1)

    predictions = []
    model_info = ""
    accuracy = None

    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(x, y)
        future_x = np.arange(len(y), len(y)+forecast_periods).reshape(-1, 1)
        predictions = model.predict(future_x)
        accuracy = r2_score(y, model.predict(x))
        model_info = f"Slope: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}"

    elif model_choice == "Multiple Regression":
        if not independent_vars:
            st.error("Please select at least one independent variable.")
        else:
            X = data[independent_vars].values
            model = LinearRegression()
            model.fit(X, y)
            last_values = X[-1].reshape(1, -1)
            predictions = [model.predict(last_values)[0]] * forecast_periods
            accuracy = r2_score(y, model.predict(X))
            model_info = f"Coefficients: {model.coef_}, Intercept: {model.intercept_:.4f}"

    elif model_choice == "Random Forest":
        if not independent_vars:
            independent_vars = [dependent_var]
        X = np.arange(len(y)).reshape(-1, 1)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        future_x = np.arange(len(y), len(y)+forecast_periods).reshape(-1, 1)
        predictions = model.predict(future_x)
        accuracy = r2_score(y, model.predict(X))
        model_info = "Random Forest with 100 trees."

    elif model_choice == "Moving Average":
        window = min(3, len(y))
        avg_value = np.mean(y[-window:])
        predictions = [avg_value] * forecast_periods
        model_info = f"Window: {window}, Average: {avg_value:.2f}"

    elif model_choice == "Polynomial Regression":
        poly_features = np.vstack([x.flatten(), x.flatten()**2]).T
        model = LinearRegression()
        model.fit(poly_features, y)
        future_poly = np.vstack([
            np.arange(len(y), len(y)+forecast_periods),
            np.arange(len(y), len(y)+forecast_periods)**2
        ]).T
        predictions = model.predict(future_poly)
        accuracy = r2_score(y, model.predict(poly_features))
        model_info = f"Equation: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x + {model.coef_[1]:.4f}x²"

    # Generate future dates
    last_date = pd.to_datetime(data['date'].iloc[-1] + '-01')
    future_dates = [(last_date + pd.DateOffset(months=i+1)).strftime("%Y-%m") for i in range(forecast_periods)]

    # Combine historical + forecast
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': np.round(predictions)
    })
    chart_data = pd.concat([data[['date', dependent_var]].rename(columns={dependent_var: 'value'}),
                             forecast_df.rename(columns={'forecast': 'value'})],
                            ignore_index=True)

    # Display chart
    fig = px.line(chart_data, x="date", y="value", title="Forecast Visualization", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Display model info and results
    st.subheader("Model Performance")
    st.write(f"R² Score: {accuracy*100:.2f}%" if accuracy else "N/A")
    st.write(model_info)

    st.subheader("Forecast Results")
    st.dataframe(forecast_df)
