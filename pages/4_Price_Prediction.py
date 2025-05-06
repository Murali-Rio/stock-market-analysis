import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Custom CSS
st.markdown("""
<style>
.header {
    padding: 1rem;
    margin-bottom: 2rem;
    background: linear-gradient(to right, rgba(25, 118, 210, 0.1), transparent);
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)
def load_stock_data():
    """Load stock data from CSV file"""
    try:
        df = pd.read_csv('stock_data.csv')
        numeric_columns = ['price', 'change', 'volume', 'market_cap', 'pe_ratio', 'dividend_yield']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Error loading stock data: {str(e)}")
        return None

def generate_historical_data(current_price, days=365):
    """Generate historical price data based on current price"""
    dates = pd.date_range(end=datetime.now(), periods=days).tolist()
    
    # Generate realistic price movements
    volatility = 0.02
    returns = np.random.normal(loc=0.0001, scale=volatility, size=days-1)
    price_factors = np.exp(np.cumsum(returns[::-1]))
    prices = current_price / price_factors[-1] * price_factors
    prices = np.append(prices, current_price)
    
    # Create dataframe
    data = {
        'ds': dates,
        'y': prices
    }
    
    return pd.DataFrame(data)

def train_prophet_model(historical_data, periods=30):
    """Train Prophet model and generate forecast"""
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    model.fit(historical_data)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, forecast

def plot_prediction(historical_data, forecast, stock_symbol):
    """Create plot with historical data and prediction"""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=historical_data['y'],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4')
    ))
    
    # Add prediction
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-30:],  # Only show the prediction part
        y=forecast['yhat'][-30:],
        mode='lines',
        name='Prediction',
        line=dict(color='#ff7f0e')
    ))
    
    # Add prediction intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-30:],
        y=forecast['yhat_upper'][-30:],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-30:],
        y=forecast['yhat_lower'][-30:],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        name='95% Confidence'
    ))
    
    fig.update_layout(
        title=f'{stock_symbol} Price Prediction (30 Days)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        hovermode='x unified'
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="margin: 0;">🔮 Stock Price Prediction</h1>
        <p style="margin: 0;">Forecast future stock prices using machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_stock_data()
    if df is None or df.empty:
        st.error("No data available. Please check if the CSV file exists and contains valid data.")
        return
    
    # Stock selection
    selected_stock = st.selectbox(
        "Select a stock to predict",
        options=df['symbol'].unique(),
        format_func=lambda x: f"{x} - {df[df['symbol'] == x]['name'].iloc[0]}"
    )
    
    # Get current stock info
    stock_info = df[df['symbol'] == selected_stock].iloc[0]
    
    # Prediction period selection
    prediction_days = st.slider(
        "Prediction period (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )
    
    # Generate historical data
    historical_data = generate_historical_data(stock_info['price'])
    
    # Train model and get forecast
    with st.spinner("Training prediction model..."):
        model, forecast = train_prophet_model(historical_data, periods=prediction_days)
    
    # Display current price and predicted price
    col1, col2, col3 = st.columns(3)
    
    current_price = stock_info['price']
    predicted_price = forecast['yhat'].iloc[-1]
    price_change = ((predicted_price - current_price) / current_price) * 100
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{stock_info['change']:.2f}%"
        )
    
    with col2:
        st.metric(
            f"Predicted Price ({prediction_days} days)",
            f"${predicted_price:.2f}",
            f"{price_change:.2f}%"
        )
    
    with col3:
        st.metric(
            "Prediction Confidence",
            f"±${(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1])/2:.2f}"
        )
    
    # Plot prediction
    fig = plot_prediction(historical_data, forecast, selected_stock)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show forecast components
    st.subheader("Forecast Components")
    
    # Plot forecast components
    with st.spinner("Generating component plots..."):
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
    
    # Disclaimer
  

if __name__ == "__main__":
    main()