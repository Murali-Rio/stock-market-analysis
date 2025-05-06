import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Market Overview",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        background-color: #343a40;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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

def get_most_active_stocks(df, n=10):
    """Get the most active stocks based on trading volume"""
    return df.nlargest(n, 'volume').copy()

def create_active_stocks_chart(active_stocks):
    """Create a visualization for most active stocks"""
    # Volume bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=active_stocks['symbol'],
        y=active_stocks['volume'],
        name='Volume',
        marker_color='#2196f3',
        text=active_stocks['volume'].apply(lambda x: f'{x:,.0f}'),
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Most Active Stocks by Volume',
        xaxis_title='Stock Symbol',
        yaxis_title='Volume',
        height=500,
        showlegend=False,
        yaxis=dict(type='log')  # Log scale for better visualization
    )
    
    return fig

def display_active_stocks_metrics(active_stocks):
    """Display metrics for most active stocks in a grid"""
    cols = st.columns(2)
    
    for idx, stock in active_stocks.iterrows():
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stock['symbol']}</h3>
                <p>Price: ${stock['price']:,.2f}</p>
                <p>Volume: {stock['volume']:,.0f}</p>
                <p>Change: <span style="color: {'green' if stock['change'] >= 0 else 'red'}">{stock['change']:+.2f}%</span></p>
                <p>Market Cap: ${stock['market_cap']/1e9:.2f}B</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="margin: 0;">🌐 Market Overview</h1>
        <p style="margin: 0;">Market Activity and Trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_stock_data()
    if df is None or df.empty:
        st.error("No data available. Please check if the CSV file exists and contains valid data.")
        return
    
    # Market summary metrics
    st.markdown("### 📊 Market Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Market Cap</h4>
            <h3>${df['market_cap'].sum()/1e12:.2f}T</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_change = df['change'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4>Average Change</h4>
            <h3 style="color: {'green' if avg_change >= 0 else 'red'}">{avg_change:+.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_volume = df['volume'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Volume</h4>
            <h3>{total_volume:,.0f}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_pe = df['pe_ratio'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4>Average P/E</h4>
            <h3>{avg_pe:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Most Active Stocks Section
    st.markdown("### 🔥 Most Active Stocks")
    
    # Get most active stocks
    active_stocks = get_most_active_stocks(df)
    
    # Display volume chart
    volume_chart = create_active_stocks_chart(active_stocks)
    st.plotly_chart(volume_chart, use_container_width=True)
    
    # Display detailed metrics for active stocks
    st.markdown("### 📊 Active Stocks Details")
    display_active_stocks_metrics(active_stocks)

if __name__ == "__main__":
    main() 
