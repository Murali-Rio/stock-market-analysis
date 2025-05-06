import streamlit as st
import pandas as pd
from datetime import datetime
import logging
import plotly.express as px
import plotly.graph_objects as go
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stock-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .stock-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .positive-change {
        color: #28a745;
        font-weight: bold;
    }
    .negative-change {
        color: #dc3545;
        font-weight: bold;
    }
    .header {
        background: linear-gradient(135deg, #343a40 0%, #212529 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chart-container {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        color: #343a40;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e9ecef;
    }
    .last-updated {
        color: #6c757d;
        font-size: 12px;
        text-align: center;
        margin-top: 10px;
    }
    .refresh-info {
        color: #6c757d;
        font-size: 14px;
        text-align: center;
        margin: 10px 0;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=30)  # Cache data for 30 seconds
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

def display_stock_card(stock, is_top_performer=True):
    """Display a stock card with key metrics"""
    change_class = "positive-change" if stock['change'] >= 0 else "negative-change"
    change_symbol = "↑" if stock['change'] >= 0 else "↓"
    card_class = "top-performer" if is_top_performer else "bottom-performer"
    
    st.markdown(f"""
    <div class="stock-card {card_class}">
        <div class="stock-header">
            <h3 style="margin: 0;">{stock['symbol']}</h3>
            <span class="sector-badge">{stock['sector']}</span>
        </div>
        <div class="stock-price">
            <span class="price-label">Current Price</span>
            <span class="price-value">${stock['price']:,.2f}</span>
        </div>
        <div class="stock-metrics">
            <div class="{change_class}">
                <span class="metric-label">Change</span>
                <span class="metric-value">{change_symbol} {abs(stock['change']):.2f}%</span>
            </div>
            <div class="volume">
                <span class="metric-label">Volume</span>
                <span class="metric-value">{stock['volume']:,.0f}</span>
            </div>
            <div class="market-cap">
                <span class="metric-label">Market Cap</span>
                <span class="metric-value">${stock['market_cap']/1e9:.2f}B</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_visualizations(df):
    """Create visualizations for the dashboard"""
    # Top and bottom performers
    top_performers = df.nlargest(10, 'change')
    bottom_performers = df.nsmallest(10, 'change')
    
    # Create a combined dataframe with performance category
    top_performers['performance'] = 'Top 10'
    bottom_performers['performance'] = 'Bottom 10'
    combined = pd.concat([top_performers, bottom_performers])
    
    # Create a single bar chart for both top and bottom performers
    fig = px.bar(
        combined,
        x='symbol',
        y='change',
        color='performance',
        title='Top 10 vs Bottom 10 Stock Performers',
        labels={'change': 'Daily Change (%)', 'symbol': 'Stock Symbol'},
        color_discrete_map={
            'Top 10': '#28a745',  # Green for top performers
            'Bottom 10': '#dc3545'  # Red for bottom performers
        }
    )
    
    # Update layout for better visualization
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title='Performance Category',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis=dict(
            title='Stock Symbol',
            tickangle=45
        ),
        yaxis=dict(
            title='Daily Change (%)'
        ),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#343a40"
        )
    )
    
    # Add value labels on bars
    fig.update_traces(
        texttemplate='%{y:.2f}%',
        textposition='outside',
        textfont=dict(
            size=12,
            color='black'
        )
    )
    
    return fig

def main():
    # Add this CSS to your existing styles
    st.markdown("""
    <style>
    .stock-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .top-performer {
        border-left: 5px solid #28a745;
        background: linear-gradient(to right, rgba(40, 167, 69, 0.05), white);
    }
    
    .bottom-performer {
        border-left: 5px solid #dc3545;
        background: linear-gradient(to right, rgba(220, 53, 69, 0.05), white);
    }
    
    .stock-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .sector-badge {
        background: #f8f9fa;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        color: #6c757d;
    }
    
    .stock-price {
        margin-bottom: 15px;
    }
    
    .price-label {
        display: block;
        font-size: 0.8em;
        color: #6c757d;
    }
    
    .price-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .stock-metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
    }
    
    .metric-label {
        display: block;
        font-size: 0.8em;
        color: #6c757d;
    }
    
    .metric-value {
        display: block;
        font-weight: bold;
    }
    
    .positive-change {
        color: #28a745;
    }
    
    .negative-change {
        color: #dc3545;
    }
    
    .stock-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    
    .top-section-title {
        background: linear-gradient(to right, rgba(40, 167, 69, 0.1), transparent);
        color: #28a745;
    }
    
    .bottom-section-title {
        background: linear-gradient(to right, rgba(220, 53, 69, 0.1), transparent);
        color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with gradient background
    st.markdown("""
    <div class="header">
        <h1 style="margin: 0; font-size: 36px;">📈 Stock Market Dashboard</h1>
        <p style="margin: 10px 0 0 0; font-size: 18px; opacity: 0.9;">Real-time market data and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Refresh info
    st.markdown("""
    <div class="refresh-info">
        Data automatically refreshes every 30 seconds
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_stock_data()
    if df is None or df.empty:
        st.error("No data available. Please check if the CSV file exists and contains valid data.")
        return
    
    # Stock Overview section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title top-section-title">🚀 Top Performers</div>', unsafe_allow_html=True)
        for stock in df.nlargest(10, 'change').iterrows():
            display_stock_card(stock[1], is_top_performer=True)
    
    with col2:
        st.markdown('<div class="section-title bottom-section-title">📉 Bottom Performers</div>', unsafe_allow_html=True)
        for stock in df.nsmallest(10, 'change').iterrows():
            display_stock_card(stock[1], is_top_performer=False)
    
    # Performance Analysis
    st.markdown('<div class="section-title">📊 Performance Analysis</div>', unsafe_allow_html=True)
    fig = create_visualizations(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Last updated time
    st.markdown(f"""
    <div class="last-updated">
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(30)
    st.experimental_rerun()

if __name__ == "__main__":
    main()
