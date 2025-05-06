# Stock Market Dashboard

A web application that displays the top and bottom 10 performing stocks, with data stored in MongoDB.

## Prerequisites

- Python 3.7+
- MongoDB running locally on port 27017
- pip (Python package manager)

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure MongoDB is running on your local machine:
```bash
mongod
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features

- Displays top 10 and bottom 10 performing stocks
- Real-time stock data from Yahoo Finance
- Data persistence in MongoDB
- Modern, responsive UI
- Shows key metrics:
  - Current price
  - Price change percentage
  - Trading volume
  - Market capitalization

## Data Storage

The application stores stock data in MongoDB with the following structure:
- Database: `stock_dashboard`
- Collection: `stock_data`

Each document contains:
- symbol
- name
- price
- change_percent
- volume
- market_cap
- timestamp