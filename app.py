import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="My First Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add a title
st.title('My First Streamlit Dashboard')

# Add some text
st.write("Welcome to my dashboard! This is a simple example.")

# Create sample data
@st.cache_data
def load_data():
    df = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        'Sales': np.random.randint(1000, 5000, 5),
        'Costs': np.random.randint(500, 2000, 5)
    })
    return df

# Load the data
data = load_data()

# Display the data
st.subheader('Sample Sales Data')
st.dataframe(data)

# Create a simple bar chart
st.subheader('Sales by Month')
st.bar_chart(data.set_index('Month')['Sales'])

# Add an interactive element
month_filter = st.selectbox('Select a month to see details:', data['Month'])
filtered_data = data[data['Month'] == month_filter]

col1, col2 = st.columns(2)
with col1:
    st.metric("Sales", f"${filtered_data['Sales'].values[0]:,}")
with col2:
    st.metric("Costs", f"${filtered_data['Costs'].values[0]:,}")
