import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib

# Set page configuration
st.set_page_config(
    page_title="Energy Analytics Platform",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS to improve UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #26A69A;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #303F9F;
    }
    .metric-label {
        font-size: 1rem;
        color: #455A64;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #e0f7fa 0%, #80deea 100%);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title
st.markdown("<h1 class='main-header'> Energy Analytics & Prediction Platform</h1>", unsafe_allow_html=True)

# Function to load and clean solar energy data
@st.cache_data
def load_solar_data():
    # Load the solar energy data
    try:
        df = pd.read_csv('solarenergy.csv')
        
        # Convert Datetime to datetime format
        # Format is "DD/MM/YYYY H:MM" based on the provided sample
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M', dayfirst=True)
        
        # Handle missing values
        df = df.dropna(subset=['solar_mw'])  # Drop rows where solar_mw is missing
        
        # Replace zero values in solar_mw with NaN when temperature is high enough
        # This is based on the assumption that zero production during daylight is likely a data error
        # First convert solar_mw to numeric if it's not already
        df['solar_mw'] = pd.to_numeric(df['solar_mw'], errors='coerce')
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        
        # Now replace zeros when temperature is high
        df.loc[(df['solar_mw'] == 0) & (df['temperature'] > 15), 'solar_mw'] = np.nan
        
        # Create hour and day columns for analysis
        df['hour'] = df['Datetime'].dt.hour
        df['day'] = df['Datetime'].dt.date
        df['Day'] = df['Datetime'].dt.day
        df['Month'] = df['Datetime'].dt.month
        df['Year'] = df['Datetime'].dt.year
        
        # Create bins for humidity and wind speed
        df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
        df['wind-speed'] = pd.to_numeric(df['wind-speed'], errors='coerce')
        df['humidity_bin'] = pd.cut(df['humidity'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df['wind_speed_bin'] = pd.cut(df['wind-speed'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        return df
    except Exception as e:
        st.error(f"Error loading solar data: {e}")
        return pd.DataFrame()  # Return empty dataframe if there's an error

# Function to load and clean wind energy data
@st.cache_data
def load_wind_data():
    # Load the wind energy data
    try:
        df = pd.read_csv('T1.csv')
        
        # Convert Date/Time to datetime format
        # Format is "DD MM YYYY HH:MM" based on the provided sample
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
        
        # Handle missing values
        df = df.dropna(subset=['LV ActivePower (kW)'])
        
        # Create hour and day columns for analysis
        df['hour'] = df['Date/Time'].dt.hour
        df['day'] = df['Date/Time'].dt.date
        df['Day'] = df['Date/Time'].dt.day
        df['Month'] = df['Date/Time'].dt.month
        df['Year'] = df['Date/Time'].dt.year
        
        # Convert numeric columns to appropriate data types
        numeric_columns = ['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create wind speed bins for boxplot
        df['wind_speed_bin'] = pd.cut(df['Wind Speed (m/s)'], bins=5, 
                                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        return df
    except Exception as e:
        st.error(f"Error loading wind data: {e}")
        return pd.DataFrame()  # Return empty dataframe if there's an error

# Function to load machine learning models
@st.cache_resource
def load_models():
    try:
        # Load models and scalers
        models = {
            'wind_model': joblib.load('wind_model.joblib'),
            'wind_scaler': joblib.load('scaler-2.joblib'),
            'solar_model': joblib.load('best_model_solar-2.joblib'),
            'solar_scaler': joblib.load('scaler_solar.joblib')
        }
        return models, None
    except Exception as e:
        return None, f"Error loading models: {e}"

# Create tabs for navigation
tab_names = ["Dashboard", "Predictions",]
dashboard_tab, predictions_tab = st.tabs(tab_names)

with dashboard_tab:
    st.markdown("<h2 class='sub-header'>Energy Production Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Interactive analytics for solar and wind energy production data")
    
    # Load data
    try:
        solar_df = load_solar_data()
        wind_df = load_wind_data()
        
        # Check if data loaded correctly
        if solar_df.empty or wind_df.empty:
            st.warning("Some data couldn't be loaded. Please check the file paths.")
        else:
            st.success("Data loaded successfully!")
            
            # Display basic info about the datasets
            with st.expander("Dataset Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Solar Energy Dataset")
                    st.write(f"Records: {solar_df.shape[0]}")
                    if not solar_df.empty:
                        st.write(f"Time Range: {solar_df['Datetime'].min()} to {solar_df['Datetime'].max()}")
                    
                with col2:
                    st.subheader("Wind Energy Dataset")
                    st.write(f"Records: {wind_df.shape[0]}")
                    if not wind_df.empty:
                        st.write(f"Time Range: {wind_df['Date/Time'].min()} to {wind_df['Date/Time'].max()}")
            
            # Create subtabs for Solar and Wind Energy
            solar_subtab, wind_subtab = st.tabs(["Solar Energy", "Wind Energy"])
            
            #######################
            # SOLAR ENERGY CHARTS #
            #######################
            with solar_subtab:
                st.markdown("<h3 class='sub-header'>Solar Energy Analysis</h3>", unsafe_allow_html=True)
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # 1. Solar Radiation vs Energy Output (using temperature as proxy)
                    st.subheader("Solar Radiation vs Energy Output")
                    
                    fig = px.scatter(solar_df, x='temperature', y='solar_mw', 
                                    color='temperature', 
                                    title='Solar Radiation (Temperature) vs Energy Output',
                                    labels={'temperature': 'Temperature (°C)', 'solar_mw': 'Solar Power (MW)'},
                                    color_continuous_scale='magma')
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. Solar Power vs Temperature
                    st.subheader("Solar Power vs Temperature")
                    
                    fig = px.scatter(solar_df, x='temperature', y='solar_mw',
                                    trendline='ols',
                                    title='Solar Power Output vs Temperature',
                                    labels={'temperature': 'Temperature (°C)', 'solar_mw': 'Solar Power (MW)'})
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 5. Solar Power vs Wind Speed
                    st.subheader("Solar Power vs Wind Speed")
                    
                    fig = px.scatter(solar_df, x='wind-speed', y='solar_mw',
                                    color='temperature', 
                                    title='Solar Power vs Wind Speed',
                                    labels={'wind-speed': 'Wind Speed', 'solar_mw': 'Solar Power (MW)'},
                                    color_continuous_scale='magma')
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 2. Hourly Solar Power Trend
                    st.subheader("Hourly Solar Power Trend")
                    
                    # Get date range for filter
                    if not solar_df.empty:
                        min_date = solar_df['Datetime'].dt.date.min()
                        max_date = solar_df['Datetime'].dt.date.max()
                        selected_date = st.date_input("Select Date", min_date, min_value=min_date, max_value=max_date)
                        
                        # Filter data for selected date
                        date_filtered_data = solar_df[solar_df['Datetime'].dt.date == selected_date]
                        
                        if not date_filtered_data.empty:
                            fig = px.line(date_filtered_data, x='Datetime', y='solar_mw',
                                        title=f'Hourly Solar Power Production ({selected_date})',
                                        labels={'Datetime': 'Time', 'solar_mw': 'Solar Power (MW)'})
                            
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("No data available for selected date.")
                    else:
                        st.warning("Solar data not available.")
                    
                    # 4. Solar Output Distribution by Humidity Level
                    st.subheader("Solar Output Distribution by Humidity Level")
                    
                    # Filter out any rows with missing humidity_bin
                    humidity_filtered_data = solar_df.dropna(subset=['humidity_bin', 'solar_mw'])
                    
                    if not humidity_filtered_data.empty:
                        fig = px.box(humidity_filtered_data, x='humidity_bin', y='solar_mw',
                                    title='Solar Power Distribution by Humidity',
                                    labels={'humidity_bin': 'Humidity Level', 'solar_mw': 'Solar Power (MW)'},
                                    category_orders={"humidity_bin": ['Very Low', 'Low', 'Medium', 'High', 'Very High']})
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Not enough data for humidity analysis.")
                    
                    # Additional chart: Solar Power Production by Hour of Day (Average)
                    st.subheader("Solar Power Production by Hour of Day")
                    
                    # Calculate average solar power by hour
                    if not solar_df.empty:
                        hourly_avg = solar_df.groupby('hour')['solar_mw'].mean().reset_index()
                        
                        fig = px.bar(hourly_avg, x='hour', y='solar_mw',
                                    title='Average Solar Power by Hour of Day',
                                    labels={'hour': 'Hour of Day', 'solar_mw': 'Avg Solar Power (MW)'})
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Solar data not available.")
            
            ######################
            # WIND ENERGY CHARTS #
            ######################
            with wind_subtab:
                st.markdown("<h3 class='sub-header'>Wind Energy Analysis</h3>", unsafe_allow_html=True)
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # 1. Wind Direction Distribution (Wind Rose)
                    st.subheader("Wind Direction Distribution")
                    
                    if not wind_df.empty:
                        fig = px.scatter_polar(wind_df, r='Wind Speed (m/s)', 
                                            theta='Wind Direction (°)',
                                            color='LV ActivePower (kW)',
                                            title='Wind Direction Distribution',
                                            color_continuous_scale='viridis')
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Wind data not available.")
                    
                    # 3. Wind Power Distribution by Wind Speed
                    st.subheader("Wind Power Distribution by Wind Speed")
                    
                    if not wind_df.empty:
                        fig = px.box(wind_df, x='wind_speed_bin', y='LV ActivePower (kW)',
                                    title='Wind Power Distribution by Wind Speed',
                                    labels={'wind_speed_bin': 'Wind Speed Category', 
                                            'LV ActivePower (kW)': 'Power Output (kW)'},
                                    category_orders={"wind_speed_bin": ['Very Low', 'Low', 'Medium', 'High', 'Very High']})
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Wind data not available.")
                    
                    # 5. Hourly Wind Power Production
                    st.subheader("Hourly Wind Power Production")
                    
                    # Get date range for filter
                    if not wind_df.empty:
                        wind_min_date = wind_df['Date/Time'].dt.date.min()
                        wind_max_date = wind_df['Date/Time'].dt.date.max()
                        wind_selected_date = st.date_input("Select Date", wind_min_date, 
                                                        min_value=wind_min_date, max_value=wind_max_date,
                                                        key="wind_date")
                        
                        # Filter data for selected date
                        wind_date_filtered = wind_df[wind_df['Date/Time'].dt.date == wind_selected_date]
                        
                        if not wind_date_filtered.empty:
                            fig = px.line(wind_date_filtered, x='Date/Time', y='LV ActivePower (kW)',
                                        title=f'Hourly Wind Power Output ({wind_selected_date})',
                                        labels={'Date/Time': 'Time', 'LV ActivePower (kW)': 'Power Output (kW)'})
                            
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("No data available for selected date.")
                    else:
                        st.warning("Wind data not available.")
                
                with col2:
                    # 2. Actual vs Theoretical Wind Power
                    st.subheader("Actual vs Theoretical Wind Power")
                    
                    # Get date range for filter
                    if not wind_df.empty:
                        theory_min_date = wind_df['Date/Time'].dt.date.min()
                        theory_max_date = wind_df['Date/Time'].dt.date.max()
                        theory_selected_date = st.date_input("Select Date", theory_min_date, 
                                                        min_value=theory_min_date, max_value=theory_max_date,
                                                        key="theory_date")
                        
                        # Filter data for selected date
                        theory_date_filtered = wind_df[wind_df['Date/Time'].dt.date == theory_selected_date]
                        
                        if not theory_date_filtered.empty:
                            fig = go.Figure()
                            
                            # Add Actual Power
                            fig.add_trace(go.Scatter(
                                x=theory_date_filtered['Date/Time'],
                                y=theory_date_filtered['LV ActivePower (kW)'],
                                name='Actual Power',
                                line=dict(color='blue')
                            ))
                            
                            # Add Theoretical Power
                            fig.add_trace(go.Scatter(
                                x=theory_date_filtered['Date/Time'],
                                y=theory_date_filtered['Theoretical_Power_Curve (KWh)'],
                                name='Theoretical Power',
                                line=dict(color='green', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f'Actual vs Theoretical Wind Power ({theory_selected_date})',
                                xaxis_title='Time',
                                yaxis_title='Power (kW)',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("No data available for selected date.")
                    else:
                        st.warning("Wind data not available.")
                    
                    # 4. Wind Speed vs Power Output
                    st.subheader("Wind Speed vs Power Output")
                    
                    if not wind_df.empty:
                        fig = px.scatter(wind_df, x='Wind Speed (m/s)', y='LV ActivePower (kW)',
                                    color='Wind Speed (m/s)', 
                                    title='Wind Speed vs Power Output',
                                    labels={'Wind Speed (m/s)': 'Wind Speed (m/s)', 
                                            'LV ActivePower (kW)': 'Power Output (kW)'},
                                    color_continuous_scale='viridis')
                        
                        # Add trendline
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Wind data not available.")
                    
                    # Additional chart: Power Curve Analysis
                    st.subheader("Wind Power Curve Analysis")
                    
                    if not wind_df.empty:
                        # Bin wind speeds and calculate average power output
                        power_curve = wind_df.groupby(pd.cut(wind_df['Wind Speed (m/s)'], 
                                                        bins=20))['LV ActivePower (kW)'].mean().reset_index()
                        
                        # Convert interval to string for x-axis
                        power_curve['Wind Speed (m/s)'] = power_curve['Wind Speed (m/s)'].astype(str)
                        
                        fig = px.line(power_curve, x='Wind Speed (m/s)', y='LV ActivePower (kW)',
                                    markers=True,
                                    title='Wind Power Curve',
                                    labels={'Wind Speed (m/s)': 'Wind Speed Range (m/s)', 
                                        'LV ActivePower (kW)': 'Average Power Output (kW)'})
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Wind data not available.")
            
            # Summary metrics at the bottom
            st.markdown("<h2 class='sub-header'>Summary Statistics</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if not solar_df.empty:
                    max_solar = solar_df['solar_mw'].max()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{max_solar:.2f} MW</div>
                        <div class="metric-label">Max Solar Output</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Solar data not available")
            
            with col2:
                if not solar_df.empty:
                    avg_solar = solar_df['solar_mw'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_solar:.2f} MW</div>
                        <div class="metric-label">Avg Solar Output</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Solar data not available")
            
            with col3:
                if not wind_df.empty:
                    max_wind = wind_df['LV ActivePower (kW)'].max() / 1000  # Convert to MW
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{max_wind:.2f} MW</div>
                        <div class="metric-label">Max Wind Output</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Wind data not available")
            
            with col4:
                if not wind_df.empty:
                    avg_wind = wind_df['LV ActivePower (kW)'].mean() / 1000  # Convert to MW
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_wind:.2f} MW</div>
                        <div class="metric-label">Avg Wind Output</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Wind data not available")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please make sure both 'solarenergy.csv' and 'T1.csv' files are in the same directory as the app.")

# ==================== PREDICTIONS TAB ====================
with predictions_tab:
    st.markdown("<h2 class='sub-header'>Energy Production Prediction</h2>", unsafe_allow_html=True)
    st.markdown("Use machine learning models to predict energy output based on environmental conditions")
    
    # Load models
    models, error = load_models()
    
    if error:
        st.error(error)
        st.error("Please make sure model files are in the same directory as the app.")
    elif models:
        # Create subtabs for prediction types
        wind_pred_tab, solar_pred_tab = st.tabs(["Wind Power Prediction", "Solar Energy Prediction"])
        
        # ======= WIND POWER PREDICTION =======
        with wind_pred_tab:
            st.markdown("<h3 class='sub-header'>Wind Power Prediction</h3>", unsafe_allow_html=True)
            
            # Create columns for input form
            col1, col2 = st.columns(2)
            
            with col1:
                # Date and time inputs
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Date & Time")
                day = st.number_input('Day', min_value=1, max_value=31, value=15, key="wind_day")
                month = st.number_input('Month', min_value=1, max_value=12, value=6, key="wind_month")
                year = st.number_input('Year', min_value=2006, max_value=2025, value=2023, key="wind_year")
                hour = st.number_input('Hour', min_value=0, max_value=23, value=12, key="wind_hour")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Wind condition inputs
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Wind Conditions")
                wind_speed = st.number_input('Wind Speed (m/s)', min_value=0.0, value=5.0, key="wind_speed_input")
                wind_dir = st.number_input('Wind Direction (°)', min_value=0.0, max_value=360.0, value=180.0)
                th_power = st.number_input('Theoretical Power Curve (KWh)', value=500.0)
                st.markdown("</div>", unsafe_allow_html=True)

            # Combine inputs
            input_data = {
                'Day': day,
                'Month': month,
                'Year': year,
                'hour': hour,
                'Theoretical_Power_Curve (KWh)': th_power,
                'Wind Direction (°)': wind_dir,
                'Wind Speed (m/s)': wind_speed
            }

            def preprocess_wind_input(user_input):
                input_df = pd.DataFrame([user_input])
                input_df = input_df[['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)', 'Day', 'Month', 'Year', 'hour']]
                return models['wind_scaler'].transform(input_df)

            # Power efficiency info
            with st.expander("Wind Turbine Efficiency Information"):
                st.write("""
                Wind turbine efficiency is typically measured by comparing actual power output to theoretical power. 
                The theoretical power is calculated based on the wind speed and the turbine specifications.
                Factors that affect efficiency include:
                - Wind direction relative to turbine orientation
                - Air density (affected by temperature and humidity)
                - Mechanical losses in the turbine system
                """)
                
                # Create a sample efficiency curve
                wind_speeds = np.linspace(0, 25, 100)
                theoretical_power = np.where(wind_speeds < 3, 0, 
                                        np.where(wind_speeds > 20, 1500,
                                                (wind_speeds**3) * 5))
                actual_power = theoretical_power * np.where(wind_speeds < 3, 0,
                                                        np.where(wind_speeds > 20, 0.7,
                                                                0.9 * np.exp(-(wind_speeds-12)**2/50)))
                
                efficiency_df = pd.DataFrame({
                    'Wind Speed (m/s)': wind_speeds,
                    'Theoretical Power (kW)': theoretical_power,
                    'Actual Power (kW)': actual_power,
                    'Efficiency (%)': np.where(theoretical_power > 0, actual_power / theoretical_power * 100, 0)
                })
                
                fig = px.line(efficiency_df, x='Wind Speed (m/s)', y=['Theoretical Power (kW)', 'Actual Power (kW)'],
                            title='Typical Wind Turbine Power Curve')
                st.plotly_chart(fig, use_container_width=True)

            if st.button('Predict Wind Power', key="wind_predict_btn"):
                try:
                    processed_input = preprocess_wind_input(input_data)
                    prediction = models['wind_model'].predict(processed_input)
                    
                    # Calculate efficiency percentage
                    if input_data['Theoretical_Power_Curve (KWh)'] > 0:
                        efficiency = (prediction[0] / input_data['Theoretical_Power_Curve (KWh)']) * 100
                    else:
                        efficiency = 0
                    
                    # Display prediction with improved styling
                    st.markdown(f"""
                    <div class="prediction-result">
                        Predicted Power Output: {prediction[0]:.2f} kW
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show additional metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{efficiency:.1f}%</div>
                            <div class="metric-label">Turbine Efficiency</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        daily_energy = prediction[0] * 24 / 1000  # Convert to MWh
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{daily_energy:.2f} MWh</div>
                            <div class="metric-label">Est. Daily Energy</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        
        # ======= SOLAR ENERGY PREDICTION =======
        with solar_pred_tab:
            st.markdown("<h3 class='sub-header'>Solar Energy Prediction</h3>", unsafe_allow_html=True)
            
            # Create columns for input form
            col1, col2 = st.columns(2)
            
            with col1:
                # Weather condition inputs
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Weather Conditions")
                temperature = st.number_input("Temperature (°F)", value=69.0, key="solar_temp")
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, key="solar_humidity")
                wind_speed = st.number_input("Wind Speed", value=7.5, key="solar_wind")
                avg_pressure = st.number_input("Average Pressure (Period)", value=29.82, key="solar_pressure")
                avg_wind_speed = st.number_input("Average Wind Speed (Period)", value=8.0, key="solar_avg_wind")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Date and time inputs
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Date & Time")
                day = st.number_input("Day", min_value=1, max_value=31, value=8, key="solar_day")
                month = st.number_input("Month", min_value=1, max_value=12, value=3, key="solar_month")
                hour = st.number_input("Time (Hour)", min_value=0, max_value=23, value=12, key="solar_hour")
                st.markdown("</div>", unsafe_allow_html=True)

            # Create input dataframe
            solar_input_df = pd.DataFrame({
                'wind-speed': [wind_speed],
                'humidity': [humidity],
                'average-wind-speed-(period)': [avg_wind_speed],
                'average-pressure-(period)': [avg_pressure],
                'temperature': [temperature],
                'Day': [day],
                'Month': [month],
                'Time': [hour]  # This matches the 'hour' column but using the name from the original code
            })
            
            # Solar energy information
            with st.expander("Solar Energy Efficiency Information"):
                st.write("""
                Solar energy production is influenced by several key factors:
                
                1. **Solar Irradiance**: The amount of solar radiation reaching the panels (correlated with temperature in our model)
                2. **Panel Temperature**: Performance decreases as panels get too hot
                3. **Weather Conditions**: Cloud cover, humidity, and air clarity affect energy production
                4. **Time of Day**: Peak production typically occurs around solar noon
                5. **Season**: Varies throughout the year based on the sun's position
                
                The model predicts production based on historical patterns of these variables.
                """)
                
                # Sample daily solar production curve
                hours = np.arange(0, 24)
                production = np.zeros(24)
                for i in range(24):
                    if i >= 6 and i <= 18:  # Daylight hours
                        production[i] = 30 * np.sin(np.pi * (i - 6) / 12)
                    
                daily_curve = pd.DataFrame({
                    'Hour': hours,
                    'Production (MW)': production
                })
                
                fig = px.line(daily_curve, x='Hour', y='Production (MW)', 
                             title='Typical Daily Solar Production Curve')
                st.plotly_chart(fig, use_container_width=True)

            if st.button("Predict Solar Energy", key="solar_predict_btn"):
                try:
                    scaled_input = models['solar_scaler'].transform(solar_input_df)
                    prediction = models['solar_model'].predict(scaled_input)
                    
                    # Display prediction with improved styling
                    st.markdown(f"""
                    <div class="prediction-result">
                        Predicted Solar Energy Output: {prediction[0]:.2f} MW
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate and display additional metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Calculate daily energy production estimate (based on daylight hours)
                        # Assuming this prediction is for the specified hour, scale for a full day
                        # considering solar production follows a sine curve during daylight
                        
                        # Estimate daylight hours based on month (simplified)
                        daylight_hours = {
                            1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15,
                            7: 15, 8: 14, 9: 12, 10: 11, 11: 10, 12: 9
                        }
                        hours = daylight_hours.get(month, 12)
                        
                        # Calculate daily energy assuming a sine distribution across daylight hours
                        # and that our prediction is for the peak hour
                        daily_energy = prediction[0] * hours * 0.64  # 0.64 is average value of sine curve over half period
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{daily_energy:.2f} MWh</div>
                            <div class="metric-label">Est. Daily Energy</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Calculate estimated homes powered
                        # Assuming average home uses 30 kWh per day
                        homes_powered = int(daily_energy * 1000 / 30)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{homes_powered}</div>
                            <div class="metric-label">Est. Homes Powered</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show environmental impact
                    st.subheader("Environmental Impact")
                    co2_saved = daily_energy * 0.94  # tons of CO2 saved compared to coal
                    st.markdown(f"This clean energy production saves approximately **{co2_saved:.2f} tons** of CO2 emissions daily compared to coal power generation.")
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Models could not be loaded. Prediction features are unavailable.")
