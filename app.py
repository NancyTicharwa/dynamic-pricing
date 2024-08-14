import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
np.float_ = np.float64
from prophet import Prophet
import pickle
import os
import plotly.graph_objs as go
import plotly.express as px


#Problem Statement : https://docs.google.com/document/d/1bnKJLT-WC6Hz5--kl00IHjgkeFiXid0u5kuydSq69ZI/edit
# Load the dataset
df = pd.read_csv('processed_airline_data_sample.csv')
df = df.head(1000)
# Define categorical features globally
categorical_features = ['Flight_Number', 'Departure_Airport',
                        'Arrival_Airport', 'Travel_Class',
                        'Booking_Channel', 'Fare_Class',
                        'Holiday_Name', 'Holiday_Type',
                        'Weather_Condition', 'Airline']

# Define all features needed for training
pricing_features = ['Flight_Number', 'Departure_Airport', 'Arrival_Airport',
                    'Booking_Channel', 'Age', 'Travel_Class', 'Fare_Class',
                    'GDP_Growth_Rate', 'Inflation_Rate', 'Unemployment_Rate',
                    'Holiday_Name', 'Holiday_Type', 'Holiday_Indicator',
                    'Weather_Condition', 'Temperature', 'Wind_Speed',
                    'Precipitation', 'Departure_Delay', 'Cancellation',
                    'Competitor_Price', 'Booking_Lead_Time', 'Airline']

# Preprocess the data
def preprocess_data(df):
    # Convert date columns to numeric
    reference_date = pd.to_datetime(df['Flight_Date']).min()
    df['Flight_Date_Numeric'] = (pd.to_datetime(df['Flight_Date']) - reference_date).dt.days
    reference_date_booking = pd.to_datetime(df['Booking_Date']).min()
    df['Booking_Date_Numeric'] = (pd.to_datetime(df['Booking_Date']) - reference_date_booking).dt.days

    # Convert flight time to minutes
    df['Flight_Time_Minutes'] = df['Flight_Time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

    # Convert holiday date to numeric
    df['Holiday_Date_Numeric'] = (pd.to_datetime(df['Holiday_Date']) - reference_date).dt.days

    # Add these new numeric columns to the feature list
    pricing_features_extended = pricing_features + ['Flight_Date_Numeric', 'Booking_Date_Numeric',
                                                    'Flight_Time_Minutes', 'Holiday_Date_Numeric']

    # Prepare the dataset for training
    X = df[pricing_features_extended]
    y = df['Ticket_Price']

    # OneHotEncode categorical features
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_encoded = pd.DataFrame(ohe.fit_transform(X[categorical_features]).toarray(), index=X.index)
    encoded_feature_names = ohe.get_feature_names_out(categorical_features)
    X_encoded.columns = encoded_feature_names

    # Combine encoded features with the rest of the data
    X = pd.concat([X.drop(categorical_features, axis=1), X_encoded], axis=1)

    # Save the final feature order after preprocessing
    final_feature_order = list(X.columns)

    # Print feature names before fitting the model
    # print("Features used during training:")
    # print(final_feature_order)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Return everything needed for model training and predictions
    return X_train, X_test, y_train, y_test, ohe, final_feature_order, df, reference_date, reference_date_booking

# Train the RandomForestRegressor model
def train_pricing_model(X_train, y_train):
    pricing_model = RandomForestRegressor(n_estimators=100, random_state=42)
    pricing_model.fit(X_train, y_train)
    return pricing_model

# Train the Prophet model
def train_prophet_model(df):
    prophet_df = df[['Flight_Date', 'Ticket_Price']]
    prophet_df.columns = ['ds', 'y']
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    return prophet_model

# Prepare the data
X_train, X_test, y_train, y_test, ohe, final_feature_order, df, reference_date, reference_date_booking = preprocess_data(df)

# Train models
pricing_model = train_pricing_model(X_train, y_train)
prophet_model = train_prophet_model(df)

# Save the models and the final feature order for later use
with open('pricing_model.pkl', 'wb') as f:
    pickle.dump(pricing_model, f)
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(prophet_model, f)
with open('final_feature_order.pkl', 'wb') as f:
    pickle.dump(final_feature_order, f)

# Define airport names and codes
airport_mapping = {
    'JFK': 'John F. Kennedy International Airport',
    'LAX': 'Los Angeles International Airport',
    'ORD': "O'Hare International Airport",
    'ATL': 'Hartsfield-Jackson Atlanta International Airport',
    'DFW': 'Dallas/Fort Worth International Airport',
    'SFO': 'San Francisco International Airport',
    'MIA': 'Miami International Airport',
    'SEA': 'Seattle-Tacoma International Airport',
    'BOS': 'Logan International Airport',
    'DEN': 'Denver International Airport'
}

departure_airports = [f"{name} ({code})" for code, name in airport_mapping.items()]
arrival_airports = departure_airports

# Sidebar for inputs
st.sidebar.image("img/Bethel (2).gif", use_column_width=True)  # Add your logo here
st.sidebar.title("Flight Details")


#-------------------------FLIGHT DETAILS

# Number of Scenarios slider with a unique key
num_scenarios = st.sidebar.slider("Number of Scenarios", 1,5, 3, key="num_scenarios_slider")

# Flight Number and Airline Name
flight_number = st.sidebar.selectbox("Flight Number", df['Flight_Number'].unique(), key="flight_number_selectbox")
airline_name = df[df['Flight_Number'] == flight_number]['Airline'].iloc[0]
st.sidebar.write(f"**Airline Name:** {airline_name}")

# Age input
age = st.sidebar.slider("Age", 0, 100, 30, key="age_slider")

# Travel Class and Fare Class
travel_class = st.sidebar.selectbox("Travel Class", ["Economy", "Business", "First Class"],
                                    key="travel_class_selectbox")
fare_class = st.sidebar.selectbox("Fare Class", ["Discount", "Regular", "Premium"], key="fare_class_selectbox")

holiday_name = st.sidebar.selectbox("Holiday Name", ["None", "Labor Day", "Christmas", "New Year's Day"],
                                    key="holiday_name_selectbox")
holiday_type = st.sidebar.selectbox("Holiday Type", ["Public", "School", "Religious"], key="holiday_type_selectbox")
booking_channel = st.sidebar.selectbox("Booking Channel", ["Online", "Travel Agent", "Mobile App"],
                                       key="booking_channel_selectbox")
holiday_indicator = st.sidebar.checkbox("Holiday Indicator", value=False, key="holiday_indicator_checkbox")
lead_time = st.sidebar.slider("Advanced Booking Days", 0, 365, 30, key="lead_time_slider")

#-------------------weather conditions

st.sidebar.markdown('### Weather Conditions')
weather_condition = st.sidebar.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"],
                                         key="weather_condition_selectbox")
temperature = st.sidebar.slider("Temperature (Celsius)", -10.0, 40.0, 25.0, key="temperature_slider")
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, key="wind_speed_slider")

#------------------economic conditions
st.sidebar.markdown('### Economic Conditions')
competitor_price = st.sidebar.slider("Competitor Price ($)", 100.0, 2000.0, 500.0, key="competitor_price_slider")
gdp_growth_rate = st.sidebar.slider("GDP Growth Rate (%)", 0.0, 10.0, 2.5, key="gdp_growth_rate_slider")
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 1.8, key="inflation_rate_slider")
unemployment_rate = st.sidebar.slider("Unemployment Rate (%)", 0.0, 10.0, 4.3, key="unemployment_rate_slider")



#--------------MAIN PAGE
# Main content layout
st.title("Airline Booking Demand Forecasting and Dynamic Pricing Optimization")
st.write("")
# st.caption("Dynamic Price Model")

# Display custom subheading with airline name
st.markdown(f"### Welcome to {airline_name}")

# Display departure and arrival inputs under the main title
col1, col2 = st.columns(2)

# Move existing departure and arrival airport inputs from sidebar to main page
with col1:
    departure_airport = st.selectbox("Departure Airport", departure_airports, key="departure_airport_selectbox")

with col2:
    arrival_airport = st.selectbox("Arrival Airport", arrival_airports, key="arrival_airport_selectbox")

# Display date and time inputs on the main page
col3, col4, col5, col6 = st.columns(4)

with col3:
    booking_date = st.date_input("Booking Date", pd.to_datetime("2023-07-15"), key="booking_date_input")

with col4:
    flight_date = st.date_input("Flight Date", pd.to_datetime("2023-08-15"), key="flight_date_input")

with col5:
    flight_time = st.time_input("Flight Time", pd.to_datetime('10:00').time(), key="flight_time_input")

with col6:
    booking_time = st.time_input("Booking Time", pd.to_datetime('09:00').time(), key="booking_time_input")

# Determine the image path based on the selected airline
image_path = f"img/{airline_name}.png"
default_image_path = "img/Other.png"

# Check if the image exists; if not, use the default image
if not os.path.exists(image_path):
    image_path = default_image_path

# Move the image to the desired location, right before "Now Analyzing Your Trip"
st.image(image_path, use_column_width=True)

st.markdown("### Now Analyzing Your Trip\nA new way of pricing for airlines")

# Initialize session state to track the current scenario step
if 'scenario_step' not in st.session_state:
    st.session_state.scenario_step = 0
    st.session_state.selected_scenarios = []

# Start scenario selection
if st.session_state.scenario_step == 0:
    if st.button('Start Scenario Selection', key="start_scenario_button"):
        st.session_state.scenario_step = 1

# Display previous scenarios
if st.session_state.selected_scenarios:
    st.markdown("### Selected Scenarios")
    for i, scenario in enumerate(st.session_state.selected_scenarios):
        st.markdown(f"**Customer {i + 1}**")
        st.write(pd.DataFrame([scenario]))

# Process scenario selection
if 0 < st.session_state.scenario_step <= num_scenarios:
    scenario_number = st.session_state.scenario_step
    st.markdown(f"### Select Data for Scenario {scenario_number}")

    scenario_data = {
        'Flight_Number': flight_number,
        'Departure_Airport': departure_airport.split('(')[-1].strip(')'),
        'Arrival_Airport': arrival_airport.split('(')[-1].strip(')'),
        'Age': age,
        'Travel_Class': travel_class,
        'Fare_Class': fare_class,
        'Holiday_Name': holiday_name,
        'Holiday_Type': holiday_type,
        'Booking_Channel': booking_channel,
        'Holiday_Indicator': holiday_indicator,
        'Booking_Lead_Time': lead_time,
        'Weather_Condition': weather_condition,
        'Temperature': temperature,
        'Wind_Speed': wind_speed,
        'Competitor_Price': competitor_price,
        'GDP_Growth_Rate': gdp_growth_rate,
        'Inflation_Rate': inflation_rate,
        'Unemployment_Rate': unemployment_rate,
        'Flight_Date': flight_date,
        'Booking_Date': booking_date,
        'Flight_Time_Minutes': flight_time.hour * 60 + flight_time.minute,
        'Airline': airline_name
    }

    st.write(pd.DataFrame([scenario_data]))  # Display the current scenario data

    if st.button(f'Confirm Scenario {scenario_number}', key=f"confirm_scenario_{scenario_number}_button"):
        st.session_state.selected_scenarios.append(scenario_data)
        st.session_state.scenario_step += 1  # Move to the next scenario
        # st.experimental_rerun()  # Force a rerun to update the scenario step

# Once all scenarios are selected
if st.session_state.scenario_step > num_scenarios:
    # Display the "Forecast Pricing" button
    if st.button("Forecast", key="forecast_pricing_button"):
        # Display forecast pricing predictions
        st.markdown("### Dynamic Price Model")

        # Load the models and the feature order
        with open('pricing_model.pkl', 'rb') as f:
            pricing_model = pickle.load(f)
        with open('prophet_model.pkl', 'rb') as f:
            prophet_model = pickle.load(f)
        with open('final_feature_order.pkl', 'rb') as f:
            final_feature_order = pickle.load(f)

        # Prepare columns for scenario predictions
        cols = st.columns(num_scenarios)

        # Prepare the predictions display
        for i, scenario in enumerate(st.session_state.selected_scenarios):
            scenario_df = pd.DataFrame([scenario])

            # Ensure all expected columns are present with default values if missing
            for col in final_feature_order:
                if col not in scenario_df.columns:
                    # Handle missing numeric columns with default value 0
                    if col not in categorical_features:
                        scenario_df[col] = 0
                    else:
                        # Handle missing categorical columns with an appropriate default value
                        scenario_df[col] = ''

            # Convert dates to numeric and drop original date columns
            scenario_df['Flight_Date_Numeric'] = (pd.to_datetime(scenario_df['Flight_Date']) - reference_date).dt.days
            scenario_df['Booking_Date_Numeric'] = (pd.to_datetime(scenario_df['Booking_Date']) - reference_date_booking).dt.days
            scenario_df = scenario_df.drop(['Flight_Date', 'Booking_Date'], axis=1)

            # One-hot encoding categorical features
            scenario_encoded = pd.DataFrame(ohe.transform(scenario_df[categorical_features]).toarray(),
                                            index=scenario_df.index)
            encoded_feature_names = ohe.get_feature_names_out(categorical_features)
            scenario_encoded.columns = encoded_feature_names

            # Prepare final DataFrame for prediction
            X_prepared = pd.concat([scenario_df.drop(categorical_features, axis=1), scenario_encoded], axis=1)

            # Remove any duplicate columns (if any exist) to avoid reindexing errors
            X_prepared = X_prepared.loc[:, ~X_prepared.columns.duplicated()]

            # Reorder columns to match the order during training
            X_prepared = X_prepared.reindex(columns=final_feature_order, fill_value=0)

            # Predict pricing
            optimized_price = pricing_model.predict(X_prepared)[0]

            # Display the scenario label above the price box
            with cols[i]:
                st.markdown(f"Price for Customer {i + 1}")
                st.markdown(f"""
                    <div style="margin: 10px; padding: 10px; background-color: #03346E; border-radius: 20px; color: white; text-align: center; width: 120px; height: 70px; display: flex; align-items: center; justify-content: center;">
                        <h2 style="font-size: 24px; color: white; margin: 0;">${optimized_price:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

        #Display Forecasted Demand
        future = pd.DataFrame({'ds': [scenario['Flight_Date']]})
        forecast = prophet_model.predict(future)
        forecasted_demand = forecast['yhat'].iloc[0]
        forecasted_date = forecast['ds'].iloc[0].date()  # Extract only the date part

        st.markdown(f"""
            <div style="margin-top: 20px; padding: 15px; background-color: #00BFFF; border-radius: 10px; color: white; text-align: center;">
                <h3>Forecasted Demand for {forecasted_date} is: <span style="color: white;">{forecasted_demand:.2f}</span> bookings</h3>
            </div>
            """, unsafe_allow_html=True)

        # Display Booking Lead Time distribution using Plotly
        st.write('')
        st.write('')
        st.write('')
        st.markdown("### Advanced Booking Distribution")
        lead_time_fig = px.histogram(df, x='Booking_Lead_Time', nbins=30, marginal="rug")
        lead_time_fig.update_layout(bargap=0.2)
        st.plotly_chart(lead_time_fig)

        # Display Booking Demand Over Time using Plotly
        st.markdown("### Booking Demand Over Time")
        time_series_data = df.groupby('Flight_Date').size().reset_index(name='Bookings')
        demand_time_series_fig = px.line(time_series_data, x='Flight_Date', y='Bookings')
        st.plotly_chart(demand_time_series_fig)

        # Display Prophet visual forecast
        st.markdown("### Demand Forecast")

        # Prepare data for Prophet visualization
        df_prophet = df.groupby('Flight_Date').size().reset_index(name='y')
        df_prophet.rename(columns={'Flight_Date': 'ds'}, inplace=True)

        # Fit the Prophet model
        prophet_model = Prophet()
        prophet_model.fit(df_prophet)

        # Set default forecast period and create the slider
        default_forecast_days = 60
        forecast_days = st.slider("Select forecast period in days:", 30, 365, default_forecast_days, key="forecast_slider")

        # Forecast the selected number of days
        future = prophet_model.make_future_dataframe(periods=forecast_days)
        forecast = prophet_model.predict(future)

        # Plot the forecast using Plotly
        fig = go.Figure()

        # Add the historical data
        fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical Data'))

        # Add the forecasted data
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

        # Add the upper and lower bounds of the forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Confidence Interval',
            line=dict(width=0),
            fillcolor='rgba(0, 0, 255, 0.2)',
            fill='tonexty'
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Lower Confidence Interval',
            line=dict(width=0),
            fillcolor='rgba(0, 0, 255, 0.2)',
            fill='tonexty'
        ))

        # Customize the layout: Remove background and set slider color
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Bookings",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

# Footer or additional content
st.write("---")
st.write("### Additional Information")
st.write("Adjust the input values on the left sidebar to see how different factors affect airline ticket prices.")
