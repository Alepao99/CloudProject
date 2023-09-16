import joblib
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

#model=joblib.load('TemperatureMLModel.joblib')
st.set_page_config(page_title="Temperature Predictor",page_icon=":bar_chart:")
st.title('Welcome to Temperature Predictor!')


with st.container():
    st.header('How we predict ?')
    st.write('We use a Machine Learning Algorithm called Neural Prophet to learn from over 2,200,000 data entries'
             ' collected over 100 years across 95 cities and 49 countries to create a complex and accurate model to predict the temperature of any'
             ' city at any given day of the year. In the first version of the app we predict the temperature of 3 African countries, 3 European countries and 3 Asian countries ')


with st.container():
    st.write('---')
    st.header('Select from Options below')
    city = st.selectbox('Select the City (Start Typing to Search)',['Africa','Nairobi', 'Um', 'Dakar', 'Madrid', 'Europe', 'Rome',
    'Paris', 'Asia', 'Singapore', 'Bangkok','Bombay'])
    citydict={'Nairobi': 'africa_nairobi_forecasting.pkl', 'Um': 'africa_um_forecasting.pkl', 'Dakar': 'africa_dakar_forecasting.pkl', 
    'Africa': 'africa_forecasting.pkl', 'Madrid': 'europe_madrid_forecasting.pkl', 'Rome': 'europe_rome_forecasting.pkl', 
    'Paris': 'europe_paris_forecasting.pkl', 'Europe': 'europe_forecasting.pkl', 'Asia': 'asia_forecasting.pkl','Singapore': 'asia_singapore_forecasting.pkl','Bangkok': 'asia_bangkok_forecasting.pkl',
    'Bombay': 'asia_bombay_forecasting.pkl'}
    forecast_city=citydict[city]

    month = st.select_slider('Select Month',['Jan','Feb','March','April','May','June','July','Aug','Sep','Oct','Nov','Dec'])
    monthdict={'Jan':1,'Feb':2,'March':3,'April':4,'May':5,'June':6,'July':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    m=monthdict[month]

    #d = st.slider('Select Day', 1, 31)

    today = datetime.date.today()
    y = today.year

    #year = st.select_slider('Select Year',[y,y+1,y+2,y+3,y+4,y+5,y+6,y+7,y+8,y+9,y+10,y+11,y+12,y+13,y+14,y+15,y+16,y+17,y+18,y+19,y+20,y+21,y+22,y+23,y+24,y+25,y+26,y+27,y+28])
    year = st.slider('Select year', y, y + 10)
    data = pd.read_pickle('forecast/' + forecast_city)
    data["Date"] = pd.to_datetime(data['ds'])
    data.set_index(data['Date'], inplace = True)

with st.container():
    st.header('Average temperature original')
    original = data['1995':'2020']
    cols = ['Date', 'ds', 'yhat1', 'season_weekly', 'season_daily']
    original.drop(cols, axis = 1, inplace = True)
    yearly = original['y'].rolling(window = 12).mean()
    fiveyearly = original['y'].rolling(window = 60).mean()
    MAax = yearly['1995':].plot(figsize = (30, 6), label = '12-Month Moving Average')
    fiveyearly['1995':].plot(ax = MAax, color = 'red', label = '5-Year Moving Average')
    plt.xlabel('Date', fontsize = 14)
    plt.ylabel('Temperature', fontsize = 14)
    plt.legend()
    st.pyplot(plt)


with st.container():
    st.header('Average temperature prediction')
    end_date = str(year) + '-' + str(m) + '-1'
    prediction = data['2022':end_date]
    preidction = prediction.dropna()
    cols = ['Date', 'ds', 'y', 'season_weekly', 'season_daily']
    prediction.drop(cols, axis = 1, inplace = True)
    prediction.plot(figsize = (30, 6), legend = None)
    yearly = prediction['yhat1'].rolling(window = 12).mean()
    MAax = yearly['2021':].plot(figsize = (30, 6), label = '12-Month Moving Average', color= 'r')
    plt.xlabel('Date', fontsize = 14)
    plt.ylabel('Temperature', fontsize = 14)
    plt.legend()
    st.pyplot(plt)