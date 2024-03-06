import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import requests
import re
from io import StringIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

# Functions
def send_welcome_email(email):
    # Your email credentials
    sender_email = "your_email@gmail.com"
    sender_password = "your_password"

    subject = "Welcome to Our Website!"
    message = "Hello! Welcome to our platform. We are excited to have you on board."

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        
url = 'https://github.com/dudilu/Advisor/blob/main/list_advisor.csv'
response = requests.get(url)
html = response.text

pattern = re.compile(':\\[\\[(.*?)\\]\\],')
matches = pattern.findall(html)
result = ''.join(matches)

cleaned_string = result.replace('],[', '"\n"').replace('","', '\t')

cleaned_string = '"'+cleaned_string+'"'

data = StringIO(cleaned_string)

list_advisor = pd.read_csv(data, sep='\t', header=None, skiprows=1, names=['symbol', 'publish', 'count'])

list_advisor['count'] = pd.to_numeric(list_advisor['count'].str.replace('"', ''), errors='coerce')

count_list_advisor = list_advisor

list_advisor.set_index('symbol', inplace=True)
list_advisor = list_advisor['publish'].to_dict()

filtered_dict = {}

for key, value in list_advisor.items():
    try:
        date_object = datetime.strptime(value, '%b %d, %Y')
        formatted_date = date_object.strftime('%Y-%m-%d')
        filtered_dict[key] = formatted_date
    except ValueError:
        pass

if 'GOOG' in filtered_dict:
    del filtered_dict['GOOG']

list_advisor = filtered_dict

oldest_date = min(list_advisor.values())

list_advisor['SPY'] = oldest_date

list_advisor = {symbol: pd.to_datetime(date) for symbol, date in list_advisor.items()}

list_advisor = dict(sorted(list_advisor.items(), key=lambda x: x[1]))

close_prices_dict = {}

for symbol, start_date in list_advisor.items():

    end_date = (pd.to_datetime('today') + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

    stock_data = yf.download(symbol, start=start_date + pd.DateOffset(days=4), end=end_date)

    close_prices = stock_data['Close'].to_dict()

    close_prices_dict[symbol] = close_prices

df_close_prices = pd.DataFrame(close_prices_dict)

first_values = df_close_prices.apply(lambda col: col.dropna().iloc[0])

df_change = df_close_prices.apply(lambda col: ((col - first_values[col.name]) / first_values[col.name]) * 100)

daily_report = pd.DataFrame({
    'Symbol': df_change.columns,
    '%Change': df_change.iloc[-1].values})

daily_report['Date'] = daily_report['Symbol'].map(list_advisor)

daily_report = pd.merge(daily_report, count_list_advisor, left_on='Symbol', right_on='symbol', how='left')

daily_report = daily_report[['Symbol', 'publish','%Change', 'count']]

df_change['InvesTec'] = df_change.drop('SPY', axis=1).mean(axis=1)

df_change['Day'] = df_close_prices.index

plt.plot(df_change['Day'], df_change['InvesTec'], label='InvesTec')
plt.plot(df_change['Day'], df_change['SPY'], label='SPY')

plt.xlabel('Date')
plt.ylabel('%Change')
plt.legend()
plt.grid(True)

plt.show()

filtered_df = daily_report[daily_report['Symbol'] != 'SPY']
mean_without_spy = filtered_df['%Change'].mean()
spy_percentage_change = daily_report.loc[daily_report['Symbol'] == 'SPY', '%Change'].iloc[0]

# Plotting using Matplotlib
fig, ax = plt.subplots()
ax.plot(df_change['Day'], df_change['InvesTec'], label='InvesTec')
ax.plot(df_change['Day'], df_change['SPY'], label='SPY')
ax.set_xlabel('Date')
ax.set_ylabel('%Change')
ax.legend()
ax.grid(True)

# Show the Matplotlib plot in Streamlit
st.pyplot(fig)

# Display additional information using Streamlit
st.write("Mean without SPY:", np.round(mean_without_spy, 2))
st.write("SPY Percentage Change:", np.round(spy_percentage_change, 2))

sections = ['Home', 'Plot', 'Mean without SPY', 'SPY Percentage Change']

# Create a sidebar with links to different sections
selected_section = st.sidebar.button('Home')
if selected_section:
    st.title('Home')
    st.write('Welcome to the Home section.')

selected_section = st.sidebar.button('Plot')
if selected_section:
    st.title('Plot')

    # Plotting using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(df_change['Day'], df_change['InvesTec'], label='InvesTec')
    ax.plot(df_change['Day'], df_change['SPY'], label='SPY')
    ax.set_xlabel('Date')
    ax.set_ylabel('%Change')
    ax.legend()
    ax.grid(True)

    # Show the Matplotlib plot in Streamlit
    st.pyplot(fig)

selected_section = st.sidebar.button('Mean without SPY')
if selected_section:
    st.title('Mean without SPY')
    st.title("Send Welcome Email")
    email = st.text_input("Enter your email address:")
    if st.button("Send"):
        send_welcome_email(email)
        st.success("Welcome email sent to " + email)

    # Display additional information using Streamlit
    st.write("Mean without SPY:", np.round(mean_without_spy, 2))

selected_section = st.sidebar.button('SPY Percentage Change')
if selected_section:
    st.title('SPY Percentage Change')

    # Display additional information using Streamlit
    st.write("SPY Percentage Change:", np.round(spy_percentage_change, 2))

