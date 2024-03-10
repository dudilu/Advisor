import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import math
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import re
import base64
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import urllib.request
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_username = "dudilu86@gmail.com"
smtp_password = "qhnj pmve zcpv ycds"
def display_dudi(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Remove EXIF orientation
    if hasattr(img, '_getexif'):
        orientation = 0x0112
        exif = img._getexif()
        if exif is not None:
            orientation = exif[orientation]
            rotations = {
                3: Image.ROTATE_180,
                6: Image.ROTATE_270,
                8: Image.ROTATE_90
            }
            if orientation in rotations:
                img = img.transpose(rotations[orientation])

    st.image(img, use_column_width=True)
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def display_image(url):
    with urllib.request.urlopen(url) as response:
        img = Image.open(response)
        img = np.array(img)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        st.pyplot(fig)
def calculate_cagr(df, column):
    first_value = df[column].iloc[0]
    last_value = df[column].iloc[-1]
    num_years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    cagr = (last_value / first_value) ** (1 / num_years) - 1
    return cagr * 100
def is_valid_email(email):
    """
    Check if the email address is valid.
    """
    # Regular expression for basic email validation
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email)
def send_email(receiver_email, subject, content):
    sender_email = smtp_username
    password = smtp_password

    if not is_valid_email(receiver_email):
        st.error(f"Invalid email address: {receiver_email}")
        return

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Create the HTML content of the email
    html = f"""
    <html>
    <body>
        <h2>{subject}</h2>
        <p>{content}</p>
    </body>
    </html>
    """

    message.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        if (receiver_email != "dudilu86@gmail.com"):
            st.success(f"Email sent successfully to {receiver_email}!")
    except Exception as e:
        st.error(f"Error sending email: {e}")
    finally:
        server.quit()
def return_period(selected_tab):
    if selected_tab == '1Y':
        return_period_invest = df_change1Y['InvesTec'].iloc[-1]
    if selected_tab == '1Q':
        return_period_invest = df_change1Q['InvesTec'].iloc[-1]
    if selected_tab == '0.5Y':
        return_period_invest = df_change0p5Y['InvesTec'].iloc[-1]
    return return_period_invest
def modify_symbol(row):
    if (row['days_from_buy'] > 200):
        row['symbol'] += '_1Y'
    if (row['days_from_buy'] < 200) and (row['days_from_buy'] > 120):
        row['symbol'] += '_0_5Y'
    if row['days_from_buy'] < 120:
        row['symbol'] += '_1Q'
    return row
def create_line_chart(container, df, title, chart_height=250, background_image=None, percentage=None):
    fig = px.line(df, x='date', y=df.columns[1], title=title)
    color = "green" if percentage >= 0 else "red"
    arrow_icon = "▲" if percentage >= 0 else "▼"

    fig.update_layout(
        height=chart_height,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=True, title=''),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=True, title=''),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(t=40, l=0, r=0, b=0),
        title=dict(
            text=f"{title} ({arrow_icon} {percentage:.2f}%)",
            font=dict(size=24, color=color),
            x=0.5,
            xanchor = 'center'
        )
    )
    trendline = go.Scatter(x=df['date'], y=df[df.columns[1]], mode='lines', line=dict(color=color), name='Trendline')
    fig.add_trace(trendline)

    if background_image:
        img = Image.open(background_image)
        fig.add_layout_image(dict(source=img, x=0, y=1, xref="paper", yref="paper", xanchor="left", yanchor="top", sizex=1, sizey=1))

    with container:
        st.plotly_chart(fig, use_container_width=True, key=title)
def replace_duplicate_headers(header):
    if header not in header_count:
        header_count[header] = 1
        return header
    else:
        header_count[header] += 1
        return f"{header}{header_count[header]}"
def return_plot(df):
    trace_investec = go.Scatter(x=df['Day'], y=df['InvesTec'], mode='lines',
                                name='Moolah', line=dict(color='green', width=2), opacity=0.75,
                                hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.1f}%')
    trace_spy = go.Scatter(x=df['Day'], y=df['SPY'], mode='lines', name='SPY',
                           line=dict(color='red', width=2), opacity=0.75,
                           hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.1f}%')
    fig = go.Figure(data=[trace_investec, trace_spy])

    fig.update_layout(
        hovermode='x',
        showlegend=True,
        yaxis=dict(showgrid=True),
        xaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig)
def cumulative_plot(df):
    trace_Me = go.Scatter(x=df['Date'], y=df['cumulative'], mode='lines',
                          name='Moolah', line=dict(color='green', width=3), opacity=0.75,
                          hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> $%{y:,.0f}')
    trace_SPY = go.Scatter(x=df['Date'], y=df['SPY_Price'], mode='lines',
                           name='Spy', line=dict(color='red', width=3), opacity=0.75,
                           hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> $%{y:,.0f}')
    fig = go.Figure(data=[trace_Me, trace_SPY])

    fig.update_layout(
        title="Cumulative Returns Over Time",
        hovermode='x',
        showlegend=True,
        #yaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, type='log'),  # Set y-axis to log scale
        xaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0),
        height = 450,
        title_font=dict(size=22)
    )
    st.plotly_chart(fig)
def process_dataframe(df):
    df['number of nan'] = df.shape[1] - df.isnull().sum(axis=1) - 2
    df = df[df['number of nan'] > 5]

    first_values = df.apply(lambda col: col.dropna().iloc[0])

    columns_to_exclude = ['Day']

    df[df.columns.difference(columns_to_exclude)] = df[
        df.columns.difference(columns_to_exclude)].apply(
        lambda col: ((col - first_values[col.name]) / first_values[col.name]) * 100)

    df['InvesTec'] = df.drop(['SPY', 'Day'], axis=1).mean(axis=1)

    return df
def pie_plot(df):
    industry_counts = df.groupby('industry')['symbol'].count().reset_index()

    industry_counts.columns = ['Industry', 'Count']

    df['symbols'] = df.groupby('industry')['symbol'].transform(lambda x: ', '.join(x.unique()))

    hover_texts = []
    for industry in industry_counts['Industry']:
        symbols = ", ".join(df[df['industry'] == industry]['symbols'].unique())
        hover_texts.append(f"<b>Symbols:</b> {symbols}")

    fig = px.pie(industry_counts, values='Count', names='Industry',
                 hover_data={'Industry': False, 'Count': True}, labels={'Count': 'Number of Symbols'})

    fig.update_traces(hovertemplate='<b>Industry:</b> %{label}<br><b>% of portfolio:</b> %{percent}<br>%{text}',
                      textinfo='percent', textposition='inside')

    fig.update_traces(text=hover_texts)

    st.plotly_chart(fig)
##############################################################################################################################################################################################
# Data prep
list_advisor = pd.read_csv('https://raw.githubusercontent.com/dudilu/Advisor/main/portfolio.csv')
list_advisor = list_advisor[list_advisor['Active'] == 'active']

df_pie = list_advisor
unique_symbols = list_advisor['symbol'].unique()

list_advisor = list_advisor.apply(modify_symbol, axis=1)

industry_counts = df_pie.groupby('industry')['symbol'].count().reset_index()

industry_counts.columns = ['Industry', 'Count']

df_pie['symbols'] = df_pie.groupby('industry')['symbol'].transform(lambda x: ', '.join(x.unique()))

df_pie = df_pie.apply(modify_symbol, axis=1)

[df_pie1Q, df_pie_0_5Y, df_pie1Y] = [df_pie[~df_pie['symbol'].str.contains('_0_5Y|_1Y')], df_pie[~df_pie['symbol'].str.contains('_1Y')], df_pie]
[df_pie1Q.loc[:, 'symbol'], df_pie_0_5Y.loc[:, 'symbol'], df_pie1Y.loc[:, 'symbol']] = [df_pie1Q['symbol'].str.replace('_1Q', '', regex=True), df_pie_0_5Y['symbol'].str.replace('_0_5Y|_1Q', '', regex=True), df_pie1Y['symbol'].str.replace('_0_5Y|_1Y|_1Q', '', regex=True)]

unique_symbols_pie1Y = df_pie1Y['symbol'].unique()
df_unique_symbols_pie1Y = pd.DataFrame(unique_symbols_pie1Y, columns=['symbol'])

symbol_dates_list = []
for _, row in list_advisor.iterrows():
    symbol = row['symbol']
    publish_date = row['publish']
    symbol_dates_list.append({symbol: publish_date})

list_advisor = symbol_dates_list
dates = [datetime.strptime(list(item.values())[0], '%b %d, %Y') for item in list_advisor]
oldest_date = min(dates)

new_item = {'SPY': oldest_date}
list_advisor.append(new_item)

for item in list_advisor:
    for symbol, date_str in item.items():
        item[symbol] = pd.to_datetime(date_str)

list_advisor = sorted(list_advisor, key=lambda x: list(x.values())[0])

close_prices_data = []

for advisor in list_advisor:
    symbol = list(advisor.keys())[0]
    start_date = advisor[symbol]
    symbolWO = symbol.split('_')[0]
    end_date = (pd.to_datetime('today') + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    if start_date == oldest_date:
        retries = 10
        while retries > 0:
            try:
                dates_oldest_date = yf.download(symbolWO, start=start_date + pd.DateOffset(days=35), end=end_date)
                break
            except KeyError:
                retries -= 1
        if retries == 0:
            continue
    retries = 10
    while retries > 0:
        try:
            stock_data = yf.download(symbolWO, start=start_date + pd.DateOffset(days=35), end=end_date)
            break
        except KeyError:
            retries -= 1
    if retries == 0:
        continue

    close_prices = stock_data['Close'].tolist()
    close_prices_data.append([symbol] + close_prices)

df_close_prices = pd.DataFrame(close_prices_data)
df_close_prices = df_close_prices.T

df_close_prices.columns = df_close_prices.iloc[0]
df_close_prices = df_close_prices.drop(0)

df_close_prices.index = dates_oldest_date.index

header_count = {}

df_close_prices.columns = [replace_duplicate_headers(col) for col in df_close_prices.columns]

for col in df_close_prices.columns:
    nan_values = df_close_prices[col][df_close_prices[col].isna()]
    non_nan_values = df_close_prices[col][~df_close_prices[col].isna()]
    new_column = np.concatenate([nan_values, non_nan_values])
    df_close_prices[col] = new_column

df_close_prices['Day'] = df_close_prices.index

[df_change1Q, df_change0p5Y, df_change1Y] = [df_close_prices.filter(regex='(_1Q|SPY|InvesTec|Day)'), df_close_prices.filter(regex='(1Q|_0_5Y|SPY|InvesTec|Day)'), df_close_prices]
[df_change1Q, df_change0p5Y, df_change1Y] = [process_dataframe(df_change1Q), process_dataframe(df_change0p5Y), process_dataframe(df_change1Y)]

to_drop = ['Day', 'number of nan', 'InvesTec', 'SPY']

symbols1Y = [col.split('_')[0] for col in df_change1Y.columns]

symbols1Y = [symbol for symbol in symbols1Y if symbol not in to_drop]
seen = set()
symbols1Y = [x for x in symbols1Y if not (x in seen or seen.add(x))]
df_symbols1Y = pd.DataFrame(symbols1Y, columns=['symbol'])
for symbol in symbols1Y:
    cols = [col for col in df_change1Y.columns if symbol == col.split('_')[0]]
    df_change1Y[symbol] = df_change1Y[cols].mean(axis=1)

symbols0_5Y = [col.split('_')[0] for col in df_change0p5Y.columns]

symbols0_5Y = [symbol for symbol in symbols0_5Y if symbol not in to_drop]
seen = set()
symbols0_5Y = [x for x in symbols0_5Y if not (x in seen or seen.add(x))]
df_symbols0_5Y = pd.DataFrame(symbols0_5Y, columns=['symbol'])
for symbol in symbols0_5Y:
    cols = [col for col in df_change0p5Y.columns if symbol == col.split('_')[0]]
    df_change0p5Y[symbol] = df_change0p5Y[cols].mean(axis=1)


symbols1Q = [col.split('_')[0] for col in df_change1Q.columns]

symbols1Q = [symbol for symbol in symbols1Q if symbol not in to_drop]
seen = set()
symbols1Q = [x for x in symbols1Q if not (x in seen or seen.add(x))]
df_symbols1Q = pd.DataFrame(symbols1Q, columns=['symbol'])
for symbol in symbols1Q:
    cols = [col for col in df_change1Q.columns if symbol == col.split('_')[0]]
    df_change1Q[symbol] = df_change1Q[cols].mean(axis=1)

logo_dir = 'https://raw.githubusercontent.com/dudilu/Advisor/main/DASH_canva.png'

logo_paths = {}

for symbol in unique_symbols:
    background_image_path = os.path.join(logo_dir, f'{symbol}_canva.png')
    logo_paths[symbol] = background_image_path

performance = pd.read_csv('https://raw.githubusercontent.com/dudilu/Advisor/main/cumulative_values.csv')
performance.reset_index(inplace=True)
performance['Date'] = pd.to_datetime(performance['Date'])

backtesting_over_time = pd.read_csv('https://raw.githubusercontent.com/dudilu/Advisor/main/backtesting_over_time.csv')
backtesting_over_time['date'] = pd.to_datetime(backtesting_over_time['start'], format='%Y-%m-%d')
backtesting_over_time['change[%]'] = (backtesting_over_time['change[%]']).round(2).astype(str) + '%'

##############################################################################################################################################################################################
# Settings
st.set_page_config(page_title="Moolah",layout='wide',initial_sidebar_state="auto")

with st.sidebar:
    #selected = option_menu("Main Menu", ['Our Strategic', 'Our Portfolio', 'Fundamentals', 'Strategic Performance'], icons=['briefcase', 'star', 'clock', 'question-circle'], menu_icon="cast")
    selected = option_menu("Main Menu", ['🎯 Our Strategic', '📊 Our Portfolio', '📈 Fundamentals', '🚀 Strategic Performance', '🕵️‍♂️ About'], menu_icon="cast")
    num_empty_spaces_default = 10

    if selected == "📊 Our Portfolio":
        selected_tab = st.selectbox("Select a Period", ["1Y", "0.5Y", "1Q"])
        num_empty_spaces = 20

    elif selected == "📈 Fundamentals":
        num_empty_spaces = 22

    elif selected == "🚀 Strategic Performance":
        start_date = pd.Timestamp(st.sidebar.date_input("Start date", min_value=performance['Date'].min(),max_value=performance['Date'].max(),value=performance['Date'].min()))
        end_date = pd.Timestamp(st.sidebar.date_input("End date", min_value=performance['Date'].min(),max_value=performance['Date'].max(),value=performance['Date'].max()))
        num_empty_spaces = 15

    elif selected == "🎯 Our Strategic":
        num_empty_spaces = 25
    elif selected == "🕵️‍♂️ About":
        num_empty_spaces = 25

    else:
        num_empty_spaces = num_empty_spaces_default

    for _ in range(num_empty_spaces):
        st.write("")

    user_email = st.text_input("**Stay on top of your investments! Sign up for our stock alerts.**")
    if st.button("Submit"):
        if user_email:
            send_email("dudilu86@gmail.com", "User Details", f"User Email: {user_email}")
            send_email(user_email, "Hello", "world")
        else:
            st.warning("Please enter your email address.")
##############################################################################################################################################################################################
if selected == "📊 Our Portfolio":
    col_title, col_metric = st.columns([6, 1])
    col_explain = st.columns([6, 1])
    col_return, col_pie = st.columns([1.1, 1])

    col_explain[0].markdown(
        """
        <style>
        body {
            background-color: #f0f0f0; /* Set your desired background color */
        }

        .fade-in {
            animation: fadeIn 4.5s;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    num_rows = math.ceil((df_pie1Y['symbol'].nunique()) / 4)
    rows = []
    for i in range(num_rows):
        row = st.columns(4)
        rows.append(row)

    with col_pie:
        if selected_tab == "1Q":
            pie_plot(df_pie1Q)

            with col_return:
                return_plot(df_change1Q)

            last_column = df_change1Q.loc[:, 'Day']

            for i in range(num_rows):
                for j in range(4):
                    symbol_index = 4 * i + j
                    if symbol_index < len(df_symbols1Q):
                        symbol = df_symbols1Q.loc[symbol_index, 'symbol']
                        first_column = df_change1Q.loc[:, symbol]
                        new_df = pd.DataFrame({'date': last_column, 'first': first_column})
                        background_image = logo_paths[symbol]
                        create_line_chart(rows[i][j], new_df, symbol, background_image=background_image,percentage=new_df['first'].iloc[-1])

        elif selected_tab == "0.5Y":
            pie_plot(df_pie_0_5Y)
            with col_return:
                return_plot(df_change0p5Y)

            last_column = df_change0p5Y.loc[:, 'Day']

            for i in range(num_rows):
                for j in range(4):
                    symbol_index = 4 * i + j
                    if symbol_index < len(df_symbols0_5Y):
                        symbol = df_symbols0_5Y.loc[symbol_index, 'symbol']
                        first_column = df_change0p5Y.loc[:, symbol]
                        new_df = pd.DataFrame({'date': last_column, 'first': first_column})
                        background_image = logo_paths[symbol]
                        create_line_chart(rows[i][j], new_df, symbol, background_image=background_image,percentage=new_df['first'].iloc[-1])

        elif selected_tab == "1Y":
            pie_plot(df_pie1Y)
            with col_return:
                return_plot(df_change1Y)

            last_column = df_change1Y.loc[:, 'Day']
            for i in range(num_rows):
                for j in range(4):
                    symbol_index = 4 * i + j
                    if symbol_index < len(df_symbols1Y):
                        symbol = df_symbols1Y.loc[symbol_index, 'symbol']
                        first_column = df_change1Y.loc[:, symbol]
                        new_df = pd.DataFrame({'date': last_column, 'first': first_column})
                        background_image = logo_paths[symbol]
                        create_line_chart(rows[i][j], new_df, symbol, background_image=background_image, percentage=new_df['first'].iloc[-1])

    st.markdown(
        """
        <style>
            .stPlotlyChart {
                border: 1px solid #e2e2e2;
                border-radius: 20px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    col_title.markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Our Portfolio</h1>""",unsafe_allow_html=True)

    percentage = return_period(selected_tab)
    color = "green" if percentage >= 0 else "red"
    arrow_icon = "▲" if percentage >= 0 else "▼"
    col_metric.markdown("""<div style='border: 1px solid #e2e2e2; padding: 15px; border-radius: 800px;'>
            <p><span style='color: {color}; font-size: 36px;'>{arrow_icon} {percentage:.2f}%</span></p></div>""".format(color=color, arrow_icon=arrow_icon, percentage=percentage),unsafe_allow_html=True)
##############################################################################################################################################################################################
elif selected == "🚀 Strategic Performance":
    col_title, col_metric = st.columns([6, 1])
    col_explain = st.columns(1)
    col_performance, col_bar_chart = st.columns([1.1, 1])
    col_backtesting_over_time = st.columns([1])

    col_title.markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Strategic Performance - Backtesting</h1>""",unsafe_allow_html=True)
    with col_explain[0]:
        st.markdown("<h2 style='color:#74B6FF;'>What is Backtesting</h2>", unsafe_allow_html=True)
        st.write("Backtesting is a simulation technique used to evaluate a trading strategy using historical data.")
        st.write("Essentially, it allows traders and investors to test how a particular strategy would have performed if it had been used during a specific period in the past.")

        st.markdown("<h2 style='color:#74B6FF;'>Why is Backtesting Important for Trading</h2>", unsafe_allow_html=True)

        st.write("**1. Validating Strategy Performance:**")
        st.write("   Backtesting allows traders to validate their trading strategies by objectively testing them against historical data. ")
        st.write("This helps in understanding if the strategy is robust and capable of generating profits.")

        st.write("**2. Building Confidence:**")
        st.write("   Successful backtesting results can provide traders with confidence in their strategies. ")
        st.write("It offers a level of assurance that the chosen approach has the potential to generate profits based on historical performance.")

        st.write("**3. Comparing Strategies:**")
        st.write("   Traders can compare multiple strategies using backtesting to determine which ones are more effective. ")
        st.write("This allows for data-driven decision-making in strategy selection.")

    performance = performance[(performance['Date'] >= start_date) & (performance['Date'] <= end_date)]
    backtesting_over_time = backtesting_over_time[(backtesting_over_time['date'] >= start_date) & (backtesting_over_time['date'] <= end_date)]

    years = performance['Date'].dt.year.unique()
    cagr_data = {
        'Year': years,
        'Moolah': [calculate_cagr(performance[performance['Date'].dt.year == year], 'cumulative') for year in years],
        'SPY': [calculate_cagr(performance[performance['Date'].dt.year == year], 'SPY_Price') for year in years]
    }
    df_cagr = pd.DataFrame(cagr_data)
    df_cagr['Moolah'] = df_cagr['Moolah']/100
    df_cagr['SPY'] = df_cagr['SPY']/100

    with col_performance:
        cumulative_plot(performance)

    with col_bar_chart:
        fig = px.bar(df_cagr, x='Year', y=['Moolah', 'SPY'],
                     title='Compound Annual Growth Rate per Year',
                     barmode='group',
                     color_discrete_map={'Moolah': 'green', 'SPY': 'red'},
                     height=500)

        fig.update_layout(title={'text': 'Compound Annual Growth Rate per Year', 'font': {'size': 22}})
        fig.update_xaxes(categoryorder='array', categoryarray=df_cagr['Year'], title=None)
        fig.update_yaxes(automargin=True, title=None, tickformat='.2%')
        fig.update_layout(hovermode='x unified')

        # Remove the word "variable" from legend labels
        fig.update_layout(legend_title_text='', legend=dict(itemsizing='constant'))

        fig.update_traces(hovertemplate='%{y:.2f}')

        st.plotly_chart(fig)

    backtesting_over_time = backtesting_over_time.drop(columns=['index','date'])
    with col_backtesting_over_time[0]:
        st.data_editor(
            backtesting_over_time,
            column_config={
                "start": "Date Start",
                "end": "Date End",
                "change[%]": "Return",
                "value_list": st.column_config.LineChartColumn("Price"),
            },
            hide_index=True,
            width=1250,
        )

    percentage = calculate_cagr(performance,'cumulative')
    color = "green" if percentage >= 0 else "red"
    arrow_icon = "▲" if percentage >= 0 else "▼"
    col_metric.markdown("""<div style='border: 1px solid #e2e2e2; padding: 15px; border-radius: 800px;'>
            <p><span style='color: {color}; font-size: 36px;'>{arrow_icon} {percentage:.2f}%</span></p></div>""".format(color=color, arrow_icon=arrow_icon, percentage=percentage),unsafe_allow_html=True)
##############################################################################################################################################################################################
elif selected == "📈 Fundamentals":
    col_title, col_metric = st.columns([6, 1])
    col_explain = st.columns(1)
    col_why = st.columns([1])

    why = pd.read_csv('https://raw.githubusercontent.com/dudilu/Advisor/main/WHY.csv')

    c1, c2, c3, c4 = [why.groupby('symbol')[col].agg(list).reset_index() for col in [
        'Property, Plant, And Equipment_4',
        'Research And Development Expenses',
        'Stock Based Compensation',
        'Total Non Cash Items'
    ]]

    c1['Property, Plant, And Equipment_4'] = c1['Property, Plant, And Equipment_4'].apply(lambda x: x[::-1])
    c2['Research And Development Expenses'] = c2['Research And Development Expenses'].apply(lambda x: x[::-1])
    c3['Stock Based Compensation'] = c3['Stock Based Compensation'].apply(lambda x: x[::-1])
    c4['Total Non Cash Items'] = c4['Total Non Cash Items'].apply(lambda x: x[::-1])

    merged_df = c1.merge(c2, on='symbol', how='outer') \
        .merge(c3, on='symbol', how='outer') \
        .merge(c4, on='symbol', how='outer')

    merged_df.columns = ['symbol', 'Property, Plant, And Equipment_4',
                         'Research And Development Expenses', 'Stock Based Compensation',
                         'Total Non Cash Items']

    col_title.markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Fundamentals</h1>""",unsafe_allow_html=True)

    with col_explain[0]:
        st.markdown("<h3 style='color:#74B6FF;'>Key Finance Indicators for Backtesting Trading Strategies</h3>", unsafe_allow_html=True)

        st.write("The finance indicators below play crucial roles in backtesting trading strategies and assessing companies future potential.")

        st.markdown("<h5 style='color:#74B6FF;'>1. Property, Plant, and Equipment</h5>", unsafe_allow_html=True)

        st.write("Property, Plant, and Equipment (PP&E) refers to the long-term tangible assets that a company uses in its operations to generate revenue. ")
        st.write("These can include buildings, machinery, equipment, vehicles, and land.")
        st.write("A rising trend in PP&E could suggest that the company is investing in its future, potentially leading to increased production capacity, efficiency, and competitiveness.")

        st.markdown("<h5 style='color:#74B6FF;'>2. Research and Development Expenses (R&D)</h5>", unsafe_allow_html=True)
        st.write("Research and Development Expenses (R&D) refer to the costs incurred by a company to develop new products, services, or technologies. ")
        st.write("These expenses are aimed at improving existing products or creating new ones.")
        st.write("Increasing R&D expenses might suggest potential for future revenue growth from new products or improved offerings.")

        st.markdown("<h5 style='color:#74B6FF;'>3. Stock-Based Compensation</h5>", unsafe_allow_html=True)

        st.write("Stock-Based Compensation refers to the issuance of company stock or stock options to employees as part of their compensation package. ")
        st.write("This can include stock options, restricted stock units (RSUs), or other equity-based incentives.")
        st.write("Increasing stock-based compensation could suggest confidence in the company's future stock performance.")

        st.markdown("<h5 style='color:#74B6FF;'>4. Total Non-Cash Items</h5>", unsafe_allow_html=True)

        st.write("Total Non-Cash Items refer to all non-cash transactions that impact a company's financial statements. ")
        st.write("This can include items such as depreciation, amortization, stock-based compensation, and other non-cash expenses or gains.")
        st.write("Higher total non-cash items might indicate a company's ability to manage expenses efficiently without affecting its cash position.")

    with col_why[0]:
        st.data_editor(
            merged_df,
            column_config={
                "symbol": "Symbol",
                "Property, Plant, And Equipment_4": st.column_config.BarChartColumn("Property, Plant, And Equipment"),
                "Research And Development Expenses": st.column_config.BarChartColumn("Research And Development Expenses"),
                "Stock Based Compensation": st.column_config.BarChartColumn("Stock Based Compensation"),
                "Total Non Cash Items": st.column_config.BarChartColumn("Total Non Cash Items"),
            },
            hide_index=True, height=int(np.round(37.17 * len(c1))), width=900,
        )
##############################################################################################################################################################################################
elif selected == "🎯 Our Strategic":
    col_title = st.columns(1)
    col_explain, col_img = st.columns([2, 2])
    #col_img = st.columns(2)

    col_title[0].markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Our Strategic</h1>""",unsafe_allow_html=True)



    with col_explain:
        st.markdown("<h2 style='color:#74B6FF;'>How Our Algorithm Identifies Winning Stocks</h2>", unsafe_allow_html=True)

        st.write("Our algorithm is designed to pinpoint stocks in the introduction stage of the product life cycle. "
                 "By analyzing company data and trends, we can identify emerging products with huge growth potential.")

        st.markdown("<h2 style='color:#74B6FF;'>Maximizing Growth Stage Profits</h2>", unsafe_allow_html=True)

        st.write("Once a stock is identified in the introduction stage, we ride the wave of growth. "
                 "Our algorithm ensures that we take full of the growth period, maximizing profits.")

        st.markdown("<h2 style='color:#74B6FF;'>Why It Works</h2>", unsafe_allow_html=True)

        st.write("Cutting-edge technology analyzes market signals to spot products on the cusp of explosive growth. "
                 "By getting in early, we reap the rewards as the product gains popularity. "
                 "Our track record speaks for itself.")

        st.write("Join Us in Profiting from Innovation.")

        st.write("Don't miss out on the opportunity to invest in the next big thing. "
                 "Our algorithm does the lifting, so you can enjoy the profits. "
                 "Invest confidently, knowing you're ahead of the curve with our proven approach. Start Investing Wisely Today.")

        st.write(
            "Discover how our algorithm can help you identify stocks in the introduction stage for maximum growth. "
            "Let's embark on a profitable journey together.")

    with col_img:
        display_image('https://github.com/dudilu/Advisor/raw/main/cash%20cow1.jpg')

elif selected == "🕵️‍♂️ About":
    col_title = st.columns(1)
    col_title[0].markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Hello !</h1>""",unsafe_allow_html=True)
    col_explain, col_img = st.columns([2, 2])

    with col_explain:
        st.write("I'm Dudi")
        st.write("")
        st.write("With a background in electrical engineering and a specialization in data science, I've always been fascinated by the power of algorithms to make sense of complex data.")
        st.write("After years of honing my skills in both academia and industry, I decided to combine my expertise to create something truly unique—a stock recommendation app driven by cutting-edge AI.")
        st.write("")
        st.write("Our app is the result of countless hours of research, development, and testing, all aimed at providing you with intelligent, data-driven stock suggestions. ")
        st.write("My passion for data and technology is what drives me to constantly improve our algorithms and ensure that you receive the most accurate and valuable insights.")
        st.write("")
        st.write("I invite you to join me on this exciting journey into the world of data-driven investing.")
        st.write("Together, let's explore new opportunities and make informed decisions that pave the way to financial success.")
        st.write("")
        st.write("Thank you for choosing Moolah,")
        st.write("")
        st.write("Dudi")
    with col_img:
        display_dudi('https://raw.githubusercontent.com/dudilu/Advisor/main/Dudi.jpg')


# url = 'https://raw.githubusercontent.com/dudilu/Advisor/main/list_advisor.csv'
# df = pd.read_csv(url)
#set_background('C:/Users/DudiLubton/PycharmProjects/pythonProject/Advisor/logo/plc.png')
