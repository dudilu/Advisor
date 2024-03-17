import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import base64
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import urllib.request
import requests
from io import BytesIO

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
def send_email(receiver_email, subject, content1, content2, content3, content4):
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
        <p>{content1}</p>
        <p>{content2}</p>
        <p>{content3}</p>
        <p>{content4}</p>
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
        try:
            response = requests.get(background_image)
            img = Image.open(BytesIO(response.content))
            fig.add_layout_image(
                dict(source=img, x=0, y=1, xref="paper", yref="paper", xanchor="left", yanchor="top", sizex=1, sizey=1))
        except Exception as e:
            st.warning(f"Failed to load background image: {e}")

    with container:
        st.plotly_chart(fig, use_container_width=True, key=title)
def replace_duplicate_headers(header):
    if header not in header_count:
        header_count[header] = 1
        return header
    else:
        header_count[header] += 1
        return f"{header}{header_count[header]}"
def return_plot(df, container):
    trace_investec = go.Scatter(x=df['Day'], y=df['InvesTec'], mode='lines',
                                name='Moolah', line=dict(color='green', width=2), opacity=0.75,
                                hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.1f}%')
    trace_spy = go.Scatter(x=df['Day'], y=df['SPY'], mode='lines', name='SPY',
                           line=dict(color='red', width=2), opacity=0.75,
                           hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.1f}%')
    fig = go.Figure(data=[trace_investec, trace_spy])

    fig.add_annotation(
        x=1.075,
        y=1.03 * df['InvesTec'].iloc[-1],
        xref='paper',
        yref='y',
        text=f'{df["InvesTec"].iloc[-1]:.2f}%',
        showarrow=False,
        font=dict(color='green'),
        bordercolor='gray',
        borderwidth=1,
        borderpad=1,
        bgcolor='yellow'
    )

    fig.add_annotation(
        x=1.075,
        y=1.03 * df['SPY'].iloc[-1],
        xref='paper',
        yref='y',
        text=f'{df["SPY"].iloc[-1]:.2f}%',
        showarrow=False,
        font=dict(color='red'),
        bordercolor='gray',
        borderwidth=1,
        borderpad=1,
        bgcolor='yellow'
    )

    fig.update_layout(
        hovermode='x',
        showlegend=True,
        yaxis=dict(showgrid=True),
        xaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=1, y=0, traceorder='normal', orientation='v')
    )
    with container:
        st.plotly_chart(fig, use_container_width=True)
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
        margin=dict(l=0, r=0, t=60, b=60),
        height = 500,
        title_font=dict(size=22)
    )
    st.plotly_chart(fig, use_container_width=True)
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
def pie_plot(df, container):
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
    with container:
        st.plotly_chart(fig, use_container_width=True)
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

logo_dir = 'https://raw.githubusercontent.com/dudilu/Advisor/main'

logo_paths = {}

for symbol in unique_symbols:
    background_image_path = f'{logo_dir}/{symbol}_canva.png'
    logo_paths[symbol] = background_image_path

performance = pd.read_csv('https://raw.githubusercontent.com/dudilu/Advisor/main/cumulative_values.csv')
performance.reset_index(inplace=True)
performance['Date'] = pd.to_datetime(performance['Date'])

backtesting_over_time = pd.read_csv('https://raw.githubusercontent.com/dudilu/Advisor/main/backtesting_over_time.csv')
backtesting_over_time['date'] = pd.to_datetime(backtesting_over_time['start'], format='%Y-%m-%d')
backtesting_over_time['change[%]'] = (backtesting_over_time['change[%]']).round(2).astype(str) + '%'

##############################################################################################################################################################################################
# Settings
st.set_page_config(page_title="Moo-lah!",layout='wide',initial_sidebar_state="auto", page_icon='🐄')

with st.sidebar:
    #selected = option_menu("Main Menu", ['Our Strategic', 'Our Portfolio', 'Fundamentals', 'Strategic Performance'], icons=['briefcase', 'star', 'clock', 'question-circle'], menu_icon="cast")
    selected = option_menu("Main Menu", ['🏠 Home', '📊 Our Portfolio', '📈 Fundamentals', '🚀 Strategic Performance', '🕵️‍♂️ About'], menu_icon="cast")

    if selected == "📊 Our Portfolio":
        selected_tab = st.selectbox("Select a Period", ["1Y", "0.5Y", "1Q"])

    elif selected == "🚀 Strategic Performance":
        start_date = pd.Timestamp(st.sidebar.date_input("Start date", min_value=performance['Date'].min(),max_value=performance['Date'].max(),value=performance['Date'].min()))
        end_date = pd.Timestamp(st.sidebar.date_input("End date", min_value=performance['Date'].min(),max_value=performance['Date'].max(),value=performance['Date'].max()))

    user_email = st.text_input("**Stay on top of your investments! Sign up for our stock alerts.**")
    if st.button("Submit"):
        if user_email:
            send_email("dudilu86@gmail.com", "User Details", f"User Email: {user_email}", "Content 2","Content 3", "Content 4")
            send_email(user_email, "Welcome, Trailblazers! 🌟", "Meet our profit rockstars, the cash cows,","Our algorithm? A cool crystal ball,","Surf growth waves like pros, join our vibe,","Plus, dairy cow updates in our witty tribe! 🐄")
        else:
            st.warning("Please enter your email address.")
##############################################################################################################################################################################################
if selected == "📊 Our Portfolio":

        if selected_tab == "1Q":
            num_rows = math.ceil((df_pie1Q['symbol'].nunique()) / 4)
            rows_1Q = []
            for i in range(num_rows + 3):
                if (i == 0):
                    row = st.columns(5)
                    rows_1Q.append(row)
                if (i == 1):
                    row = st.columns(1)
                    rows_1Q.append(row)
                if (i == 2):
                    row = st.columns(2)
                    rows_1Q.append(row)
                if (i > 2):
                    row = st.columns(4)
                    rows_1Q.append(row)

            pie_plot(df_pie1Q, rows_1Q[2][1])

            return_plot(df_change1Q, rows_1Q[2][0])

            last_column = df_change1Q.loc[:, 'Day']
            for i in range(3, num_rows + 3):
                for j in range(4):
                    symbol_index = 4 * i + j - 12
                    if symbol_index < len(df_symbols1Q):
                        symbol = df_symbols1Q.loc[symbol_index, 'symbol']
                        first_column = df_change1Q.loc[:, symbol]
                        new_df = pd.DataFrame({'date': last_column, 'first': first_column})
                        background_image = logo_paths[symbol]
                        create_line_chart(rows_1Q[i][j], new_df, symbol, background_image=background_image,
                                          percentage=new_df['first'].iloc[-1])

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
            with rows_1Q[1][0]:
                st.write("**Welcome to our eclectic collection of investment gems!**")
                st.write(
                    "**Each pick is thoughtfully selected with care and a hint of sophistication, ensuring your portfolio sparkles like a prized piece of fine art.**")

            with rows_1Q[0][0]:
                rows_1Q[0][0].markdown(
                    """<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Our Portfolio</h1>""",
                    unsafe_allow_html=True)
                rows_1Q[0][0].markdown("""<h1> </h1>""", unsafe_allow_html=True)

            percentage = return_period(selected_tab)
            color = "green" if percentage >= 0 else "red"
            arrow_icon = "▲" if percentage >= 0 else "▼"

            with rows_1Q[0][4]:
                rows_1Q[0][4].markdown("""
                            <div style='border: 1px solid #e2e2e2; padding: 3px; border-radius: 800px; text-align: center;'>
                                <p><span style='color: {color}; font-size: 36px;'>{arrow_icon} {percentage:.2f}%</span></p>
                            </div> """.format(color=color, arrow_icon=arrow_icon, percentage=percentage),
                                       unsafe_allow_html=True)

        elif selected_tab == "0.5Y":
            num_rows = math.ceil((df_pie_0_5Y['symbol'].nunique()) / 4)
            rows_0_5Y = []
            for i in range(num_rows + 3):
                if (i == 0):
                    row = st.columns(5)
                    rows_0_5Y.append(row)
                if (i == 1):
                    row = st.columns(1)
                    rows_0_5Y.append(row)
                if (i == 2):
                    row = st.columns(2)
                    rows_0_5Y.append(row)
                if (i > 2):
                    row = st.columns(4)
                    rows_0_5Y.append(row)

            pie_plot(df_pie_0_5Y, rows_0_5Y[2][1])

            return_plot(df_change0p5Y, rows_0_5Y[2][0])

            last_column = df_change0p5Y.loc[:, 'Day']
            for i in range(3, num_rows + 3):
                for j in range(4):
                    symbol_index = 4 * i + j - 12
                    if symbol_index < len(df_symbols0_5Y):
                        symbol = df_symbols0_5Y.loc[symbol_index, 'symbol']
                        first_column = df_change0p5Y.loc[:, symbol]
                        new_df = pd.DataFrame({'date': last_column, 'first': first_column})
                        background_image = logo_paths[symbol]
                        create_line_chart(rows_0_5Y[i][j], new_df, symbol, background_image=background_image,
                                          percentage=new_df['first'].iloc[-1])

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
            with rows_0_5Y[1][0]:
                st.write("**Welcome to our eclectic collection of investment gems!**")
                st.write(
                    "**Each pick is hand-curated with love and a sprinkle of flair, ensuring your portfolio is as cool as a vintage vinyl record.**")

            with rows_0_5Y[0][0]:
                rows_0_5Y[0][0].markdown(
                    """<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Our Portfolio</h1>""",
                    unsafe_allow_html=True)
                rows_0_5Y[0][0].markdown("""<h1> </h1>""", unsafe_allow_html=True)

            percentage = return_period(selected_tab)
            color = "green" if percentage >= 0 else "red"
            arrow_icon = "▲" if percentage >= 0 else "▼"

            with rows_0_5Y[0][4]:
                rows_0_5Y[0][4].markdown("""
                            <div style='border: 1px solid #e2e2e2; padding: 3px; border-radius: 800px; text-align: center;'>
                                <p><span style='color: {color}; font-size: 36px;'>{arrow_icon} {percentage:.2f}%</span></p>
                            </div> """.format(color=color, arrow_icon=arrow_icon, percentage=percentage),
                                       unsafe_allow_html=True)

        elif selected_tab == "1Y":
            num_rows = math.ceil((df_pie1Y['symbol'].nunique()) / 4)
            rows_1Y = []
            for i in range(num_rows + 3):
                if (i == 0):
                    row = st.columns(5)
                    rows_1Y.append(row)
                if (i == 1):
                    row = st.columns(1)
                    rows_1Y.append(row)
                if (i == 2):
                    row = st.columns(2)
                    rows_1Y.append(row)
                if (i > 2):
                    row = st.columns(4)
                    rows_1Y.append(row)

            pie_plot(df_pie1Y, rows_1Y[2][1])

            return_plot(df_change1Y,rows_1Y[2][0])

            last_column = df_change1Y.loc[:, 'Day']
            for i in range(3, num_rows + 3):
                for j in range(4):
                    symbol_index = 4 * i + j - 12
                    if symbol_index < len(df_symbols1Y):
                        symbol = df_symbols1Y.loc[symbol_index, 'symbol']
                        first_column = df_change1Y.loc[:, symbol]
                        new_df = pd.DataFrame({'date': last_column, 'first': first_column})
                        background_image = logo_paths[symbol]
                        create_line_chart(rows_1Y[i][j], new_df, symbol, background_image=background_image, percentage=new_df['first'].iloc[-1])

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
            with rows_1Y[1][0]:
                st.write("**Welcome to our eclectic collection of investment gems!**")
                st.write("**Each pick is hand-curated with love and a sprinkle of flair, ensuring your portfolio is as cool as a vintage vinyl record.**")

            with rows_1Y[0][0]:
                rows_1Y[0][0].markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Our Portfolio</h1>""",unsafe_allow_html=True)
                rows_1Y[0][0].markdown("""<h1> </h1>""",unsafe_allow_html=True)

            percentage = return_period(selected_tab)
            color = "green" if percentage >= 0 else "red"
            arrow_icon = "▲" if percentage >= 0 else "▼"

            with rows_1Y[0][4]:
                rows_1Y[0][4].markdown("""
                <div style='border: 1px solid #e2e2e2; padding: 3px; border-radius: 800px; text-align: center;'>
                    <p><span style='color: {color}; font-size: 36px;'>{arrow_icon} {percentage:.2f}%</span></p>
                </div> """.format(color=color, arrow_icon=arrow_icon, percentage=percentage), unsafe_allow_html=True)
##############################################################################################################################################################################################
elif selected == "🚀 Strategic Performance":
    rows = [st.columns(4), st.columns(1), st.columns(2), st.columns(1)]

    with rows[0][0]:
        rows[0][0].markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Strategic Performance</h1>""",unsafe_allow_html=True)

    with rows[1][0]:
        st.write("")
        st.write("Alright, let's dive into the history of Strategic Performance of the algo. It's like taking your trading strategy for a spin in a time machine called Backtesting.")
        st.write("Backtesting, is basically a way to throw your trading strategy back in time and see how it would've grooved with historical data.")
        st.write("It's like saying, Hey, strategy, let's see if you would've rocked it back in the '90s!")
        st.markdown("<h5 style='color:#74B6FF;'>Why should you care about Backtesting in your trading journey? Well, for starters</h5>", unsafe_allow_html=True)
        st.write("**1. Validating Strategy Performance:**")
        st.write("Backtesting lets you put your strategy through the historical ringer. It's like giving it an old-school test drive to see if it's got the moves to make you some serious coin.")
        st.write("**2. Boosting Confidence:**")
        st.write("When those backtesting results come back positive, it gives you confidence!")
        st.write("Picture yourself strolling down Wall Street, fully aware that your strategy boasts the historical prowess to make those profits harmonize.")
        st.write("**3. Comparing Strategies:**")
        st.write("Just like curating the perfect retro outfit, traders can use backtesting to compare different strategies.")
        st.write("It's all about finding that ideal match for your trading style. You'll be making data-driven decisions akin to a seasoned collector, selecting the strategy that hits all the right notes for success.")
        st.write("")
        st.write("")

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

    with rows[2][0]:
        cumulative_plot(performance)



    with rows[2][1]:
        fig = px.bar(df_cagr, x='Year', y=['Moolah', 'SPY'],
                     title='Compound Annual Growth Rate per Year',
                     barmode='group',
                     color_discrete_map={'Moolah': 'green', 'SPY': 'red'},
                     height=500)

        fig.update_layout(title={'text': 'Compound Annual Growth Rate per Year', 'font': {'size': 22}})
        fig.update_xaxes(categoryorder='array', categoryarray=df_cagr['Year'], title=None)
        fig.update_yaxes(automargin=True, title=None, tickformat='.2%')
        fig.update_layout(hovermode='x unified')

        fig.update_layout(legend_title_text='', legend=dict(itemsizing='constant'))

        fig.update_traces(hovertemplate='%{y:.2f}')

        st.plotly_chart(fig, use_container_width=True)

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

    backtesting_over_time = backtesting_over_time.drop(columns=['index','date'])
    with rows[3][0]:
        st.data_editor(
            backtesting_over_time,
            column_config={
                "start": "Date Start",
                "end": "Date End",
                "change[%]": "Return",
                "value_list": st.column_config.LineChartColumn("Price"),
            },
            hide_index=True, use_container_width=True
        )

    percentage = calculate_cagr(performance,'cumulative')
    color = "green" if percentage >= 0 else "red"
    arrow_icon = "▲" if percentage >= 0 else "▼"

    with rows[0][3]:
        rows[0][3].markdown("""
        <div style='border: 1px solid #e2e2e2; padding: 3px; border-radius: 800px; text-align: center;'>
            <p><span style='color: {color}; font-size: 36px;'>{arrow_icon} {percentage:.2f}%</span></p>
        </div> """.format(color=color, arrow_icon=arrow_icon, percentage=percentage), unsafe_allow_html=True)
##############################################################################################################################################################################################
elif selected == "📈 Fundamentals":
    rows = [st.columns(1),st.columns(1),st.columns(2),st.columns(2),st.columns(1)]

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

    merged_df.columns = ['symbol', 'Property, Plant, And Equipment_4','Research And Development Expenses', 'Stock Based Compensation','Total Non Cash Items']

    with rows[0][0]:
        rows[0][0].markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Financial Strategic Algo Metrics</h1>""",unsafe_allow_html=True)

    with rows[1][0]:
        st.write("**Alright, fellow financial aficionados, let's dive into the groovy world of Key Finance Indicators for our rad algorithm.**")
        st.write("**These metrics are like the vinyl records of our strategy—essential, timeless, and oh-so-trendy.**")
        st.write("")
    with rows[2][0]:
        st.markdown("<div class='bordered'><h5 style='color:#74B6FF;'>Property, Plant, and Equipment</h5>"
                    "<p>Picture this: buildings, machinery, equipment, vehicles, and land—these are the tangible treasures of a company.</p>"
                    "<p>A growing trend in PP&E ?</p>"
                    "<p>It suggests the company is gearing up for the future, all set to excel with increased production capacity, efficiency, and competitiveness across the board.</p>"
                    "</div>", unsafe_allow_html=True)
    with rows[2][1]:
        st.markdown("<div class='bordered'><h5 style='color:#74B6FF;'>Research and Development Expenses (R&D)</h5>"
                    "<p>Ah, the creative side of the financial stage. R&D expenses are the company's investment whether they be products, services, or technological marvels.—products, services, or tech wonders.</p>"
                    "<p>More investment in R&D ?</p>"
                    "<p>Get ready for potential revenue growth, new products, and offerings that captivate the customers</p>"
                    "</div>", unsafe_allow_html=True)
    with rows[3][0]:
        st.markdown("<div class='bordered'><h5 style='color:#74B6FF;'>Stock-Based Compensation</h5>"
                    "<p>Stock-based compensation are like distributing passports to the team, granting them access to company shares or options.</p>"
                    "<p>It's the company's way of saying, Hey, we're paving the way for exciting journeys ahead!</p>"
                    "<p>So, if you observe a surge in stock incentives, it might signify a strong belief in the promising destinations the company is headed towards.It signal a blossoming confidence in the growth and economic prosperity.</p>"
                    "</div>", unsafe_allow_html=True)

    with rows[3][1]:
        st.markdown("<div class='bordered'><h5 style='color:#74B6FF;'>Total Non-Cash Items</h5>"
                    "<p>Ah, the tale of non-cash items!</p>"
                    "<p>Is it a waltz with depreciation, more stock-based or a flirt with strategic decisions? Either way looks like the company's gearing up for growth, paving the way for future profits.</p>"
                    "</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            .bordered {
                border: 1px solid #e2e2e2;
                border-radius: 20px;
                overflow: hidden;
                padding: 15px;
                margin-bottom: 15px;
                #height: 250px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with rows[4][0]:
        st.data_editor(
            merged_df,
            column_config={
                "symbol": "Symbol",
                "Property, Plant, And Equipment_4": st.column_config.LineChartColumn("Property, Plant, And Equipment"),
                "Research And Development Expenses": st.column_config.LineChartColumn("Research And Development Expenses"),
                "Stock Based Compensation": st.column_config.LineChartColumn("Stock Based Compensation"),
                "Total Non Cash Items": st.column_config.LineChartColumn("Total Non Cash Items"),
            },
             height=int(np.round(37.17 * len(c1))),use_container_width=True
        )
##############################################################################################################################################################################################
elif selected == "🎯 Home":
    rows = [st.columns(1),st.columns(2),st.columns(2)]

    with rows[0][0]:
        css = """
        <style>
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animated {
          animation: fadeInUp 4s ease;
        }
        </style>
        """

        # Add CSS to Streamlit
        st.markdown(css, unsafe_allow_html=True)

        # Animated title
        title_html = """
        <div class="animated">
          <h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Moo-lah :</h1>
          <h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Where a Magical Algorithm Meets Market!</h1>
        </div>
        """

        st.markdown(title_html, unsafe_allow_html=True)

    with rows[1][0]:
        st.markdown("<h2 style='color:#74B6FF;'>What's the Deal with Cash Cows?</h2>", unsafe_allow_html=True)
        st.write("Cash cows are like the rockstars of the market scene. They're the top dogs, the ones with mad positive cash flow vibes, and returns that outshine market growth rates. These bad boys keep on bringing in the dough long after the initial investment's dust has settled. Talk about moo-lah magic! 🐄✨")
        st.markdown("<h2 style='color:#74B6FF;'>Why Our Algorithm Nails Winning Stocks?</h2>", unsafe_allow_html=True)
        st.write("Our algorithm? Oh, it's like having a crystal ball for spotting stocks in their cool, emerging phase. We're all about that cutting-edge, trend-surfing life.")
        st.write("")
        st.markdown("<h5 style='color:#74B6FF;'>Riding the Growth Wave</h5>", unsafe_allow_html=True)
        st.write("Once we've snatched up a stock in its prime intro phase, we're catching that growth wave like seasoned surfers. Our algorithm? It's our secret sauce for maximizing those sweet, sweet profits. 🏄‍♂️💸")
        st.markdown("<h2 style='color:#74B6FF;'>Why We're the Pinnacle of Innovation?</h2>", unsafe_allow_html=True)
        st.write("Our technology is so cutting-edge, it practically exudes the scent of innovation. We thrive on seizing opportunities early, capitalizing on the buzz surrounding a product as it gains traction. Doubt our prowess? Take a glance at our track record—it's hotter than a cup of artisanal coffee. ☕💼")
        st.write("")
        st.write("")
        st.write("")
        st.write("Get on Board the Profit Express!")
        st.write("Don't be the one left sipping last year's brew. Get in on the ground floor of the next big thing with us. Invest with swagger, knowing you're way ahead of the curve. Ready to dive into the profit pool? Let's rock this investing gig together! 🚀💰")

    with rows[1][1]:
        display_image('https://github.com/dudilu/Advisor/raw/main/cash%20cow1.jpg')
##############################################################################################################################################################################################
elif selected == "🕵️‍♂️ About":
    rows = [st.columns(1),st.columns(2)]
    with rows[0][0]:
        rows[0][0].markdown("""<h1 style='text-align: left; color: #49bd7a; font-weight: bold;'>Hello !</h1>""",unsafe_allow_html=True)
        rows[0][0].markdown("")
        rows[0][0].markdown("")

    with rows[1][0]:
        st.write("I'm Dudi")
        st.write("")
        st.write("The self-proclaimed number whisperer with a knack for dissecting mind-boggling data making numbers do the cha-cha.")
        st.write("After years of being the data nerd in both the ivory towers of academia and the soul-sucking depths of the industry, I thought, Why not merge my powers to create something... something... uh, unique?")
        st.write("")
        st.write("So, voila! We present to you our glorious creation: a stock recommendation app that's basically your financial fairy godmother, but with more lines of code.")
        st.write("This app? Oh, it's been through more makeovers than a Kardashian. Countless hours of research, testing, and sacrificing the occasional laptop to the algorithm gods have gone into this bad boy.")
        st.write("")
        st.write("I live for this stuff, folks. I mean, if I'm not neck-deep in data, am I even breathing? Improving algorithms is my jam—like a mad scientist but with fewer explosions (mostly).")
        st.write("Join me, won't you? Let's dive headfirst into the deep, dark abyss of data-driven investing. Because who needs sleep when you're riding the rollercoaster of financial success?")
        st.write("")
        st.write("Cheers to choosing Moolah,")
        st.write("")
        st.write("Dudi")
    with rows[1][1]:
        display_dudi('https://raw.githubusercontent.com/dudilu/Advisor/main/Dudi.jpg')

#set_background('C:/Users/DudiLubton/PycharmProjects/pythonProject/Advisor/logo/plc.png')
