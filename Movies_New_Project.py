import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from datetime import datetime
from collections import Counter


# Download nltk stopwords if not already available
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

st.set_page_config(page_title="Netflix Analytics Dashboard", layout="wide")
st.title("Netflix Analytics Dashboard")

# ===== NETFLIX COLOR PALETTE =====
COLORS = {
    'primary': '#E50914',      # Netflix Red
    'secondary': '#221F1F',    # Netflix Black
    'accent': '#F5F5F1',       # Netflix White
    'neutral': '#B81D24',      # Darker Red
    'purple': '#E87C03',       # Netflix Orange
    'pink': '#5A5A5A',         # Netflix Gray
    'cyan': '#1CE783',         # Netflix Green
    'background': '#000000',   # Pure Black
}

PLAN_COLORS = {
    'Basic': '#E50914',        # Netflix Red
    'Standard': '#B81D24',     # Darker Red
    'Premium': '#F5F5F1',      # Netflix White
    'Family': '#E87C03',       # Netflix Orange
}

CATEGORY_COLORS = ['#E50914', '#B81D24', '#F5F5F1', '#E87C03', '#5A5A5A', '#1CE783']

@st.cache_data
def load_data():
    conn = sqlite3.connect('movies_project.db')
    
    df = pd.read_sql("""
        SELECT
            u.user_id,
            u.age,
            u.country,
            u.monthly_spend,
            u.subscription_plan,
            m.movie_id,
            m.title,
            m.genre_primary AS genre,
            m.imdb_rating,
            m.release_year,
            wh.watch_duration_minutes,
            wh.watch_date
        FROM Movies m
        LEFT JOIN Watch_History wh ON m.movie_id = wh.movie_id
        LEFT JOIN Users u ON u.user_id = wh.user_id
        WHERE wh.watch_date >= '2020-01-01'
    """, conn)
    
    df_reviews = pd.read_sql("SELECT movie_id, sentiment, sentiment_score, review_text FROM Reviews", conn)
    df = df.merge(df_reviews, on='movie_id', how='left')
    
    df_search = pd.read_sql("SELECT user_id, search_query, search_date FROM Search_Logs WHERE search_date >= '2020-01-01'", conn)
    df_search = df_search.drop_duplicates(subset='user_id', keep='last')
    df = df.merge(df_search, on='user_id', how='left')
    
    df_reco = pd.read_sql("SELECT user_id, recommendation_type, was_clicked FROM Recommendation_Logs WHERE recommendation_date >= '2020-01-01'", conn)
    df_reco = df_reco.drop_duplicates(subset='user_id', keep='last')
    df = df.merge(df_reco, on='user_id', how='left')
    
    conn.close()
    
    # Data cleaning
    df['monthly_spend'] = df.groupby('subscription_plan')['monthly_spend'].transform(lambda x: x.fillna(x.median()))
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df.loc[(df['age'] < 13) | (df['age'] > 100), 'age'] = np.nan
    df['age'] = df.groupby(['country', 'subscription_plan'])['age'].transform(lambda x: x.fillna(x.median()))
    df['watch_duration_minutes'] = df.groupby(['age', 'subscription_plan'])['watch_duration_minutes'].transform(lambda x: x.fillna(x.median()))
    
    genre_avg = df.groupby('genre')['imdb_rating'].mean()
    df['imdb_rating'] = df['imdb_rating'].fillna(df['genre'].map(genre_avg))
    year_avg = df.groupby('release_year')['imdb_rating'].mean()
    df['imdb_rating'] = df['imdb_rating'].fillna(df['release_year'].map(year_avg))
    
    df['search_query'] = df['search_query'].fillna('no_search')
    df['search_date'] = pd.to_datetime(df['search_date'], errors='coerce')
    
    user_mode = df.groupby('user_id')['recommendation_type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df['recommendation_type'] = df.apply(lambda row: user_mode[row['user_id']] if pd.isna(row['recommendation_type']) else row['recommendation_type'], axis=1)
    most_frequent = df['recommendation_type'].mode()[0]
    df['recommendation_type'] = df['recommendation_type'].fillna(most_frequent)
    
    df['was_clicked'] = df['was_clicked'].map({1.0: True, 0.0: False})
    df['was_clicked'] = df['was_clicked'].fillna(False)
    
    movie_avg = df.groupby('movie_id')['sentiment_score'].mean()
    df['sentiment_score'] = df['sentiment_score'].fillna(df['movie_id'].map(movie_avg))
    genre_avg_sentiment = df.groupby('genre')['sentiment_score'].mean()
    df['sentiment_score'] = df['sentiment_score'].fillna(df['genre'].map(genre_avg_sentiment))
    
    df['watch_date'] = pd.to_datetime(df['watch_date'])
    
    return df

df_main = load_data()

# Sidebar Filters
st.sidebar.header("Dashboard Filters")

min_date = df_main['watch_date'].min().date()
max_date = df_main['watch_date'].max().date()
date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

countries = ['All'] + sorted(df_main['country'].unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

plans = ['All'] + sorted(df_main['subscription_plan'].unique().tolist())
selected_plan = st.sidebar.selectbox("Select Plan", plans)

min_age = int(df_main['age'].min())
max_age = int(df_main['age'].max())
age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# Multiple genre selection
genres = ['All'] + sorted(df_main['genre'].unique().tolist())
selected_genres = st.sidebar.multiselect(
    "Select Genres", 
    options=genres,
    default=['All']
)

# Apply Filters
filtered_df = df_main.copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['watch_date'].dt.date >= start_date) & (filtered_df['watch_date'].dt.date <= end_date)]

if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['country'] == selected_country]

if selected_plan != 'All':
    filtered_df = filtered_df[filtered_df['subscription_plan'] == selected_plan]

if 'All' not in selected_genres and selected_genres:
    filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]

filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]

# Dashboard Content
st.success(f"Data loaded: {len(filtered_df):,} records | {filtered_df['user_id'].nunique():,} unique users")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(filtered_df))
with col2:
    st.metric("Unique Users", filtered_df['user_id'].nunique())
with col3:
    st.metric("Avg Monthly Spend", f"${filtered_df['monthly_spend'].mean():.2f}")
with col4:
    st.metric("Avg Watch Duration", f"{filtered_df['watch_duration_minutes'].mean():.1f} min")

# Section 1: User Demographics
st.header("User Demographics")
unique_users = filtered_df.drop_duplicates(subset='user_id')

col1, col2 = st.columns(2)

with col1:
    #st.subheader("User Age Distribution")
    
    from scipy import stats
    age_data = unique_users['age'].dropna()
    kde = stats.gaussian_kde(age_data)
    x_range = np.linspace(age_data.min(), age_data.max(), 100)
    y_kde = kde(x_range)
    
    fig1 = px.histogram(unique_users, x='age', nbins=15, 
                       title="User Age Distribution",
                       color_discrete_sequence=[COLORS['primary']],
                       opacity=0.8)
    
    fig1.add_scatter(x=x_range, 
                    y=y_kde * len(age_data) * (age_data.max() - age_data.min()) / 15,
                    mode='lines', 
                    line=dict(color=COLORS['accent'], width=3),
                    name='Density', 
                    hovertemplate="<b>Age:</b> %{x}<br><b>Density:</b> %{y:.3f}")
    
    fig1.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Users",
        showlegend=True,
        template="plotly_dark",
        font=dict(color=COLORS['accent']),
	bargap=0.1
    )
    fig1.update_traces(hovertemplate="<b>Age Range:</b> %{x}<br><b>Users:</b> %{y}",
                  marker_line_color='rgba(0,0,0,0)',  # Transparent
                  marker_line_width=1.5)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    #st.subheader("Users per Subscription Plan")
    plan_counts = unique_users['subscription_plan'].value_counts().reset_index()
    plan_counts.columns = ['subscription_plan', 'count']
    
    fig2 = px.bar(plan_counts, x='subscription_plan', y='count',
                 title="Users per Subscription Plan",
                 color='subscription_plan',
                 color_discrete_map=PLAN_COLORS)
    fig2.update_layout(
        xaxis_title="Plan Type",
        yaxis_title="Number of Users",
        showlegend=False,
        template="plotly_dark",  #Dark theme
        font=dict(color=COLORS['accent'])
    )
    fig2.update_traces(hovertemplate="<b>Plan:</b> %{x}<br><b>Users:</b> %{y}")
    st.plotly_chart(fig2, use_container_width=True)

#st.subheader("Average Monthly Spend per Subscription Plan")
avg_spend_per_plan = unique_users.groupby('subscription_plan')['monthly_spend'].mean().reset_index()

fig3 = px.bar(avg_spend_per_plan, x='subscription_plan', y='monthly_spend',
             title="Average Monthly Spend per Subscription Plan",
             color='subscription_plan',
             color_discrete_map=PLAN_COLORS)
fig3.update_layout(
    xaxis_title="Plan Type",
    yaxis_title="Average Monthly Spend ($)",
    showlegend=False,
    template="plotly_dark",  #Dark theme
    font=dict(color=COLORS['accent'])
)
fig3.update_traces(hovertemplate="<b>Plan:</b> %{x}<br><b>Avg Spend:</b> $%{y:.2f}")
st.plotly_chart(fig3, use_container_width=True)


# Section 2: Customer Behaviour & Spending
st.header("Customer Behaviour & Spending")

user_viewing = (
    filtered_df.groupby('user_id')
    .agg({
        'watch_duration_minutes': 'mean',
        'monthly_spend': 'mean',
        'subscription_plan': 'first'
    })
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    #st.subheader("Watch Duration vs Monthly Spend")
    fig4 = px.scatter(user_viewing, x='watch_duration_minutes', y='monthly_spend',
                     trendline="ols",
                     title="Watch Duration vs Monthly Spend",
                     color='subscription_plan',
                     color_discrete_map=PLAN_COLORS,
                     opacity=0.7)
    fig4.update_layout(
        xaxis_title="Average Watch Duration (minutes)",
        yaxis_title="Monthly Spend ($)",
        template="plotly_dark",  # ← Beautiful dark theme
        font=dict(color=COLORS['accent'])
    )
    fig4.update_traces(hovertemplate="<b>Duration:</b> %{x} min<br><b>Spend:</b> $%{y:.2f}<br><b>Plan:</b> %{marker.color}")
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    #st.subheader("Watch Duration by Subscription Plan")
    fig5 = px.box(user_viewing, 
                 x='subscription_plan', 
                 y='watch_duration_minutes',
                 color='subscription_plan',
                 color_discrete_map=PLAN_COLORS,
                 title="Watch Duration by Subscription Plan")
    fig5.update_layout(
        xaxis_title="Plan Type",
        yaxis_title="Average Watch Duration (minutes)", 
        template="plotly_dark",
        font=dict(color=COLORS['accent']),
        showlegend=False
    )
    fig5.update_traces(hovertemplate="<b>Plan:</b> %{x}<br><b>Duration:</b> %{y:.1f} min")
    st.plotly_chart(fig5, use_container_width=True)

#st.subheader("Average Monthly Spend by Subscription Plan")
pivot_plan = (
    filtered_df.drop_duplicates(subset='user_id')
    .pivot_table(index='subscription_plan', values='monthly_spend', aggfunc='mean')
)

heatmap_data = pivot_plan.values
plan_names = pivot_plan.index.tolist()

fig6 = px.imshow(heatmap_data,
                x=['Monthly Spend'],
                y=plan_names,
                color_continuous_scale=[COLORS['secondary'], COLORS['neutral'], COLORS['primary']],
                aspect="auto")

fig6.update_layout(
    title="Average Monthly Spend by Subscription Plan",
    xaxis_title="",
    yaxis_title="Subscription Plan",
    template="plotly_dark",  # ← Beautiful dark theme
    font=dict(color=COLORS['accent'])
)

for i in range(len(plan_names)):
    for j in range(heatmap_data.shape[1]):
        fig6.add_annotation(
            x=j, y=i,
            text=f"${heatmap_data[i, j]:.2f}",
            showarrow=False,
            font=dict(color=COLORS['accent'])
        )

fig6.update_traces(hovertemplate="<b>Plan:</b> %{y}<br><b>Avg Spend:</b> $%{z:.2f}")
st.plotly_chart(fig6, use_container_width=True)

# Section 3: Engagement & Content Preferences
st.header("Engagement & Content Preferences")

col1, col2 = st.columns(2)

with col1:
    #st.subheader("Top 10 Genres by Total Watch Time")
    top_genres = (
        filtered_df.groupby('genre')['watch_duration_minutes']
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )
    top_genres.columns = ['genre', 'total_watch_duration']
    
    fig7 = px.bar(top_genres, y='genre', x='total_watch_duration',
                 title="Top 10 Genres by Total Watch Time",
                 orientation='h',
                 color='genre',
                 color_discrete_sequence=CATEGORY_COLORS)
    fig7.update_layout(
        xaxis_title="Total Watch Duration (minutes)",
        yaxis_title="Genre",
        showlegend=False,
        template="plotly_dark",  #Dark theme
        font=dict(color=COLORS['accent'])
    )
    fig7.update_traces(hovertemplate="<b>Genre:</b> %{y}<br><b>Total Watch:</b> %{x:.0f} min")
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    #st.subheader("Average Watch Duration Over Months")
    
    filtered_df['month'] = pd.to_datetime(filtered_df['watch_date']).dt.to_period('M')
    monthly_watch = filtered_df.groupby('month')['watch_duration_minutes'].mean().reset_index()
    monthly_watch['month'] = monthly_watch['month'].astype(str)
    
    fig8 = px.line(monthly_watch, x='month', y='watch_duration_minutes',
                  title="Average Watch Duration Over Months",
                  markers=True)
    
    fig8.update_traces(
        line=dict(color=COLORS['accent'], width=4),
        marker=dict(size=8, color=COLORS['primary']),
        hovertemplate="<b>Month:</b> %{x}<br><b>Avg Duration:</b> %{y:.1f} min"
    )
    
    fig8.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Watch Duration (minutes)",
        template="plotly_dark",  #Dark theme
        font=dict(color=COLORS['accent']),
        xaxis=dict(tickangle=45)
    )
    st.plotly_chart(fig8, use_container_width=True)

#st.subheader("Genre Popularity vs Average Sentiment")
genre_sentiment = filtered_df.groupby('genre').agg({
    'watch_duration_minutes': 'sum',
    'sentiment_score': 'mean'
}).reset_index()

fig9 = px.scatter(genre_sentiment, 
                 x='watch_duration_minutes', 
                 y='sentiment_score',
                 color='genre',
                 size=[10] * len(genre_sentiment),
                 title="Genre Popularity vs Average Sentiment",
                 color_discrete_sequence=CATEGORY_COLORS,
                 hover_name='genre',
                 hover_data={
                     'watch_duration_minutes': ':.0f',
                     'sentiment_score': ':.3f',
                     'genre': False
                 })

fig9.update_layout(
    xaxis_title="Total Watch Duration (minutes)",
    yaxis_title="Average Sentiment Score",
    template="plotly_dark",  #Dark theme
    font=dict(color=COLORS['accent'])
)
fig9.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>Watch Duration: %{x:.0f} min<br>Sentiment: %{y:.3f}",
    marker=dict(size=12, line=dict(width=2, color=COLORS['accent']))
)
st.plotly_chart(fig9, use_container_width=True)

# Section 4: Satisfaction & Reviews
st.header("Satisfaction & Reviews")

col1, col2 = st.columns(2)

with col1:
    #st.subheader("Average Sentiment by Subscription Plan")
    
    user_sentiment = filtered_df.groupby('user_id')['sentiment_score'].mean().reset_index()
    user_sentiment = user_sentiment.merge(filtered_df[['user_id', 'subscription_plan']].drop_duplicates(), on='user_id', how='left')
    plan_sentiment = user_sentiment.groupby('subscription_plan')['sentiment_score'].mean().sort_values(ascending=False)
    
    sentiment_df = plan_sentiment.reset_index()
    sentiment_df.columns = ['subscription_plan', 'sentiment_score']
    
    fig10 = px.bar(sentiment_df, x='subscription_plan', y='sentiment_score',
                  color='subscription_plan',
                  color_discrete_sequence=CATEGORY_COLORS,
                  title="Average Sentiment by Subscription Plan")
    
    fig10.update_layout(
        xaxis_title="Subscription Plan",
        yaxis_title="Average Sentiment Score",
        showlegend=False,
        template="plotly_dark",  #Dark theme
        font=dict(color=COLORS['accent'])
    )
    fig10.update_traces(
        hovertemplate="<b>Plan:</b> %{x}<br><b>Sentiment:</b> %{y:.3f}")
    st.plotly_chart(fig10, use_container_width=True)

with col2:
    #st.subheader("User-level Sentiment Distribution")
    user_sent = (
        filtered_df.dropna(subset=['sentiment_score'])
               .groupby('user_id')['sentiment_score']
               .mean()
               .reset_index(name='avg_sentiment')
    )
    
    # Create histogram with KDE line
    from scipy import stats
    sentiment_data = user_sent['avg_sentiment'].dropna()
    kde = stats.gaussian_kde(sentiment_data)
    x_range = np.linspace(sentiment_data.min(), sentiment_data.max(), 100)
    y_kde = kde(x_range)
    
    # Create histogram
    fig11 = px.histogram(user_sent, 
                        x='avg_sentiment', 
                        nbins=30,
                        title="User-level Sentiment Distribution",
                        color_discrete_sequence=[COLORS['primary']],
                        opacity=0.7)
    
    # Add KDE line
    fig11.add_scatter(x=x_range, 
                     y=y_kde * len(sentiment_data) * (sentiment_data.max() - sentiment_data.min()) / 30,
                     mode='lines', 
                     line=dict(color=COLORS['accent'], width=3),
                     name='Density', 
                     hovertemplate="<b>Sentiment:</b> %{x}<br><b>Density:</b> %{y:.3f}")
    
    fig11.update_layout(
        xaxis_title="Avg Sentiment Score",
        yaxis_title="Number of Users",
        template="plotly_dark",
        font=dict(color=COLORS['accent']),
        showlegend=True,
        bargap=0.3  # ← RESTORE GAP BETWEEN BARS
    )
    fig11.update_traces(hovertemplate="<b>Sentiment Range:</b> %{x}<br><b>Users:</b> %{y}")  
    st.plotly_chart(fig11, use_container_width=True)

#st.subheader("Most Frequent Words in Reviews")

try:
    all_reviews = " ".join(filtered_df['review_text'].dropna())
    words = re.findall(r'\b\w+\b', all_reviews.lower())

    custom_stopwords = {"movie", "series", "one", "watch", "watching", "t"}
    stopwords_set = STOPWORDS.union(custom_stopwords)
    words = [w for w in words if w not in stopwords_set and len(w) > 2]

    word_counts = Counter(words).most_common(30)

    if word_counts:
        top_words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        
        # INTERACTIVE BAR CHART (like you wanted)        
        fig_words = px.bar(top_words_df, 
                          x='Count', 
                          y='Word',
                          orientation='h',
                          title="Most Frequent Words in Reviews",
                          color='Count',
                          color_continuous_scale=['#B81D24', '#E50914'],  # Netflix red gradient
                          text='Count')
        
        fig_words.update_layout(
            xaxis_title="Frequency",
            yaxis_title="Words",
            yaxis={'categoryorder':'total ascending'},
            showlegend=False,
            template="plotly_dark",  #DARK THEME
            font=dict(color=COLORS['accent'])
        )
        
        fig_words.update_traces(
            hovertemplate="<b>%{y}</b><br>Count: %{x}",
            texttemplate='%{text}',
            textposition='outside'
        )
        
        st.plotly_chart(fig_words, use_container_width=True)

    else:
        st.info("No review text available for analysis")

except Exception as e:
    st.error(f"Error in text processing: {e}")


st.success("Use filters on the left to explore")
