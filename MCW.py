import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go  
import re
import seaborn as sns
import base64
import warnings
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from itertools import combinations  
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# Load the dataset
matches_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/all_t20_world_cup_matches_results.csv'
players_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/all_t20_world_cup_players_list.csv'
final_dataset_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/wc_final_dataset.csv'
captains_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/all_captains.csv'
cricket_legends_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/cricket_legends.csv'
updated_wc_final_data_df_url = 'https://raw.githubusercontent.com/Shamsvi/Streamlit_Mens_T-20_Cricket_WorldCup_2007-2024/main/updated_wc_final_data_df.csv'

# Load the datasets from the URLs
all_matches_data_df = pd.read_csv(matches_url)
players_df = pd.read_csv(players_url)
wc_final_data_df = pd.read_csv(final_dataset_url)
captains_df = pd.read_csv(captains_url)
cricket_legends_df = pd.read_csv(cricket_legends_url)
updated_wc_final_data_df = pd.read_csv(updated_wc_final_data_df_url)






#############################################################################################################################


#IDA 


# Define functions 
def extract_numeric_value(value):
    match = re.search(r'\d+', str(value))
    return int(match.group()) if match else None

def extract_t20_int_match(value):
    match = re.search(r'# (\d+)', str(value))  # Extract the number after the hash symbol
    return int(match.group(1)) if match else None

# Apply transformations to the wc_final_data_df
# Extracting Numerics from T-20 Int Match Column
wc_final_data_df['T-20 Int Match'] = wc_final_data_df['T-20 Int Match'].apply(extract_t20_int_match)

# Converting 'Match Date' to datetime and extract year, month, day
wc_final_data_df['Match Date'] = pd.to_datetime(wc_final_data_df['Match Date'], format='%Y/%m/%d')
wc_final_data_df['Match Year'] = wc_final_data_df['Match Date'].dt.year
wc_final_data_df['Match Month'] = wc_final_data_df['Match Date'].dt.month
wc_final_data_df['Match Day'] = wc_final_data_df['Match Date'].dt.day

# Drop the original 'Match Date' column
wc_final_data_df = wc_final_data_df.drop(columns=['Match Date'])

# Display the first few rows of the updated dataframe
print(wc_final_data_df.head())

# Save the updated dataframe
wc_final_data_df.to_csv('updated_wc_final_data_df.csv', index=False)





#############################################################################################################################




#Missingness
# Check for missing values in each dataset
missing_values_all_matches = all_matches_data_df.isnull().sum()
missing_values_wc_final_dataset = wc_final_data_df.isnull().sum()
missing_values_players = players_df.isnull().sum()

# Print the missing values for each dataset
print("Missing values in all_matches_data_df:\n", missing_values_all_matches)
print("Missing values in final_dataset_df:\n", missing_values_wc_final_dataset)
print("Missing values in players_df:\n", missing_values_players)


# Display rows in captains_df that contain missing values
missing_captains_rows = captains_df[captains_df.isnull().any(axis=1)]
# Show the rows with missing values
print(missing_captains_rows)


#############################################################################################################################

#Data Cleaning

# Define functions 
def extract_numeric_value(value):
    match = re.search(r'\d+', str(value))
    return int(match.group()) if match else None
def extract_t20_int_match(value):
    match = re.search(r'# (\d+)', str(value))  # Extract the number after the hash symbol
    return int(match.group(1)) if match else None
# Splitting Margins Column into Margb (runs) and Margin (Wickets)
def extract_runs_correct(margin):
    if isinstance(margin, str) and 'runs' in margin:
        return float(margin.split()[0])  # Extract the number for runs
    return None
def extract_wickets_correct(margin):
    if isinstance(margin, str) and 'wickets' in margin:
        return float(margin.split()[0])  # Extract the number for wickets
    return None
# spliting the Margin colum into Margin(Runs) and Margin (Wickets)
wc_final_data_df['Margin (Runs)'] = wc_final_data_df['Margin'].apply(extract_runs_correct)
wc_final_data_df['Margin (Wickets)'] = wc_final_data_df['Margin'].apply(extract_wickets_correct)
# Display the updated dataframe with the new columns
wc_final_data_df[['Margin', 'Margin (Runs)', 'Margin (Wickets)']].head()
print(wc_final_data_df.head())
# Save the updated dataframe
wc_final_data_df.to_csv('updated_wc_final_data_df.csv', index=False)


# Drop the unnamed columns from captains_df
captains_df_cleaned = captains_df.drop(columns=['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'])
print(captains_df_cleaned)



# Clean the folder_name column in cricket_legends_df
cricket_legends_df['folder_name'] = cricket_legends_df['folder_name'].str.replace('_', ' ').str.title()
# Merge the datasets
merged_df = pd.merge(cricket_legends_df, players_df, left_on='folder_name', right_on='Player Name', how='inner')
# Remove the 'folder_name' column
merged_df = merged_df.drop(columns=['folder_name'])
# Display the first few rows of the updated dataframe
print(merged_df.head())









#############################################################################################################################


# Feature Engineering and Data Transformation on the dataset

# 1. Derive "Home Advantage" feature
# Assume teams have an advantage when the match is played in their home country
# (Simple assumption based on team names and ground locations)
updated_wc_final_data_df['Home Advantage'] = updated_wc_final_data_df.apply(
    lambda row: 1 if row['Team1'] in row['Ground'] or row['Team2'] in row['Ground'] else 0, axis=1
)

# 2. Normalize ranking differences
# Normalize Batting and Bowling Ranking Difference columns to a 0-1 range for comparison
scaler = MinMaxScaler()
updated_wc_final_data_df[['Normalized Batting Difference', 'Normalized Bowling Difference']] = scaler.fit_transform(
    updated_wc_final_data_df[['Batting Ranking Difference', 'Bowling Ranking Difference']]
)

# 3. Create a feature for "Winning Margin Type"
# Categorize matches into "Close Match" or "Dominant Win" based on run/wicket margins
def categorize_margin(row):
    if row['Margin (Runs)'] > 20 or row['Margin (Wickets)'] > 5:
        return 'Dominant Win'
    elif row['Margin (Runs)'] > 0 or row['Margin (Wickets)'] > 0:
        return 'Close Match'
    else:
        return 'No Result'
updated_wc_final_data_df['Winning Margin Type'] = updated_wc_final_data_df.apply(categorize_margin, axis=1)

# 4. Aggregate performance by year
# Compute yearly aggregates for team performance metrics
updated_wc_final_data_df['Match Importance'] = updated_wc_final_data_df['T-20 Int Match'].apply(
    lambda x: 'High' if x > 300 else 'Low'
)

# 5. Create a feature for "Match Importance"
# Assume later-stage matches (e.g., finals) are more important based on match numbers
updated_wc_final_data_df['Rolling Win %'] = updated_wc_final_data_df.groupby('Team1')['Team1 win % over Team2'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
updated_wc_final_data_df['Rolling Margin (Runs)'] = updated_wc_final_data_df.groupby('Team1')['Margin (Runs)'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
updated_wc_final_data_df['Rolling Margin (Wickets)'] = updated_wc_final_data_df.groupby('Team1')['Margin (Wickets)'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# 6. Team Strength Index
# Combine Batting and Bowling rankings to create a Team Strength Index for both teams

updated_wc_final_data_df['Team1 Strength Index'] = (
    updated_wc_final_data_df['Team1 Avg Batting Ranking'] * 0.5 +
    updated_wc_final_data_df['Team1 Avg Bowling Ranking'] * 0.5
)
updated_wc_final_data_df['Team2 Strength Index'] = (
    updated_wc_final_data_df['Team2 Avg Batting Ranking'] * 0.5 +
    updated_wc_final_data_df['Team2 Avg Bowling Ranking'] * 0.5
)

# 7. Match Outcome as a Binary Feature
# Indicate whether Team1 won the match

updated_wc_final_data_df['Team1 Win'] = updated_wc_final_data_df['Winner'].apply(
    lambda x: 1 if x == 'Team1' else 0
)

# 8.  Derived Features for Batting/Bowling Disparity
# Calculate batting and bowling disparities between Team1 and Team2

updated_wc_final_data_df['Batting Disparity'] = updated_wc_final_data_df['Team1 Avg Batting Ranking'] - updated_wc_final_data_df['Team2 Avg Batting Ranking']
updated_wc_final_data_df['Bowling Disparity'] = updated_wc_final_data_df['Team1 Avg Bowling Ranking'] - updated_wc_final_data_df['Team2 Avg Bowling Ranking']

# 9. Performance in High-Pressure Matches
# Track wins and margins in high-pressure matches

updated_wc_final_data_df['High Pressure Win'] = updated_wc_final_data_df.apply(
    lambda row: 1 if row['Match Importance'] == 'High' and row['Team1 Win'] == 1 else 0, axis=1
)

# 10. Head-to-Head Records
# Aggregated stats for Team1 vs Team2 pairs

head_to_head_stats = updated_wc_final_data_df.groupby(['Team1', 'Team2']).agg({
    'Team1 Win': 'sum',
    'Margin (Runs)': 'mean',
    'Margin (Wickets)': 'mean'
}).reset_index()
head_to_head_stats.rename(columns={
    'Team1 Win': 'Head-to-Head Wins',
    'Margin (Runs)': 'Avg Margin (Runs)',
    'Margin (Wickets)': 'Avg Margin (Wickets)'
}, inplace=True)
updated_wc_final_data_df = updated_wc_final_data_df.merge(
    head_to_head_stats, 
    on=['Team1', 'Team2'], 
    how='left', 
    suffixes=('', '_head_to_head')
)

# 11. Seasonality Analysis
# Add features for the seasonality of the match

updated_wc_final_data_df['Season'] = updated_wc_final_data_df['Match Month'].apply(
    lambda x: 'Winter' if x in [12, 1, 2] else 
              'Spring' if x in [3, 4, 5] else 
              'Summer' if x in [6, 7, 8] else 'Fall'
)




#############################################################################################################################

#EDA
# Checking the correlation between missing values for each column [Margin (Runs) and Margin(Wickets)]
missing_correlation = wc_final_data_df.isnull().corr()
wc_final_data_df['Margin (Runs)'] = wc_final_data_df['Margin (Runs)'].fillna(0)
wc_final_data_df['Margin (Wickets)'] = wc_final_data_df['Margin (Wickets)'].fillna(0)
missing_values = wc_final_data_df.isnull().sum()

wc_final_data_df['Batting Ranking Difference'] = abs(wc_final_data_df['Team1 Avg Batting Ranking'] - wc_final_data_df['Team2 Avg Batting Ranking'])
wc_final_data_df['Bowling Ranking Difference'] = abs(wc_final_data_df['Team1 Avg Bowling Ranking'] - wc_final_data_df['Team2 Avg Bowling Ranking'])

print("wc_final_data_df with new features:\n", wc_final_data_df.head())
wc_final_data_df.to_csv('updated_wc_final_data_df.csv', index=False)

# Summary statistics for both datasets
matches_summary = all_matches_data_df.describe()
final_dataset_summary = wc_final_data_df.describe()
print("Summary statistics for matches_results_df:\n", matches_summary)
print("Summary statistics for final_dataset_df:\n", final_dataset_summary)

if 'Team1 Avg Batting Ranking' in wc_final_data_df.columns and 'Team2 Avg Batting Ranking' in wc_final_data_df.columns:
    wc_final_data_df['Batting Ranking Difference'] = abs(wc_final_data_df['Team1 Avg Batting Ranking'] - wc_final_data_df['Team2 Avg Batting Ranking'])

if 'Team1 Avg Bowling Ranking' in wc_final_data_df.columns and 'Team2 Avg Bowling Ranking' in wc_final_data_df.columns:
    wc_final_data_df['Bowling Ranking Difference'] = abs(wc_final_data_df['Team1 Avg Bowling Ranking'] - wc_final_data_df['Team2 Avg Bowling Ranking'])








#############################################################################################################################

# Documentation Section
with st.expander("📖 How to Navigate This App"):
    st.markdown("""
    ## Welcome to the ICC Men's T20 World Cup App!

    This app is designed to provide **cricket enthusiasts** and **data science aficionados** with an engaging and insightful experience. Whether you’re here to relive glorious matches or dive into predictive analytics, this app has something for everyone!

    ### 🔍 Two Aspects of This App:
    1. **🎭 Fan Favorites (User Interface)**  
       This section is tailored for cricket fans who want to explore:
       - **Matches and Wins by each Team**: Dive into performance stats of your favorite teams across matches.
       - **Grounds**: Gain insights into match outcomes based on specific venues.
       - **Participation**: Uncover trends in player participation across the years.
       - **Search For Your Favourite Teams and Players**: Quickly search and explore details about your beloved teams or players.

       Perfect for fans who want to relive the excitement of cricket or analyze historical performances in a simple, visual way.

    2. **🧪 Data Wizardry (Data Science Interface)**  
       This section is for data enthusiasts who want to:
       - **Distribution of Features**: Visualize and understand the distribution of key features in the dataset.
       - **Feature Factory**: Explore advanced feature engineering insights that power predictions.
       - **Predictor's Playground**: Experiment with machine learning models, view their predictions, and see which team might win the next World Cup!

       Aimed at users who love working with data and want to see the science behind cricket analytics.

    ### 🏏 How to Use the App:
    - Use the **sidebar** to switch between the two sections: 🎭 Fan Favorites and 🧪 Data Wizardry.
    - Each section has its own unique features, designed for specific audiences:
        - If you’re a **cricket fan**, start with **Fan Favorites** to explore match stats, player data, and more.
        - If you’re a **data science enthusiast**, head to **Data Wizardry** to play with data visualizations and predictive models.
    - You can easily toggle between sections and features from the sidebar.

    ### 🐻 What Makes This App Special?
    - Combines **cricket insights** and **data science** in one interactive platform.
    - Simple, intuitive navigation makes it accessible to everyone.
    - Advanced predictions provide a glimpse into which team might dominate the next T20 World Cup!

    Start exploring now and uncover the magic of cricket through data and visuals. Enjoy!
    """)


#############################################################################################################################

# Sidebar
st.sidebar.title("Explore the App")

# Dropdowns
summary = " 🌏 Beyond the Boundary: A Summary"
ui_name = "🎭 Fan Favorites"
ds_name = "🧪 Data Wizardry"



section_selector = st.sidebar.radio(
    "Choose a Section to Explore:",
    [ui_name, ds_name, summary ]
)

# Initialize variables to avoid NameError
ui_section = None
ds_section = None

if section_selector == ui_name:
    # Fan Favorites
    ui_section = st.sidebar.selectbox(
        ui_name,
        [
            "Welcome!",
            "Team Battles",
            "Ground Chronicles",
            "Player Glory",
            "Forecasting the Next Champions",
            "Search Magic"
        ]
    )



elif section_selector == ds_name:
    # Data Wizardry
    ds_section = st.sidebar.selectbox(
        ds_name,
        [
            "Welcome!",
            "About the Data",
            "Cricket Stats",
            "Feature Factory",
            "Modeling the Game: Unveiling Predictions",
            "Forecasting the Next Champions"
        ]
    )

elif section_selector == summary:
    # Summary Section
    st.subheader(" 🌏 Beyond the Boundary: Cricket Analytics Unveiled")
    st.write("Explore a comprehensive overview of this app, including its goals, features, and insights.")









#############################################################################################################################
    
if section_selector == summary:
    st.markdown("""
        # Real-world Application and Impact

        This project is a comprehensive analysis and prediction system for cricket matches, specifically focusing on the ICC Men's T20 Cricket World Cup. It leverages advanced data science techniques and machine learning models to deliver actionable insights and predictions. Here's how the project achieves real-world applicability and provides impactful conclusions:

        ## Real-world Applicability

        1. **Strategic Decision-making for Teams**:
        - Teams can analyze their performance metrics such as batting and bowling disparities, home advantage, and rolling win percentages.
        - Insights into strengths and weaknesses enable teams to strategize for upcoming matches more effectively.

        2. **Enhanced Fan Engagement**:
        - Fans can explore detailed visualizations of team performances, historical trends, and key statistics.
        - Predictive analytics allows fans to simulate potential outcomes, fostering a deeper connection with the sport.

        3. **Broadcaster and Media Insights**:
        - Broadcasters can use the predictive model to generate content around potential match outcomes and key player performances.
        - Real-time analytics can enhance pre- and post-match discussions.

        4. **Data-driven Policy for Cricket Boards**:
        - Cricket boards can leverage insights into team strengths and weaknesses to make decisions about player selections, training focus areas, and long-term planning.

        5. **Educational and Analytical Tool**:
        - The project serves as a template for teaching data science concepts, including feature engineering, machine learning, and visualization techniques.

        ## Insightful Conclusions

        1. **Feature Importance Analysis**:
        - Features such as **Team Strength Index**, **Rolling Win %**, and **Home Advantage** were identified as key predictors of match outcomes. Teams with higher rolling win percentages consistently outperformed their peers.

        2. **Historical Trends and Predictions**:
        - Teams with balanced batting and bowling disparities tend to perform better in high-pressure situations.
        - The prediction system revealed probable winners based on the simulation of round-robin matches, showcasing the power of machine learning in forecasting outcomes.

        3. **Model Performance Insights**:
        - Among the models compared, **XGBoost** demonstrated the highest accuracy and F1 score, making it the most reliable for predicting match results.

        4. **Applicability of Engineered Features**:
        - Engineered features like **Normalized Batting/Bowling Differences** and **High-Pressure Wins** provided critical insights into team dynamics that traditional metrics might overlook.

        ## Recommendations

        1. **Data-Driven Training Strategies**:
        - Teams should focus on narrowing their disparities in batting and bowling performance, as these metrics significantly influence match outcomes.

        2. **Utilization of Rolling Metrics**:
        - Rolling averages for margins and win percentages can provide teams with insights into recent performance trends, helping them refine strategies for critical matches.

        3. **Improved Scheduling for Home Advantage**:
        - Cricket boards can optimize match schedules to capitalize on home advantage, as this has a tangible impact on team performance.

        4. **Expand the Predictive Framework**:
        - Incorporate player-level data and situational metrics (e.g., match pressure) for a more granular prediction model.

        5. **Future Applications**:
        - Expand this framework to other sports or tournaments to enable cross-sport analytical comparisons and broader use cases.

        ---

        The project not only demonstrates the power of data analytics in sports but also highlights the growing relevance of machine learning in solving complex real-world problems. With enhanced predictive capabilities and strategic insights, this system stands as a valuable asset for stakeholders across the cricket ecosystem.
        """)


#############################################################################################################################


if ui_section == "Welcome!":    
    st.title("🏏 Welcome to the Ultimate Men's T20 World Cup Analysis App! 🏆")
    st.subheader('Cricket Fever') 

        # Displaying the GIF from the raw GitHub link
    gif_url = "https://raw.githubusercontent.com/Shamsvi/Streamlit_Mens_T-20_Cricket_WorldCup_2007-2024/main/giphy.gif"
    st.image(gif_url, use_container_width=True)

    st.markdown("""
    # 🏏 Welcome to the Men's T20 World Cup Data Explorer!

    Are you ready to dive into the thrilling world of cricket? Whether you’re a die-hard fan, a stats geek, or just someone who loves the spirit of the game, this app is your one-stop destination to explore and analyze everything about the Men's T20 World Cup!

    ✨ From nail-biting finishes to record-breaking performances, this app unpacks the data behind the drama. Explore:

    🔥 **Team Battles**: Who dominated the field and who needs to up their game?  
    
    🌍 **Ground Chronicles**: Which stadiums turned into fortresses for teams?  
    
    🌟 **Player Glory**: Discover stars who shone brightest under pressure.  
    
    🏆 **Forecasting the Next Champions**: Using advanced machine learning models and historical data, predict which team could lift the next Men's T20 World Cup trophy!  
    
    🕵️‍♂️ **Search Magic**: Zero in on your favorite teams or players in an instant!

    ### 🎉 Why this app?
    Because cricket isn’t just a sport—it’s a passion, a science, and a celebration. And with this app, you can experience it all in an interactive, fun, and data-driven way.
    """)


        # Footer or call-to-action
    st.markdown("---")
    st.markdown("### 🏏 Let the cricket journey begin! Navigate using the sidebar to explore more insights.")

#############################################################################################################################




#  Display the dataset
elif ui_section == "Dataset Overview":
    st.subheader('Dataset Overview')    
    st.write("The Men's Cricket World Cup is one of the most prestigious tournaments in the cricketing calendar, showcasing the talents of the world's best teams and players. This project aims to analyze and visualize key data from the tournament, focusing on match outcomes, team performances, and individual player statistics. By leveraging advanced data analysis techniques, we will explore trends in match margins, batting and bowling averages, and historical rivalries. Through this comprehensive analysis, we seek to provide valuable insights into the dynamics of the tournament, enhancing our understanding of competitiveness and performance in international cricket.")
    wc_final_data_df = pd.read_csv('updated_wc_final_data_df.csv')
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(wc_final_data_df, key = 'dataframe')
    with col2:
        st.dataframe(players_df, key = 'dataframe')






#############################################################################################################################












#############################################################################################################################


# Matches and Wins by each Team

elif ui_section == "Team Battles":
    st.subheader("Team Battles")
    st.markdown("""
    Welcome to the **Matches and Wins by Each Team** section—a place where cricket history comes alive! 🏏

    Here, you’ll find out how many matches each team has played over the years and where they’ve played them. 
    We’ve got a colorful **bar chart** that shows the total matches for every team. Want to know where your favorite 
    team has battled it out? Check out the interactive **map**, which highlights the countries where matches were played.
    """)
    # Calculate total matches for each team
    team1_counts = wc_final_data_df['Team1'].value_counts()
    team2_counts = wc_final_data_df['Team2'].value_counts()
    total_matches = pd.DataFrame({'Team': team1_counts.index, 'Matches': team1_counts.values})
    team2_matches = pd.DataFrame({'Team': team2_counts.index, 'Matches': team2_counts.values})
    total_matches = pd.concat([total_matches, team2_matches], ignore_index=True)
    total_matches = total_matches.groupby('Team', as_index=False).sum()

    # Bar Plot: Total Matches Played by Each Team
    fig_total_matches = px.bar(
        total_matches,
        x='Team',
        y='Matches',
        color='Matches',  # Color by Matches
        title='Total Matches Played by Each Team',
        labels={'Matches': 'Number of Matches', 'Team': 'Team'},
        text='Matches',
        template='plotly_white',  # Set template to plotly_white
        color_continuous_scale='Viridis'  # Use Viridis palette
    )
    fig_total_matches.update_traces(marker_line_color='black', marker_line_width=1.5)

    # Geospatial Visualization: Choropleth Map
    team_country_mapping = {
        'India': 'India',
        'Australia': 'Australia',
        'England': 'United Kingdom',
        'Pakistan': 'Pakistan',
        'South Africa': 'South Africa',
        'Sri Lanka': 'Sri Lanka',
        'West Indies': 'Jamaica',  # Using Jamaica to represent West Indies
        'Bangladesh': 'Bangladesh',
        'Nepal': 'Nepal',
        'Zimbabwe': 'Zimbabwe',
        'Afghanistan': 'Afghanistan',
        'New Zealand': 'New Zealand',
        'Netherlands': 'Netherlands',
        'Scotland': 'United Kingdom',  # Mapping Scotland to United Kingdom
        'USA': 'United States',
        'Ireland': 'Ireland',
        'Kenya': 'Kenya',
        'Oman': 'Oman',
        'United Arab Emirates': 'United Arab Emirates',
        'Hong Kong': 'Hong Kong',
        'PNG': 'Papua New Guinea',  # Ensure correct mapping for P.N.G
        'Canada': 'Canada',
        'Uganda': 'Uganda'
    }

    # Map teams to their respective countries
    total_matches['Country'] = total_matches['Team'].map(lambda x: team_country_mapping.get(x, 'Unknown'))

    # Create a choropleth map
    fig_geo_total_matches = px.choropleth(
        total_matches,
        locations='Country',
        locationmode='country names',
        color='Matches',
        hover_name='Country',
        hover_data=['Matches'],
        title='Total Matches Played by Each Team',
        color_continuous_scale='Viridis'  # Use Viridis palette
    )

    # Customize the choropleth layout
    fig_geo_total_matches.update_geos(
        showcoastlines=True,
        coastlinecolor='Black',
        landcolor='LightGray',
        countrycolor='Black',
        showsubunits=True,
        showcountries=True
    )

    # Display plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_total_matches, use_container_width=True)
    with col2:
        st.plotly_chart(fig_geo_total_matches, use_container_width=True)



    st.subheader("Wins - Total No. of Wins by Each Team")
    st.markdown("""
    Here we see who’s been crushing it on the field with a chart that shows the **total wins for each team**. 
    Want to know which countries are the real champs? We’ve got another **map** that paints the picture of 
    victories across the globe.
    """)
    # Function to extract numeric values for margins
    def extract_numeric(value):
        match = re.search(r'\d+', str(value))
        return int(match.group()) if match else None

    # Functions to extract margins (runs and wickets)
    def extract_runs_correct(margin):
        if isinstance(margin, str) and 'runs' in margin:
            return float(margin.split()[0])
        return None

    def extract_wickets_correct(margin):
        if isinstance(margin, str) and 'wickets' in margin:
            return float(margin.split()[0])
        return None

    # Function to determine the losing team based on the winner
    def determine_losing_team(row):
        if row['Winner'] == row['Team1']:
            return row['Team2']
        else:
            return row['Team1']

    # Load the T20 World Cup dataset
    wc_final_data_df = pd.read_csv('updated_wc_final_data_df.csv')

    # Count wins per team
    win_counts = wc_final_data_df['Winner'].value_counts().reset_index()
    win_counts.columns = ['Team', 'Wins']

    # Bar Plot for Wins with Viridis palette
    fig_bar_wins = px.bar(
        win_counts,
        x='Team',
        y='Wins',
        labels={'Wins': 'Number of Wins', 'Team': 'Team'},
        text='Wins',
        title='Number of Wins by Country (Teams)',
        color='Wins',
        color_continuous_scale='Viridis'  # Use Viridis palette
    )

    fig_bar_wins.update_layout(
        xaxis_title='Team',
        yaxis_title='Number of Wins',
        hovermode='closest'
    )

    # Mapping teams to countries
    team_mapping = {
        'India': 'India',
        'Australia': 'Australia',
        'England': 'United Kingdom',
        'Pakistan': 'Pakistan',
        'South Africa': 'South Africa',
        'Sri Lanka': 'Sri Lanka',
        'West Indies': 'Jamaica',
        'Bangladesh': 'Bangladesh',
        'Nepal': 'Nepal',
        'Zimbabwe': 'Zimbabwe',
        'Afghanistan': 'Afghanistan',
        'New Zealand': 'New Zealand',
        'Netherlands': 'Netherlands',
        'Namibia': 'Namibia',
        'Scotland': 'Scotland',
        'USA': 'United States',
        'Ireland': 'Ireland',
        'Kenya': 'Kenya',
        'Oman': 'Oman',
        'United Arab Emirates': 'United Arab Emirates',
        'Hong Kong': 'Hong Kong',
        'PNG': 'Papua New Guinea',
        'Canada': 'Canada',
        'Uganda': 'Uganda'
    }

    # Geospatial map for wins
    wc_final_data_df['Country'] = wc_final_data_df['Winner'].map(team_mapping)
    country_wins = wc_final_data_df['Country'].value_counts().reset_index()
    country_wins.columns = ['Country', 'Wins']

    # Choropleth map for total wins with Viridis palette
    fig_geo_wins = px.choropleth(
        country_wins,
        locations='Country',
        locationmode='country names',
        color='Wins',
        hover_name='Country',
        hover_data=['Wins'],
        title='Total Wins by Country (Teams)',
        color_continuous_scale='Viridis'  # Use Viridis palette
    )

    fig_geo_wins.update_geos(
        showcoastlines=True,
        coastlinecolor='Black',
        landcolor='LightGray',
        countrycolor='Black',
        showsubunits=True,
        showcountries=True
    )

    # Display the bar and choropleth side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_bar_wins, use_container_width=True)
    with col2:
        st.plotly_chart(fig_geo_wins, use_container_width=True)

    # Add the Losing Team column
    st.markdown("""
    Ever wondered how big a team’s win was—whether they won by runs or wickets? Our **scatter plot** shows just that, 
    giving you an exciting way to dive into the details of every victory. Plus, there’s a handy table if you want 
    to explore specific match results.
    """)
    wc_final_data_df['Losing Team'] = wc_final_data_df.apply(determine_losing_team, axis=1)

    # Scatter Plot for margins with Viridis palette
    wc_final_data_df['Margin (Runs)'] = wc_final_data_df['Margin'].apply(extract_runs_correct)
    wc_final_data_df['Margin (Wickets)'] = wc_final_data_df['Margin'].apply(extract_wickets_correct)

    plot_data = wc_final_data_df[['Winner', 'Losing Team', 'Margin (Runs)', 'Margin (Wickets)']].dropna(subset=['Winner', 'Losing Team'])
    plot_data['Margin Type'] = plot_data.apply(lambda row: 'Runs' if pd.notnull(row['Margin (Runs)']) else 'Wickets', axis=1)
    plot_data['Margin Numeric'] = plot_data.apply(lambda row: row['Margin (Runs)'] if pd.notnull(row['Margin (Runs)']) else row['Margin (Wickets)'], axis=1)

    fig_win_margin = px.scatter(
        plot_data,
        x='Winner',
        y='Margin Numeric',
        title='Match Margin by Winner (Scatter Plot)',
        labels={'Margin Numeric': 'Match Margin', 'Winner': 'Winning Team'},
        color='Margin Numeric',
        hover_data={
            'Winner': True,
            'Margin Numeric': True,
            'Losing Team': True,
            'Margin Type': True
        },
        color_continuous_scale='Viridis'  # Use Viridis palette
    )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_win_margin, use_container_width=True)
    with col4:
        st.dataframe(plot_data.drop(columns=['Margin Numeric']), use_container_width=True)

    # Add Match Date column by parsing year, month, and day
    wc_final_data_df['Match Date'] = pd.to_datetime(
        wc_final_data_df[['Match Year', 'Match Month', 'Match Day']].astype(str).agg('-'.join, axis=1), errors='coerce')

    # List of specific final match dates
    final_dates = [
        '2007-09-24', '2009-06-21', '2010-05-16', '2012-10-07',
        '2014-04-06', '2016-04-03', '2021-11-14', '2022-11-13', '2024-06-29'
    ]
    final_dates = pd.to_datetime(final_dates)

    final_matches_df = wc_final_data_df[wc_final_data_df['Match Date'].isin(final_dates)]
    winner_titles = final_matches_df.groupby('Winner')['Match Date'].apply(list).reset_index(name='Final Dates')
    winner_titles['Final Dates'] = winner_titles['Final Dates'].apply(lambda dates: [date.strftime('%Y-%m-%d') for date in dates])
    winner_titles['Titles'] = winner_titles['Final Dates'].apply(len)

    # Horizontal bar chart for titles
    fig_titles = px.bar(
        winner_titles,
        y='Winner',
        x='Titles',
        orientation='h',
        text='Titles',
        hover_data={'Final Dates': True},
        labels={'Titles': 'Number of Titles', 'Winner': 'Team'},
        title='T20 World Cup Title Holders',
        color='Titles',
        color_continuous_scale='Viridis'  # Use Viridis palette
    )

    st.plotly_chart(fig_titles)






#############################################################################################################################



#Total Matches played at each grounds


elif ui_section == "Ground Chronicles":
    st.subheader("Ground Chronicles")
    st.write("""
    In this section, we explore the impact of different cricket grounds on match outcomes. We dive into the total matches played at each ground, the geographic distribution of these venues, and the winning trends at specific locations. Each visualization provides unique insights into how grounds influence team performances.
    """)    

    
    
    st.subheader("Total Matches Played at Each Ground")
    st.markdown(""" The bar chart shows the total number of matches played at various cricket grounds, providing insights into the most frequently used venues.  
    The accompanying scatter map visualizes the geographic distribution of these grounds. The size of the points on the map reflects the number of matches played at each venue, offering a clear view of cricket’s global footprint.
    """)
    ground_country_mapping = {
        'Abu Dhabi': 'Abu Dhabi',
        'Adelaide': 'Australia',
        'Al Amerat': 'Oman',
        'Bengaluru': 'India',
        'Bridgetown': 'West Indies',
        'Brisbane': 'Australia',
        'Cape Town': 'South Africa',
        'Chattogram': 'Bangladesh',
        'Colombo (RPS)': 'Sri Lanka',
        'Dallas': 'United States',
        'Delhi': 'India',
        'Dharamsala': 'India',
        'Dubai (DICS)': 'United Arab Emirates',
        'Durban': 'South Africa',
        'Eden Gardens': 'India',
        'Geelong': 'Australia',
        'Gros Islet': 'West Indies',
        'Hambantota': 'Sri Lanka',
        'Hobart': 'Australia',
        'Johannesburg': 'South Africa',
        'Kingstown': 'West Indies',
        'Lauderhill': 'United States',
        'Lord\'s': 'United Kingdom',
        'Melbourne': 'Australia',
        'Mirpur': 'Bangladesh',
        'Mohali': 'India',
        'Nagpur': 'India',
        'New York': 'United States',
        'North Sound': 'West Indies',
        'Nottingham': 'United Kingdom',
        'Pallekele': 'Sri Lanka',
        'Perth': 'Australia',
        'Providence': 'West Indies',
        'Sharjah': 'United Arab Emirates',
        'Sydney': 'Australia',
        'Sylhet': 'Bangladesh',
        'Tarouba': 'West Indies',
        'The Oval': 'United Kingdom',
        'Wankhede': 'India'
    }
    ground_data = {
        'Abu Dhabi': {'Country': 'United Arab Emirates', 'Lat': 24.4539, 'Lon': 54.3773},
        'Adelaide': {'Country': 'Australia', 'Lat': -34.9285, 'Lon': 138.6007},
        'Al Amerat': {'Country': 'Oman', 'Lat': 23.5881, 'Lon': 58.1364},
        'Bengaluru': {'Country': 'India', 'Lat': 12.9716, 'Lon': 77.5946},
        'Bridgetown': {'Country': 'West Indies', 'Lat': 13.1939, 'Lon': -59.6131},
        'Brisbane': {'Country': 'Australia', 'Lat': -27.4698, 'Lon': 153.0251},
        'Cape Town': {'Country': 'South Africa', 'Lat': -33.9249, 'Lon': 18.4241},
        'Chattogram': {'Country': 'Bangladesh', 'Lat': 22.3475, 'Lon': 91.8123},
        'Colombo (RPS)': {'Country': 'Sri Lanka', 'Lat': 6.9271, 'Lon': 79.9553},
        'Dallas': {'Country': 'United States', 'Lat': 32.7767, 'Lon': -96.7970},
        'Delhi': {'Country': 'India', 'Lat': 28.6139, 'Lon': 77.2090},
        'Dharamsala': {'Country': 'India', 'Lat': 32.2196, 'Lon': 76.3238},
        'Dubai (DICS)': {'Country': 'United Arab Emirates', 'Lat': 25.276987, 'Lon': 55.296249},
        'Durban': {'Country': 'South Africa', 'Lat': -29.8587, 'Lon': 31.0218},
        'Eden Gardens': {'Country': 'India', 'Lat': 22.5697, 'Lon': 88.3426},
        'Geelong': {'Country': 'Australia', 'Lat': -38.1499, 'Lon': 144.3617},
        'Gros Islet': {'Country': 'West Indies', 'Lat': 14.0589, 'Lon': -60.9492},
        'Hambantota': {'Country': 'Sri Lanka', 'Lat': 6.1246, 'Lon': 81.1183},
        'Hobart': {'Country': 'Australia', 'Lat': -42.8821, 'Lon': 147.3272},
        'Johannesburg': {'Country': 'South Africa', 'Lat': -26.2041, 'Lon': 28.0473},
        'Kingstown': {'Country': 'West Indies', 'Lat': 13.1579, 'Lon': -61.2248},
        'Lauderhill': {'Country': 'United States', 'Lat': 26.1483, 'Lon': -80.2133},
        'Lord\'s': {'Country': 'United Kingdom', 'Lat': 51.5264, 'Lon': -0.1965},
        'Melbourne': {'Country': 'Australia', 'Lat': -37.8136, 'Lon': 144.9631},
        'Mirpur': {'Country': 'Bangladesh', 'Lat': 23.8103, 'Lon': 90.4125},
        'Mohali': {'Country': 'India', 'Lat': 30.6928, 'Lon': 76.7480},
        'Nagpur': {'Country': 'India', 'Lat': 21.1458, 'Lon': 79.0882},
        'New York': {'Country': 'United States', 'Lat': 40.7128, 'Lon': -74.0060},
        'North Sound': {'Country': 'West Indies', 'Lat': 17.1381, 'Lon': -61.8456},
        'Nottingham': {'Country': 'United Kingdom', 'Lat': 52.9541, 'Lon': -1.1580},
        'Pallekele': {'Country': 'Sri Lanka', 'Lat': 7.2868, 'Lon': 80.5906},
        'Perth': {'Country': 'Australia', 'Lat': -31.9505, 'Lon': 115.8605},
        'Providence': {'Country': 'West Indies', 'Lat': 6.5030, 'Lon': -55.1708},
        'Sharjah': {'Country': 'United Arab Emirates', 'Lat': 25.3375, 'Lon': 55.5123},
        'Sydney': {'Country': 'Australia', 'Lat': -33.8688, 'Lon': 151.2093},
        'Sylhet': {'Country': 'Bangladesh', 'Lat': 24.8949, 'Lon': 91.8687},
        'Tarouba': {'Country': 'West Indies', 'Lat': 10.2900, 'Lon': -61.4240},
        'The Oval': {'Country': 'United Kingdom', 'Lat': 51.4815, 'Lon': -0.1071},
        'Wankhede': {'Country': 'India', 'Lat': 18.9385, 'Lon': 72.8347}
    }

    ground_df = pd.DataFrame.from_dict(ground_data, orient='index')
    ground_df.reset_index(inplace=True)
    ground_df.columns = ['Ground', 'Country', 'Lat', 'Lon']
    ground_match_counts = wc_final_data_df['Ground'].value_counts().reset_index()
    ground_match_counts.columns = ['Ground', 'Matches'] 
    merged_data = pd.merge(ground_df, ground_match_counts, on='Ground', how='left')
    #Bar chart
    fig_grounds_bar = px.bar(merged_data,
                            x='Ground',
                            y='Matches',
                            labels={'Matches': 'Number of Matches', 'Ground': 'Ground'},
                            title= 'Total No. of Matches Played at each Ground',
                            color='Matches',  
                            color_continuous_scale='Viridis',
                            text='Matches')  
    fig_grounds_bar.update_layout(
        xaxis_title='Ground',
        yaxis_title='Number of Matches',)
    #Geospatial chart
    fig_grounds = px.scatter_geo(merged_data,
                        lat='Lat',
                        lon='Lon',
                        size='Matches',  
                        hover_name='Ground',  
                        hover_data=['Country', 'Matches'],  
                        color='Matches', 
                        color_continuous_scale='Viridis')  
    
    fig_grounds.update_geos(
        showcoastlines=True,  
        coastlinecolor='Black',  
        showsubunits=True,  
        showcountries=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_grounds_bar, use_container_width=True)
    with col2:
        st.plotly_chart(fig_grounds, use_container_width=True)
    #Winners at each Ground
    ground_matches = wc_final_data_df['Ground'].value_counts().reset_index()
    ground_matches.columns = ['Ground', 'Matches']  
    ground_matches['Winning Teams'] = ground_matches['Ground'].map(
        lambda x: ', '.join(wc_final_data_df[wc_final_data_df['Ground'] == x]['Winner'].unique())
    )
    # Heatmap
    st.subheader("Heatmaps")
    st.markdown("""
    **How Grounds Influence Team Success:**  
    This section helps you discover which teams perform best at specific cricket grounds. The visual shows how many times each team has won at various venues, giving you a clear picture of the places where teams dominate. Alongside, the table provides a detailed breakdown of the connection between grounds and winning teams, offering easy-to-understand numbers to support the visual patterns. It's a simple way to explore which teams thrive where and uncover surprising insights about their winning streaks.
    """)

    ground_winner_pivot = wc_final_data_df.pivot_table(index='Ground', 
                                                    columns='Winner', 
                                                    aggfunc='size', 
                                                    fill_value=0)
    fig_heatmap = px.imshow(ground_winner_pivot,
                            title='Heatmap of Teams and Their Maximum Wins at a Ground',
                            labels={'color': 'Number of Wins'},
                            color_continuous_scale='Viridis',
                            text_auto=True) 
    fig_heatmap.update_layout(
        xaxis_title='Winning Countries',
        yaxis_title='Grounds',
        width=800,
        height=600
    )
    st.plotly_chart(fig_heatmap)

    #Correlation Heatmap - one hot encoding
    st.subheader("Correlatioanl Heatmap")
    df_encoded = pd.get_dummies(wc_final_data_df[['Winner', 'Ground']])
    # Compute the correlation matrix
    correlation_matrix = df_encoded.corr()
    # Filter the correlation matrix to show only 'Winner' and 'Ground' correlations
    winner_columns = [col for col in df_encoded.columns if col.startswith('Winner')]
    ground_columns = [col for col in df_encoded.columns if col.startswith('Ground')]

    correlation_matrix_filtered = correlation_matrix.loc[ground_columns, winner_columns]
    fig_corr_heatmap = px.imshow(
        correlation_matrix_filtered,
        title='Correlation Heatmap between Encoded Grounds and Winning Teams',
        labels=dict(x='Winning Teams', y='Grounds', color='Correlation Coefficient'),
        color_continuous_scale='Viridis',
        text_auto=':.2f'  
    )
    fig_corr_heatmap.update_layout(
        xaxis_title='Winning Teams',
        yaxis_title='Grounds',
        width=800,
        height=600,
        xaxis=dict(tickangle=45),  
        yaxis=dict(tickangle=0)   
    )
    st.plotly_chart(fig_corr_heatmap)
    st.dataframe(correlation_matrix_filtered)







#############################################################################################################################

#Participation

elif ui_section == "Player Glory":
    st.subheader("Player Glory")
    st.markdown("""
    **Unveiling Player Glory**  

    Explore the incredible contributions of players to the T20 World Cup in this engaging section. From participation and wins across years to standout performances, we've got it all! 

    """)

    if 'Match Year' not in wc_final_data_df.columns:
        wc_final_data_df['Match Year'] = pd.to_datetime(wc_final_data_df['Match Date'], errors='coerce').dt.year

    all_years = pd.Series(range(2007, 2025))
    
    # Team 1 participation and wins
    team1_participation = wc_final_data_df.groupby(['Match Year', 'Team1']).size().reset_index(name='Total Participation')
    team1_wins = wc_final_data_df[wc_final_data_df['Winner'] == wc_final_data_df['Team1']].groupby('Match Year').size().reset_index(name='Total Wins')
    team1_stats = pd.merge(all_years.to_frame(name='Match Year'), team1_participation, how='left', on='Match Year')
    team1_stats = pd.merge(team1_stats, team1_wins, how='left', on='Match Year')
    team1_stats['Total Participation'] = team1_stats['Total Participation'].fillna(0)
    team1_stats['Total Wins'] = team1_stats['Total Wins'].fillna(0)

    # Team 2 participation and wins
    team2_participation = wc_final_data_df.groupby(['Match Year', 'Team2']).size().reset_index(name='Total Participation')
    team2_wins = wc_final_data_df[wc_final_data_df['Winner'] == wc_final_data_df['Team2']].groupby('Match Year').size().reset_index(name='Total Wins')
    team2_stats = pd.merge(all_years.to_frame(name='Match Year'), team2_participation, how='left', on='Match Year')
    team2_stats = pd.merge(team2_stats, team2_wins, how='left', on='Match Year')
    team2_stats['Total Participation'] = team2_stats['Total Participation'].fillna(0)
    team2_stats['Total Wins'] = team2_stats['Total Wins'].fillna(0)

    # Combined Bar and Line Plot for Team 1 and Team 2
    st.subheader("Head-to-Head Wins Between Teams")
    st.markdown("""
    Dive into the intense rivalries between Team 1 and Team 2! This section showcases the number of times one team has defeated the other in direct matchups, giving a sense of which team holds the upper hand.
    """)
    fig_team1_team2 = go.Figure()
    fig_team1_team2.add_trace(go.Bar(
        x=team1_stats['Match Year'], 
        y=team1_stats['Total Participation'], 
        name='Team 1 Participation', 
        marker_color=px.colors.sequential.Viridis[3],  # Viridis color for Team 1 Participation
        yaxis='y'
    ))
    fig_team1_team2.add_trace(go.Scatter(
        x=team1_stats['Match Year'], 
        y=team1_stats['Total Wins'], 
        mode='lines', 
        name='Team 1 Wins', 
        line=dict(color=px.colors.sequential.Viridis[6]),  # Viridis color for Team 1 Wins
        yaxis='y2'
    ))
    fig_team1_team2.add_trace(go.Bar(
        x=team2_stats['Match Year'], 
        y=team2_stats['Total Participation'], 
        name='Team 2 Participation', 
        marker_color=px.colors.sequential.Viridis[1],  # Viridis color for Team 2 Participation
        yaxis='y'
    ))
    fig_team1_team2.add_trace(go.Scatter(
        x=team2_stats['Match Year'], 
        y=team2_stats['Total Wins'], 
        mode='lines', 
        name='Team 2 Wins', 
        line=dict(color=px.colors.sequential.Viridis[9]),  # Viridis color for Team 2 Wins
        yaxis='y2'
    ))

    fig_team1_team2.update_layout(
        title='WC Participation (Bar) and Wins (Line) for Team 1 and Team 2 Over The Years',
        xaxis_title='Match Year',
        yaxis=dict(
            title='Total Participation',
            showgrid=False
        ),
        yaxis2=dict(
            title='Total Wins',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        width=1000,
        height=600,
        barmode='group',
        hovermode='closest'
    )
    st.plotly_chart(fig_team1_team2, use_container_width=True)

    # Head-to-Head Wins Analysis
    win_counts = all_matches_data_df.groupby(['Winner', 'Team1', 'Team2']).size().reset_index(name='Wins')
    win_counts['Hover Text'] = win_counts.apply(lambda row: f"Team 1: {row['Team1']}<br>Team 2: {row['Team2']}<br>Winner: {row['Winner']}<br>Wins: {row['Wins']}", axis=1)

    # Stacked Bar Chart for Wins of Team 1 Against Team 2
    fig_team1_over_team2 = px.bar(
        win_counts,
        x='Team1',
        y='Wins',
        color='Winner',
        title='Wins of Team 1 Against Team 2',
        labels={'Wins': 'Number of Wins', 'Team1': 'Team 1'},
        text='Wins',
        hover_name='Hover Text',
        color_discrete_sequence=px.colors.sequential.Viridis  # Use Viridis palette
    )
    fig_team1_over_team2.update_layout(
        xaxis_title='Team 1',
        yaxis_title='Number of Wins',
        hovermode='closest',
        barmode='stack',
        xaxis_tickangle=-45
    )

    # Stacked Bar Chart for Wins of Team 2 Against Team 1
    fig_team2_over_team1 = px.bar(
        win_counts,
        x='Team2',
        y='Wins',
        color='Winner',
        title='Wins of Team 2 Against Team 1',
        labels={'Wins': 'Number of Wins', 'Team2': 'Team 2'},
        text='Wins',
        hover_name='Hover Text',
        color_discrete_sequence=px.colors.sequential.Viridis  # Use Viridis palette
    )
    fig_team2_over_team1.update_layout(
        xaxis_title='Team 2',
        yaxis_title='Number of Wins',
        hovermode='closest',
        barmode='stack',
        xaxis_tickangle=-45
    )

    # Display the visualizations side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_team1_over_team2, use_container_width=True)
    with col2:
        st.plotly_chart(fig_team2_over_team1, use_container_width=True)


    st.subheader("Player Participation Trends")
    st.markdown("""
    Watch how teams have been represented by their players over the years. This section highlights how many players from each country participated in different tournaments, giving a sense of their consistent presence in the competition.
    """)
    # Calculate player participation trends
    players_by_country = players_df['Team'].value_counts()
    player_participation_trends = players_df.groupby(['Year', 'Team']).size().reset_index(name='Player Count')
    player_distribution = player_participation_trends.pivot_table(index='Year', columns='Team', values='Player Count', fill_value=0)
    all_years = pd.DataFrame({'Year': list(range(2007, 2025))})
    all_teams = player_participation_trends['Team'].unique()
    all_years_teams = pd.MultiIndex.from_product([all_years['Year'], all_teams], names=['Year', 'Team']).to_frame(index=False)
    player_participation_trends = pd.merge(all_years_teams, player_participation_trends, on=['Year', 'Team'], how='left')
    player_participation_trends['Player Count'] = player_participation_trends['Player Count'].fillna(0)

    # Line chart for player participation trends
    fig_players = px.line(
        player_participation_trends, 
        x='Year', 
        y='Player Count', 
        color='Team', 
        title='Player Participation Trends Over the Years by Country',
        labels={'Player Count': 'Number of Players', 'Year': 'Year'},
        template='plotly_white',
        color_discrete_sequence=px.colors.sequential.Viridis  # Use Viridis palette
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_players, use_container_width=True)
    with col2:
        st.dataframe(player_participation_trends, use_container_width=True)

    # Players with the longest participation
    st.subheader("Legends of Longevity")
    st.markdown("""
    Meet the players who have participated in the most World Cups for their teams. Their dedication to the game shines through in this section, celebrating their enduring contributions to cricket.
    """)
    player_participation = players_df.groupby(['Player Name', 'Team'])['Year'].nunique().reset_index(name='Years Participated')
    longest_participation = player_participation.loc[player_participation.groupby('Team')['Years Participated'].idxmax()]

    # Bar chart for longest participation
    fig_longest_participation = px.bar(
        longest_participation, 
        x='Team', 
        y='Years Participated', 
        color='Player Name', 
        title='Player with the Longest Participation for Each Team',
        labels={'Years Participated': 'Number of Years', 'Team': 'Team'},
        text='Player Name',
        template='plotly_white',
        color_discrete_sequence=px.colors.sequential.Viridis  # Use Viridis palette
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_longest_participation, use_container_width=True)
    with col2:
        st.dataframe(longest_participation, use_container_width=True)

    # Merging captains_df and players_df to include captaincy information
       
    if 'Year' in players_df.columns and 'Year' in captains_df.columns:
        merged_data_captains = pd.merge(players_df, captains_df, on=['Player Name', 'Team', 'Year'], how='inner')

        # Players with the longest captaincy duration
        captain_durations = merged_data_captains.groupby(['Player Name', 'Team'])['Year'].nunique().reset_index(name='Captaincy Duration')
        longest_captaincy = captain_durations.loc[captain_durations.groupby('Team')['Captaincy Duration'].idxmax()]

        # Bar chart for captains with longest durations
        fig_longest_captains = px.bar(
            longest_captaincy, 
            x='Team', 
            y='Captaincy Duration', 
            color='Player Name', 
            title='Captains with the Longest Duration for Each Team',
            labels={'Captaincy Duration': 'Number of Years as Captain', 'Team': 'Team'},
            text='Player Name',
            template='plotly_white',
            color_discrete_sequence=px.colors.sequential.Viridis  # Use Viridis palette
        )
    else:
        st.write("Error: 'Year' column is missing in either players_df or captains_df.")

    # Merging all_matches_data_df and players_df to calculate wins
    merged_data = pd.merge(all_matches_data_df, players_df, how='inner', left_on='Winner', right_on='Team')
    player_wins = merged_data.groupby(['Player Name', 'Team']).size().reset_index(name='Wins')

    # Top players by wins
    top_players_by_wins = player_wins.loc[player_wins.groupby('Team')['Wins'].idxmax()]

    # Funnel Plot for players with maximum wins
    
    fig_funnel = px.funnel(
        top_players_by_wins, 
        x='Player Name', 
        y='Wins', 
        color='Team', 
        title='Wins by Top Player for Each Team',
        labels={'Wins': 'Number of Wins', 'Player Name': 'Player Name'},
        template='plotly_white',
        color_discrete_sequence=px.colors.sequential.Viridis  # Use Viridis palette
    )
    st.subheader("Top Match Winners")
    st.markdown("""
    Find out which players secured the most wins for their teams. This section highlights the standout performers who consistently delivered victories, earning them a spot in cricketing glory.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_funnel, use_container_width=True)
    with col2:
        st.dataframe(top_players_by_wins, use_container_width=True)
    
    #Captains
    st.subheader("Captains Who Led the Way")
    st.markdown("""
        Leadership matters! Discover the captains who led their teams for the longest time in World Cups. These individuals not only inspired their teammates but also etched their names in cricketing history.
        """)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_longest_captains, use_container_width=True)
    with col2:
        st.dataframe(longest_captaincy, use_container_width=True)



############################################################################################################################



# Predictions
elif ui_section == "Forecasting the Next Champions":
    # Placeholder for updated dataset
    if 'updated_wc_final_data_df' not in locals():
        st.error("Dataset `updated_wc_final_data_df` is not loaded. Please load the dataset.")
    else:
        # Apply Feature Engineering
        if 'Home Advantage' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Home Advantage'] = updated_wc_final_data_df.apply(
                lambda row: 1 if row['Team1'] in row['Ground'] or row['Team2'] in row['Ground'] else 0, axis=1
            )

        if 'Normalized Batting Difference' not in updated_wc_final_data_df.columns:
            scaler = MinMaxScaler()
            updated_wc_final_data_df[['Normalized Batting Difference', 'Normalized Bowling Difference']] = scaler.fit_transform(
                updated_wc_final_data_df[['Batting Ranking Difference', 'Bowling Ranking Difference']]
            )

        if 'Rolling Win %' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Rolling Win %'] = updated_wc_final_data_df.groupby('Team1')['Team1 win % over Team2'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        if 'Rolling Margin (Runs)' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Rolling Margin (Runs)'] = updated_wc_final_data_df.groupby('Team1')['Margin (Runs)'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        if 'Rolling Margin (Wickets)' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Rolling Margin (Wickets)'] = updated_wc_final_data_df.groupby('Team1')['Margin (Wickets)'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        if 'Team1 Strength Index' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Team1 Strength Index'] = (
                updated_wc_final_data_df['Team1 Avg Batting Ranking'] * 0.5 +
                updated_wc_final_data_df['Team1 Avg Bowling Ranking'] * 0.5
            )

        if 'Team2 Strength Index' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Team2 Strength Index'] = (
                updated_wc_final_data_df['Team2 Avg Batting Ranking'] * 0.5 +
                updated_wc_final_data_df['Team2 Avg Bowling Ranking'] * 0.5
            )

        if 'Batting Disparity' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Batting Disparity'] = (
                updated_wc_final_data_df['Team1 Avg Batting Ranking'] - updated_wc_final_data_df['Team2 Avg Batting Ranking']
            )

        if 'Bowling Disparity' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Bowling Disparity'] = (
                updated_wc_final_data_df['Team1 Avg Bowling Ranking'] - updated_wc_final_data_df['Team2 Avg Bowling Ranking']
            )

        # Forecasting Section
        st.subheader("Predictions: Who Will Reign Supreme?")
        st.write("""
        The battle for cricket supremacy intensifies as we bring you an exciting glimpse into the future. 
        Imagine every team competing in a dramatic round-robin format, each match filled with edge-of-the-seat moments. 
        Who will emerge as the ultimate champion of the ICC Men's T20 World Cup 2026? 

        With insights derived from historical performances, team strengths, and other key factors, 
        this prediction reveals the team most likely to etch their name in cricketing glory.
        Let's dive into the results!
    """)
        # Unique teams from the dataset
        teams = updated_wc_final_data_df['Team1'].unique()

        # Simulate Matchups
        matchups = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                matchups.append({"Team_A": teams[i], "Team_B": teams[j]})

        simulated_data = pd.DataFrame(matchups)

        # Calculate Historical Averages
        team_stats = updated_wc_final_data_df.groupby('Team1').agg({
            'Team1 Strength Index': 'mean',
            'Batting Disparity': 'mean',
            'Bowling Disparity': 'mean',
            'Normalized Batting Difference': 'mean',
            'Normalized Bowling Difference': 'mean',
            'Rolling Win %': 'mean',
            'Rolling Margin (Runs)': 'mean',
            'Rolling Margin (Wickets)': 'mean',
            'Home Advantage': 'mean'
        }).reset_index()

        # Merge Features for Both Teams
        simulated_data = simulated_data.merge(team_stats, how='left', left_on='Team_A', right_on='Team1')
        simulated_data = simulated_data.merge(
            team_stats, how='left', left_on='Team_B', right_on='Team1', suffixes=('_A', '_B')
        )

        # Create Relative Difference Features
        simulated_data['Team1 Strength Index'] = simulated_data['Team1 Strength Index_A']
        simulated_data['Team2 Strength Index'] = simulated_data['Team1 Strength Index_B']
        simulated_data['Batting Disparity'] = simulated_data['Batting Disparity_A'] - simulated_data['Batting Disparity_B']
        simulated_data['Bowling Disparity'] = simulated_data['Bowling Disparity_A'] - simulated_data['Bowling Disparity_B']
        simulated_data['Normalized Batting Difference'] = simulated_data['Normalized Batting Difference_A'] - simulated_data['Normalized Batting Difference_B']
        simulated_data['Normalized Bowling Difference'] = simulated_data['Normalized Bowling Difference_A'] - simulated_data['Normalized Bowling Difference_B']
        simulated_data['Rolling Win %'] = simulated_data['Rolling Win %_A'] - simulated_data['Rolling Win %_B']
        simulated_data['Rolling Margin (Runs)'] = simulated_data['Rolling Margin (Runs)_A'] - simulated_data['Rolling Margin (Runs)_B']
        simulated_data['Rolling Margin (Wickets)'] = simulated_data['Rolling Margin (Wickets)_A'] - simulated_data['Rolling Margin (Wickets)_B']
        simulated_data['Home Advantage'] = simulated_data['Home Advantage_A'] - simulated_data['Home Advantage_B']

        # Define Features for Prediction
        features = [
            'Team1 Strength Index',
            'Team2 Strength Index',
            'Batting Disparity',
            'Bowling Disparity',
            'Normalized Batting Difference',
            'Normalized Bowling Difference',
            'Rolling Win %',
            'Rolling Margin (Runs)',
            'Rolling Margin (Wickets)',
            'Home Advantage'
        ]

        # Train a Random Forest model
        updated_wc_final_data_df['Target'] = updated_wc_final_data_df['Winner'].apply(
            lambda x: 0 if x == 'Team1' else 1
        )
        X = updated_wc_final_data_df[features]
        y = updated_wc_final_data_df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)

        # Predict Match Outcomes
        simulated_data['Predicted Team_A Win'] = rf_clf.predict(simulated_data[features])

        # Aggregate Results
        win_counts = pd.concat([
            simulated_data.loc[simulated_data['Predicted Team_A Win'] == 1, 'Team_A'],
            simulated_data.loc[simulated_data['Predicted Team_A Win'] == 0, 'Team_B']
        ]).value_counts()

        # Plot the Predicted Win Counts
        predictions_fig = px.bar(
            win_counts,
            x=win_counts.index,
            y=win_counts.values,
            title="Predicted Win Counts for Each Team in ICC Men's T20 World Cup 2026",
            labels={'x': 'Teams', 'y': 'Predicted Wins'},
            color=win_counts.values,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        predictions_fig.update_layout(
            xaxis_title="Teams",
            yaxis_title="Predicted Wins",
            xaxis_tickangle=-45
        )

        # Display Predictions
        st.plotly_chart(predictions_fig)
        st.write(f"### Predictions: The team most likely to win the ICC Men's T20 World Cup 2026 is **{win_counts.idxmax()}**!")







#############################################################################################################################


# Search for your favourite teams and players
elif ui_section == "Search Magic":
    st.subheader("Search Magic")
    st.markdown("""
    Welcome to the ultimate search tool for cricket enthusiasts! This section is all about putting you in control to explore your favorite teams, players, and matches. Whether you’re looking to relive the glory of your team in a specific year, find the captains who led them to victory, or discover how a player has performed over the years, this is the place for you.

    **Find Your Team’s Legacy 🌏:**  
    Type your favorite team's name and a specific year (or just one of them) to uncover their matches, the teams they faced, and the grounds where they played. You can also dive deeper into details like batting and bowling averages, match margins, and even the captain who led the team. A colorful and interactive display will bring all this information to life, giving you a fresh perspective on your team’s performance.

    **Explore Player Achievements 🏅:**  
    Enter the name of a cricketing star, and let the app take you on a journey through their career. Find out which teams they played for, the number of years they participated, and the span of their contributions. You’ll also discover if they ever wore the captain’s armband and led their team to glory.

    **What You’ll See:**  
    - An interactive visual showcasing how batting and bowling averages compare for your team across matches, with the size and color of the markers reflecting the match margins (whether it was a nail-biter or a landslide victory).
    - A detailed table summarizing match stats, including opponents, captains, and outcomes.  
    - Player-specific highlights, showing their years of participation and leadership roles, if any.  

    It’s your personal gateway to explore cricket history, filled with insights and moments that matter the most to you. Dive in and uncover the magic!
    """)
    # Load the datasets
    players_df = pd.read_csv(players_url)
    wc_final_data_df = pd.read_csv(final_dataset_url)
    captains_df = pd.read_csv(captains_url)

    # Separate 'Match Date' into 'Day', 'Month', 'Year'
    wc_final_data_df['Match Date'] = pd.to_datetime(wc_final_data_df['Match Date'], errors='coerce')
    wc_final_data_df['Day'] = wc_final_data_df['Match Date'].dt.day
    wc_final_data_df['Month'] = wc_final_data_df['Match Date'].dt.month
    wc_final_data_df['Year'] = wc_final_data_df['Match Date'].dt.year

    # Function to extract runs from the 'Margin' column
    def extract_runs(margin):
        if isinstance(margin, str) and 'runs' in margin:
            return float(margin.split()[0])
        return None

    # Function to extract wickets from the 'Margin' column
    def extract_wickets(margin):
        if isinstance(margin, str) and 'wickets' in margin:
            return float(margin.split()[0])
        return None

    # Apply the functions to extract 'Margin (Runs)' and 'Margin (Wickets)'
    wc_final_data_df['Margin (Runs)'] = wc_final_data_df['Margin'].apply(extract_runs)
    wc_final_data_df['Margin (Wickets)'] = wc_final_data_df['Margin'].apply(extract_wickets)

    # Team search bar
    team_name = st.text_input("Search for a team (optional):")

    # Year search bar
    year = st.text_input("Enter the year (optional):")

    # Initialize an empty DataFrame for the filtered data
    filtered_data = pd.DataFrame()

    # Apply filters based on the user's input
    if team_name and year:
        year = int(year)
        filtered_data_team1 = wc_final_data_df[(wc_final_data_df['Team1'].str.contains(team_name, case=False, na=False)) & (wc_final_data_df['Year'] == year)]
        filtered_data_team2 = wc_final_data_df[(wc_final_data_df['Team2'].str.contains(team_name, case=False, na=False)) & (wc_final_data_df['Year'] == year)]
        filtered_data = pd.concat([filtered_data_team1, filtered_data_team2])
    elif team_name:
        filtered_data_team1 = wc_final_data_df[wc_final_data_df['Team1'].str.contains(team_name, case=False, na=False)]
        filtered_data_team2 = wc_final_data_df[wc_final_data_df['Team2'].str.contains(team_name, case=False, na=False)]
        filtered_data = pd.concat([filtered_data_team1, filtered_data_team2])
    elif year:
        year = int(year)
        filtered_data = wc_final_data_df[wc_final_data_df['Year'] == year]
    else:
        st.write("Please enter a team name, a year, or both to search.")

    # Process the filtered data
    if not filtered_data.empty:
        filtered_data['Team'] = np.where(filtered_data['Team1'].str.contains(team_name, case=False, na=False), filtered_data['Team1'], filtered_data['Team2'])
        filtered_data['Against'] = np.where(filtered_data['Team'] == filtered_data['Team1'], filtered_data['Team2'], filtered_data['Team1'])

        filtered_data['Batting Avg'] = np.where(filtered_data['Team'] == filtered_data['Team1'], filtered_data['Team1 Avg Batting Ranking'], filtered_data['Team2 Avg Batting Ranking'])
        filtered_data['Bowling Avg'] = np.where(filtered_data['Team'] == filtered_data['Team1'], filtered_data['Team1 Avg Bowling Ranking'], filtered_data['Team2 Avg Bowling Ranking'])

        # Handle margins for runs and wickets
        filtered_data['Margin Type'] = np.where(filtered_data['Margin (Runs)'].notna(), 'Runs', 'Wickets')
        filtered_data['Margin Numeric'] = np.where(filtered_data['Margin (Runs)'].notna(), filtered_data['Margin (Runs)'], filtered_data['Margin (Wickets)'])

        # Fill any missing values in 'Margin Numeric' with 0
        filtered_data['Margin Numeric'] = filtered_data['Margin Numeric'].fillna(0)

        match_count_by_ground = filtered_data.groupby('Ground').size().reset_index(name=f"Number of Matches Played by {team_name}")
        filtered_data = pd.merge(filtered_data, match_count_by_ground, on='Ground', how='left')

        filtered_data = pd.merge(
            filtered_data,
            captains_df[['Team', 'Player Name', 'Year']],
            left_on=['Team', 'Year'],
            right_on=['Team', 'Year'],
            how='left'
        )

        filtered_data.rename(columns={'Player Name': 'Captain'}, inplace=True)

        # Plot the scatter plot with Viridis palette
        fig_scatter = px.scatter(
            filtered_data,
            x='Batting Avg',
            y='Bowling Avg',
            size='Margin Numeric',  # Use 'Margin Numeric' based on runs or wickets
            color='Margin Numeric',  # Use numeric margin for Viridis palette
            hover_name='Team',
            hover_data={
                'Year': True,
                'Winner': True,
                'Against': True,
                'Captain': True,
                f"Number of Matches Played by {team_name}": True,
                'Margin Type': True  # Show the type of margin (runs or wickets)
            },
            title=f'Team Performance (Batting vs Bowling Average by Ground) for {team_name}',
            labels={'Batting Avg': 'Batting Average', 'Bowling Avg': 'Bowling Average', 'Margin Numeric': 'Match Margin'},
            color_continuous_scale='Viridis'  # Use Viridis palette for continuous data
        )

        st.dataframe(filtered_data[['Year', 'Team', 'Against', 'Winner', 'Ground', 'Captain', 'Margin Type', 'Margin Numeric']])
        st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.write("No matches found for the provided input.")

    # Player search bar, independent of the team search
    st.markdown("##### Search for a Player")
    player_name = st.text_input("Enter the player name:")

    # Check if the input player name is in the player_df DataFrame
    if player_name:
        player_data = players_df[players_df['Player Name'].str.contains(player_name, case=False, na=False)]
        
        if not player_data.empty:
            teams = player_data['Team'].unique()

            for team in teams:
                team_data = player_data[player_data['Team'] == team]
                total_years = len(team_data['Year'].unique())
                year_range = f"{team_data['Year'].min()} - {team_data['Year'].max()}"

                st.write(f"**Player:** {player_name}")
                st.write(f"**Team:** {team}")
                st.write(f"**Total Number of Years in Team:** {total_years}")
                st.write(f"**Year Range in Team:** {year_range}")

                captain_data = captains_df[captains_df['Player Name'].str.contains(player_name, case=False, na=False) & (captains_df['Team'] == team)]
                if not captain_data.empty:
                    captain_years = captain_data['Year'].tolist()
                    st.write(f"**Captain for {team} in the following years:** {', '.join(map(str, captain_years))}")
                else:
                    st.write(f"{player_name} was not a captain for {team}.")
        else:
            st.write("Player not found in the dataset.")
    else:
        st.write("Please enter a player name to search.")




############################################################################################################################



if ds_section  == "Welcome!":    
    st.title("🏏 Welcome to the Ultimate Men's T20 World Cup Analysis App! 🏆")
    st.subheader('Cricket Fever: Data Edition') 

        # Displaying the GIF from the raw GitHub link
    gif_url = "https://raw.githubusercontent.com/Shamsvi/Streamlit_Mens_T-20_Cricket_WorldCup_2007-2024/main/giphy.gif"
    st.image(gif_url, use_container_width=True)

    st.markdown("""
# Step into the Magical World of Cricket Analytics! 🏏✨

Welcome to your gateway where **data meets discovery**! This app is your secret laboratory to uncover the science behind every six, every wicket, and every thrilling World Cup moment. Whether you’re a curious mind or a data enthusiast, get ready to explore, experiment, and predict like never before.

## **Here’s What Awaits You:**  

📊 **About the Data**  
  Peek behind the curtains and understand the foundation of this exploration—our dataset! Learn about the features, variables, and data sources that drive the analytics.  

📈 **Cricket Stats**  
  Dive into the **Distribution of Features** and uncover fascinating trends and stories hidden within the data. From player performances to team dynamics, visualize how the numbers come alive to tell the story of the World Cup.

🛠️ **Feature Factory**  
  The lab where the magic happens! This section breaks down the building blocks of cricket analytics, showing you how raw data transforms into meaningful insights. Discover engineered features that enhance predictions and uncover hidden relationships.

🤖 **Modeling the Game: Unveiling Predictions**  
  Take a step into the machine learning arena. See how advanced models like Logistic Regression, Random Forest, and XGBoost are trained to predict match outcomes. Dive into the performance metrics to understand what drives accurate predictions.

🔮 **Forecasting the Next Champions**  
  Enter the **Predictor’s Playground** and try your hand at being the ultimate cricket soothsayer. With the power of advanced analytics, see which team has the best odds of taking home the next World Cup trophy.  

## **Who is This For?**  
If you love playing with numbers, solving mysteries, or just want to see the “behind-the-scenes” magic of cricket analytics, this is your playground. It’s not just stats; it’s the art of making every number count!

Get ready to blend cricket passion with data brilliance, and let’s create some magic together! 🏆✨  
""")



        # Footer or call-to-action
    st.markdown("---")
    st.markdown("### 🏏 Let the cricket journey begin! Navigate using the sidebar to explore more insights.")




############################################################################################################################


elif ds_section == "About the Data":
    st.subheader("About the Data")
    st.markdown("""
    Welcome to the **Dataset Overview** section! Here’s everything you need to know about the data that powers this app.  
    
    Our dataset is primarily sourced from [Kaggle](https://www.kaggle.com/datasets/kamalisrani/mens-t20-cwc-dataset-2007-2004), a hub of diverse and high-quality data. It’s packed with fascinating cricketing stats and insights, allowing us to analyze and predict the thrilling dynamics of the Men’s T20 World Cup.  

    Additionally, we’ve manually compiled data on team captains spanning the years 2007 to 2024, enriching our exploration of leadership patterns and their impact on team performance.   
                
    Here’s a closer look at the data that powers our analysis:

    ### 1. **All T20 World Cup Matches Results**
    This dataset captures the essence of every match contested in all World Cup editions. Here’s what it includes:
    - **Team1**: One of the teams in the match.
    - **Team2**: The other team in the match.
    - **Winner**: The winner of the contest. If it’s a tie, it says "tied", and for abandoned matches, it says "no result".
    - **Margin**: The victory margin, either in runs or wickets.
    - **Ground**: Where the match was played.
    - **Match Date**: When the match was played.
    - **T-20 Int Match**: The international match number for T20 cricket.
    """)
    # Display Matches Dataset
    st.dataframe(all_matches_data_df, key='matches_dataframe')

    st.markdown("""
    ---

    ### 2. **All T20 World Cup Players List**
    Ever wondered who suited up for each World Cup edition? This dataset gives you:
    - **Team**: The name of the participating country.
    - **Year**: The year they participated.
    - **Player Name**: Names of players representing their teams.
    """)
    # Display Players Dataset
    st.dataframe(players_df, key='players_dataframe')

    st.markdown("""
    ---

    ### 3. **WC Final Dataset**
    This is the ultimate cricket dataset, with 16 attributes combining match results with advanced stats for accurate predictions. Highlights include:
    - **Teams and Winners**: Team1, Team2, and match winners.
    - **Player Rankings**: Average batting and bowling rankings of both teams, derived from ICC rankings.
    - **Historical Stats**: Total World Cup participations and wins for each team.
    - **Win Percentages**: Team1’s win percentage over Team2.
    - **Margin of Victory**: Either in runs or wickets.
    - **Web-Scraped Data**: Rankings scraped from reliable ICC sources one day before matches.
    """)
    # Display Final Dataset
    st.dataframe(wc_final_data_df, key='final_dataset_dataframe')

    st.markdown("""
    ---

    ### 4. **All Captains (2007-2024)**
    A manual compilation of all captains leading their teams in World Cups from 2007 to 2024.
    """)
    # Display Captains Dataset
    captains_df_cleaned = captains_df.loc[:, ~captains_df.columns.str.contains('^Unnamed')]

        # Replace '-' in the Player Name column with None
    if 'Player Name' in captains_df_cleaned.columns:
            captains_df_cleaned.loc[:, 'Player Name'] = captains_df_cleaned['Player Name'].replace('-', None)


        # Display cleaned Captains Dataset
    st.dataframe(captains_df_cleaned, key='captains_dataframe_cleaned')


    st.markdown("""
            ---
    
    From match results to player stats, and even a list of legendary captains, these datasets fuel the insights and visualizations you’ll see in this app. Dive in and explore the cricketing data that tells the story of T20 World Cup history!
    """)





############################################################################################################################


elif ds_section == "Cricket Stats":
    st.subheader("Cricket Stats: Hidden Stories Behind the Numbers 🏏")

    # Match Margins Split by Runs and Wickets
    st.write(
        "### The Drama of Match Margins\n"
        "Let's dive into the nail-biting world of cricket outcomes! "
        "Here, you can explore the match margins split into runs and wickets to see just how close (or one-sided) games have been. "
        "Lower values mean edge-of-your-seat thrillers, while higher values reflect dominant wins."
    )

    # Margin by Runs
    fig_margin_runs = px.histogram(
        wc_final_data_df.sort_values(by="Margin (Runs)", ascending=False),
        x="Margin (Runs)",
        nbins=20,
        title="Distribution of Match Margins (Runs)",
        labels={"x": "Margin (Runs)", "y": "Frequency"},
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_margin_runs.update_traces(marker_line_color="black", marker_line_width=1.5)
    fig_margin_runs.update_layout(xaxis_title="Margin (Runs)", yaxis_title="Frequency", width=600, height=400)

    # Margin by Wickets
    fig_margin_wickets = px.histogram(
        wc_final_data_df.sort_values(by="Margin (Wickets)", ascending=False),
        x="Margin (Wickets)",
        nbins=20,
        title="Distribution of Match Margins (Wickets)",
        labels={"x": "Margin (Wickets)", "y": "Frequency"},
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_margin_wickets.update_traces(marker_line_color="black", marker_line_width=1.5)
    fig_margin_wickets.update_layout(xaxis_title="Margin (Wickets)", yaxis_title="Frequency", width=600, height=400)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_margin_runs, use_container_width=True)
    with col2:
        st.plotly_chart(fig_margin_wickets, use_container_width=True)

    # Team 1 and Team 2 Batting Rankings
    st.write(
        "### Team Batting Rankings\n"
        "Next up, we explore the batting prowess of teams. Compare how Team 1 and Team 2 "
        "stack up based on their average batting rankings. Are they top-class hitters or underdogs? "
        "These visuals give you the answers!"
    )

    # Team 1 Avg Batting Ranking
    fig_batting_team1 = px.histogram(
        wc_final_data_df.sort_values(by="Team1 Avg Batting Ranking", ascending=False),
        x="Team1 Avg Batting Ranking",
        nbins=20,
        title="Distribution of Team 1 Avg Batting Ranking",
        labels={"x": "Batting Ranking (Team 1)", "y": "Frequency"},
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_batting_team1.update_traces(marker_line_color="black", marker_line_width=1.5)
    fig_batting_team1.update_layout(xaxis_title="Batting Ranking (Team 1)", yaxis_title="Frequency", width=600, height=400)

    # Team 2 Avg Batting Ranking
    fig_batting_team2 = px.histogram(
        wc_final_data_df.sort_values(by="Team2 Avg Batting Ranking", ascending=False),
        x="Team2 Avg Batting Ranking",
        nbins=20,
        title="Distribution of Team 2 Avg Batting Ranking",
        labels={"x": "Batting Ranking (Team 2)", "y": "Frequency"},
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_batting_team2.update_traces(marker_line_color="black", marker_line_width=1.5)
    fig_batting_team2.update_layout(xaxis_title="Batting Ranking (Team 2)", yaxis_title="Frequency", width=600, height=400)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_batting_team1, use_container_width=True)
    with col2:
        st.plotly_chart(fig_batting_team2, use_container_width=True)

    # Ranking Differences
    st.write(
        "### Ranking Differences\n"
        "Who dominates the pitch, and who struggles? These charts reveal the differences "
        "in batting and bowling rankings between teams, shedding light on where each team "
        "stands and how level the playing field really is."
    )

    # Batting Ranking Difference
    fig_batting_diff = px.histogram(
        wc_final_data_df.sort_values(by="Batting Ranking Difference", ascending=False),
        x="Batting Ranking Difference",
        nbins=20,
        title="Distribution of Batting Ranking Difference",
        labels={"x": "Batting Ranking Difference", "y": "Frequency"},
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_batting_diff.update_traces(marker_line_color="black", marker_line_width=1.5)
    fig_batting_diff.update_layout(xaxis_title="Batting Ranking Difference", yaxis_title="Frequency", width=600, height=400)

    # Bowling Ranking Difference
    fig_bowling_diff = px.histogram(
        wc_final_data_df.sort_values(by="Bowling Ranking Difference", ascending=False),
        x="Bowling Ranking Difference",
        nbins=20,
        title="Distribution of Bowling Ranking Difference",
        labels={"x": "Bowling Ranking Difference", "y": "Frequency"},
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_bowling_diff.update_traces(marker_line_color="black", marker_line_width=1.5)
    fig_bowling_diff.update_layout(xaxis_title="Bowling Ranking Difference", yaxis_title="Frequency", width=600, height=400)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_batting_diff, use_container_width=True)
    with col2:
        st.plotly_chart(fig_bowling_diff, use_container_width=True)

    # Trends Over Time
    st.write(
        "### Trends Over Time\n"
        "How has the competition changed over the years? This chart tracks the differences "
        "in batting and bowling rankings through time, showcasing the evolving dynamics of cricket's rivalries."
    )

    # Line Plot for Trends
    fig_trends = go.Figure()
    fig_trends.add_trace(
        go.Scatter(
            x=wc_final_data_df["Match Year"],
            y=wc_final_data_df["Batting Ranking Difference"],
            mode="lines",
            name="Batting Ranking Difference",
            line=dict(color="rgb(68, 1, 84)"),
            hovertext=wc_final_data_df.apply(
                lambda row: f"Team 1: {row['Team1']}<br>Team 2: {row['Team2']}", axis=1
            ),
            hoverinfo="text+y",
        )
    )
    fig_trends.add_trace(
        go.Scatter(
            x=wc_final_data_df["Match Year"],
            y=wc_final_data_df["Bowling Ranking Difference"],
            mode="lines",
            name="Bowling Ranking Difference",
            line=dict(color="rgb(33, 145, 140)", dash="dash"),
            hovertext=wc_final_data_df.apply(
                lambda row: f"Team 1: {row['Team1']}<br>Team 2: {row['Team2']}", axis=1
            ),
            hoverinfo="text+y",
        )
    )
    fig_trends.update_layout(
        title="Ranking Differences Over Time",
        xaxis_title="Match Year",
        yaxis_title="Ranking Difference",
        legend_title="Ranking Type",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig_trends)



     



############################################################################################################################


## Feature Engineering and Data Exploration

elif ds_section == "Feature Factory":
    st.subheader("Feature Factory: Cricket Analytics Unveiled")
    st.write(
        """
        In this section, we unveil the hidden stories behind cricket matches through engineered features. 
        These features offer insights into team performance, match dynamics, and more. Explore one or 
        multiple features to uncover patterns, correlations, and trends that make cricket analytics fascinating!

        Here's what you can expect:
        - **Feature Correlation**: This section showcases a correlation matrix for selected numeric features, highlighting how they are interrelated. This helps identify strong positive or negative relationships between features, which can be pivotal for modeling and understanding the data dynamics.
        - **Feature Analysis**: For each selected numeric feature, you will see a histogram illustrating the distribution of values. This allows you to understand the variability and spread of a feature across different matches.
        - **Feature Relationship**: If exactly two numeric features are selected, a scatter plot is displayed to analyze how these features interact with each other. This visualization can uncover trends, clusters, or anomalies.

        **How It Works**:
        - We start by applying feature engineering techniques, creating new columns like 'Home Advantage,' 'Normalized Batting Difference,' and 'Winning Margin Type,' among others.
        - Users can select specific features using the sidebar to focus on areas of interest.
        - Depending on the selected features, visualizations are dynamically generated, offering insights into data relationships, distributions, and patterns.
        """
    )
    
    # Apply Feature Engineering if not already done
    if 'Home Advantage' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Home Advantage'] = updated_wc_final_data_df.apply(
            lambda row: 1 if row['Team1'] in row['Ground'] or row['Team2'] in row['Ground'] else 0, axis=1
        )

    if 'Normalized Batting Difference' not in updated_wc_final_data_df.columns:
        scaler = MinMaxScaler()
        updated_wc_final_data_df[['Normalized Batting Difference', 'Normalized Bowling Difference']] = scaler.fit_transform(
            updated_wc_final_data_df[['Batting Ranking Difference', 'Bowling Ranking Difference']]
        )

    if 'Winning Margin Type' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Winning Margin Type'] = updated_wc_final_data_df.apply(
            lambda row: 'Dominant Win' if row['Margin (Runs)'] > 20 or row['Margin (Wickets)'] > 5 else 
                        ('Close Match' if row['Margin (Runs)'] > 0 or row['Margin (Wickets)'] > 0 else 'No Result'), 
            axis=1
        )

    if 'Rolling Win %' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Rolling Win %'] = updated_wc_final_data_df.groupby('Team1')['Team1 win % over Team2'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

    if 'Rolling Margin (Wickets)' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Rolling Margin (Wickets)'] = updated_wc_final_data_df.groupby('Team1')['Margin (Wickets)'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

    if 'Rolling Margin (Runs)' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Rolling Margin (Runs)'] = updated_wc_final_data_df.groupby('Team1')['Margin (Runs)'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

    if 'Team1 Strength Index' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Team1 Strength Index'] = (
            updated_wc_final_data_df['Team1 Avg Batting Ranking'] * 0.5 +
            updated_wc_final_data_df['Team1 Avg Bowling Ranking'] * 0.5
        )

    if 'Team2 Strength Index' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Team2 Strength Index'] = (
            updated_wc_final_data_df['Team2 Avg Batting Ranking'] * 0.5 +
            updated_wc_final_data_df['Team2 Avg Bowling Ranking'] * 0.5
        )

    if 'High Pressure Win' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Match Importance'] = updated_wc_final_data_df['T-20 Int Match'].apply(
            lambda x: 'High' if x > 300 else 'Low'
        )
        updated_wc_final_data_df['Team1 Win'] = updated_wc_final_data_df['Winner'].apply(
            lambda x: 1 if x == 'Team1' else 0
        )
        updated_wc_final_data_df['High Pressure Win'] = updated_wc_final_data_df.apply(
            lambda row: 1 if row['Match Importance'] == 'High' and row['Team1 Win'] == 1 else 0, axis=1
        )

    if 'Season' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df['Season'] = updated_wc_final_data_df['Match Month'].apply(
            lambda x: 'Winter' if x in [12, 1, 2] else 
                      'Spring' if x in [3, 4, 5] else 
                      'Summer' if x in [6, 7, 8] else 'Fall'
        )
        # Encode 'Season' into numeric values
        season_mapping = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
        updated_wc_final_data_df['Season'] = updated_wc_final_data_df['Season'].map(season_mapping)
    # Ensure `Team1 Win` is correctly derived
    if 'Team1 Win' not in updated_wc_final_data_df.columns:
        updated_wc_final_data_df.loc[:, 'Team1 Win'] = updated_wc_final_data_df.apply(
                 lambda row: 1 if row['Winner'] == row['Team1'] else 0, axis=1
                )


    # Verify the values in `Team1 Win`
    # Verify the values in `Team1 Win`
    if updated_wc_final_data_df['Winner'].nunique() <= 1:
        st.error("The target column `Team1 Win` does not have enough class variability (e.g., only 0s or 1s). Ensure the `Winner` column is correctly populated.")


    # Feature Selector
    st.sidebar.header("Feature Selector")
    available_features = [
        "Home Advantage",
        "Normalized Batting Difference",
        "Normalized Bowling Difference",
        "Rolling Win %",
        "Rolling Margin (Runs)",
        "Rolling Margin (Wickets)",
        "Team1 Strength Index",
        "Team2 Strength Index",
        "Batting Disparity",
        "Bowling Disparity",
        "High Pressure Win",
        "Season",  # Include the numeric Season column
    ]
    selected_features = st.sidebar.multiselect(
        "Select Features to Explore",
        options=available_features,
        default=["Normalized Bowling Difference", "Normalized Batting Difference"]
    )
    
    if selected_features:
        # Filter selected features to only include numeric columns
        numeric_features = updated_wc_final_data_df[selected_features].select_dtypes(include='number').columns.tolist()

        if numeric_features:
            # Display Correlation Matrix if multiple numeric features are selected
            if len(numeric_features) > 1:
                st.subheader("Feature Correlation")
                correlation_data = updated_wc_final_data_df[numeric_features].corr()
                fig_corr = px.imshow(
                    correlation_data,
                    title="Feature Correlation Matrix",
                    labels={"color": "Correlation Coefficient"},
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text_auto=True,
                )
                st.plotly_chart(fig_corr)

            # Display histograms for each numeric feature
            st.subheader("Feature Analysis")
            for feature in numeric_features:
                fig_hist = px.histogram(
                    updated_wc_final_data_df,
                    x=feature,
                    nbins=20,
                    title=f"Distribution of {feature}",
                    labels={"x": feature, "y": "Frequency"},
                    opacity=0.7,
                    color_discrete_sequence=px.colors.sequential.Viridis,
                )
                st.plotly_chart(fig_hist)

            # Scatter Plot for Relationships if two numeric features are selected
            if len(numeric_features) == 2:
                st.subheader("Feature Relationship")
                feature_x, feature_y = numeric_features
                fig_scatter = px.scatter(
                    updated_wc_final_data_df,
                    x=feature_x,
                    y=feature_y,
                    title=f"Relationship Between {feature_x} and {feature_y}",
                    labels={feature_x: feature_x, feature_y: feature_y},
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    opacity=0.7,
                )
                st.plotly_chart(fig_scatter)
        else:
            st.warning("None of the selected features are numeric. Please select numeric features to analyze correlations or distributions.")
    else:
        st.warning("Please select at least one feature to explore.")












############################################################################################################################





import streamlit as st
import pandas as pd
import os
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Paths to model files
MODEL_DIR = "./models/"
LOG_REG_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Preprocessing Function
@st.cache_data
def preprocess_data(df, required_features):
    missing_features = [feature for feature in required_features if feature not in df.columns]
    if missing_features:
        return None, None, None, None, missing_features

    X = df[required_features]
    y = df['Team1 Win']
    if y.nunique() <= 1:
        return None, None, None, None, ["Target variable 'Team1 Win' does not have enough variability."]
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
    )
    return X_train, X_test, y_train, y_test, None

# Train and Save Models
def train_and_save_models(df, required_features):
    X_train, X_test, y_train, y_test, issues = preprocess_data(df, required_features)
    if issues:
        for issue in issues:
            st.error(issue)
        st.stop()

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    joblib.dump(log_reg, LOG_REG_MODEL_PATH)

    # Random Forest
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, RF_MODEL_PATH)

    # XGBoost
    xgb_clf = XGBClassifier(eval_metric="logloss", random_state=42)
    xgb_clf.fit(X_train, y_train)
    joblib.dump(xgb_clf, XGB_MODEL_PATH)

    return log_reg, rf_clf, xgb_clf

# Load Models
@st.cache_resource
def load_models():
    if not os.path.exists(LOG_REG_MODEL_PATH) or not os.path.exists(RF_MODEL_PATH) or not os.path.exists(XGB_MODEL_PATH):
        return None, None, None
    try:
        log_reg = joblib.load(LOG_REG_MODEL_PATH)
        rf_clf = joblib.load(RF_MODEL_PATH)
        xgb_clf = joblib.load(XGB_MODEL_PATH)
        return log_reg, rf_clf, xgb_clf
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Modeling Section
if "Modeling the Game: Unveiling Predictions" in st.sidebar.radio("Sections", ["Modeling the Game: Unveiling Predictions"]):

    # Check Winner Column
    if 'Winner' in updated_wc_final_data_df.columns:
        updated_wc_final_data_df.loc[:, 'Team1 Win'] = updated_wc_final_data_df.apply(
            lambda row: 1 if row['Winner'] == row['Team1'] else 0, axis=1
        )
    else:
        st.error("The 'Winner' column is missing or incorrectly populated in the dataset.")
        st.stop()

    st.title("Modeling the Game: Unveiling Predictions")
    st.write("""
    **Step into the Analytics Dugout!**
    In this section, we use cutting-edge machine learning models to predict the outcomes of cricket matches. 
    We compare Logistic Regression, Random Forest, and XGBoost to identify the best-performing model.
    """)

    required_features = [
        'Team1 Strength Index', 'Team2 Strength Index', 
        'Batting Disparity', 'Bowling Disparity', 
        'Normalized Batting Difference', 'Normalized Bowling Difference', 
        'Rolling Win %', 'Rolling Margin (Runs)', 'Rolling Margin (Wickets)', 
        'Home Advantage'
    ]

    # Load or Train Models
    log_reg, rf_clf, xgb_clf = load_models()
    if log_reg is None or rf_clf is None or xgb_clf is None:
        st.warning("Model files not found. Training new models...")
        log_reg, rf_clf, xgb_clf = train_and_save_models(updated_wc_final_data_df, required_features)

    # Preprocess Data
    X_train, X_test, y_train, y_test, issues = preprocess_data(updated_wc_final_data_df, required_features)
    if issues:
        for issue in issues:
            st.error(issue)
        st.stop()

    # Logistic Regression Evaluation
    st.subheader("Logistic Regression")
    y_pred_log_reg = log_reg.predict(X_test)
    log_reg_metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred_log_reg) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred_log_reg) * 100, 2),
        "Recall": round(recall_score(y_test, y_pred_log_reg) * 100, 2),
        "F1-Score": round(f1_score(y_test, y_pred_log_reg) * 100, 2)
    }
    st.write(pd.DataFrame(log_reg_metrics, index=["Value"]).T)
    st.write("""
    Logistic Regression is a linear model that predicts match outcomes based on probabilities. 
    It works best when the relationship between the features and the target variable is linear.
    """)

    # Random Forest Evaluation
    st.subheader("Random Forest")
    y_pred_rf = rf_clf.predict(X_test)
    rf_metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred_rf) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred_rf) * 100, 2),
        "Recall": round(recall_score(y_test, y_pred_rf) * 100, 2),
        "F1-Score": round(f1_score(y_test, y_pred_rf) * 100, 2)
    }
    st.write(pd.DataFrame(rf_metrics, index=["Value"]).T)
    st.write("""
    Random Forest is an ensemble learning method that constructs multiple decision trees to 
    predict match outcomes. It handles non-linear relationships and provides feature importance scores.
    """)

    # XGBoost Evaluation
    st.subheader("XGBoost")
    y_pred_xgb = xgb_clf.predict(X_test)
    xgb_metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred_xgb) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred_xgb) * 100, 2),
        "Recall": round(recall_score(y_test, y_pred_xgb) * 100, 2),
        "F1-Score": round(f1_score(y_test, y_pred_xgb) * 100, 2)
    }
    st.write(pd.DataFrame(xgb_metrics, index=["Value"]).T)
    st.write("""
    XGBoost is a gradient boosting method that iteratively improves predictions by focusing on errors. 
    It's highly efficient and can handle both linear and non-linear relationships effectively.
    """)

    # Model Comparison
    results_df = pd.DataFrame(
        [log_reg_metrics, rf_metrics, xgb_metrics],
        index=["Logistic Regression", "Random Forest", "XGBoost"]
    )
    st.subheader("Model Performance Comparison")
    st.write(results_df)

    # Recommendation
    best_model_name = results_df['F1-Score'].idxmax()
    st.write(f"### Recommendation: The best model for this dataset is **{best_model_name}**.")

            





############################################################################################################################



# Predictions
# Predictions
elif ds_section == "Forecasting the Next Champions":
    # Placeholder for updated dataset
    if 'updated_wc_final_data_df' not in locals():
        st.error("Dataset `updated_wc_final_data_df` is not loaded. Please load the dataset.")
    else:
        # Apply Feature Engineering
        if 'Home Advantage' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Home Advantage'] = updated_wc_final_data_df.apply(
                lambda row: 1 if row['Team1'] in row['Ground'] or row['Team2'] in row['Ground'] else 0, axis=1
            )

        if 'Normalized Batting Difference' not in updated_wc_final_data_df.columns:
            scaler = MinMaxScaler()
            updated_wc_final_data_df[['Normalized Batting Difference', 'Normalized Bowling Difference']] = scaler.fit_transform(
                updated_wc_final_data_df[['Batting Ranking Difference', 'Bowling Ranking Difference']]
            )

        if 'Rolling Win %' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Rolling Win %'] = updated_wc_final_data_df.groupby('Team1')['Team1 win % over Team2'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        if 'Rolling Margin (Runs)' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Rolling Margin (Runs)'] = updated_wc_final_data_df.groupby('Team1')['Margin (Runs)'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        if 'Rolling Margin (Wickets)' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Rolling Margin (Wickets)'] = updated_wc_final_data_df.groupby('Team1')['Margin (Wickets)'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        if 'Team1 Strength Index' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Team1 Strength Index'] = (
                updated_wc_final_data_df['Team1 Avg Batting Ranking'] * 0.5 +
                updated_wc_final_data_df['Team1 Avg Bowling Ranking'] * 0.5
            )

        if 'Team2 Strength Index' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Team2 Strength Index'] = (
                updated_wc_final_data_df['Team2 Avg Batting Ranking'] * 0.5 +
                updated_wc_final_data_df['Team2 Avg Bowling Ranking'] * 0.5
            )

        if 'Batting Disparity' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Batting Disparity'] = (
                updated_wc_final_data_df['Team1 Avg Batting Ranking'] - updated_wc_final_data_df['Team2 Avg Batting Ranking']
            )

        if 'Bowling Disparity' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Bowling Disparity'] = (
                updated_wc_final_data_df['Team1 Avg Bowling Ranking'] - updated_wc_final_data_df['Team2 Avg Bowling Ranking']
            )

        # Forecasting Section
        st.subheader("Predictions: Who Will Reign Supreme?")
        st.write("""
        Let's dive into the exciting world of cricket analytics! 
        In this section, we simulate matchups between all teams in a thrilling round-robin format. 
        Using historical averages and advanced statistics, we predict the most likely winner of 
        the ICC Men's T20 World Cup 2026. Here's how it works:
        
        **Key Metrics Used**:
        - Team strength index combining batting and bowling performance.
        - Rolling averages for recent performance trends in wins and margins.
        - Home advantage as a factor.
        - Disparities in batting and bowling rankings between teams.

        **Prediction Process**:
        - Historical data is aggregated to calculate performance metrics for each team.
        - A round-robin simulation generates matchups between all teams.
        - Using a trained Random Forest model, outcomes of each matchup are predicted.
        - The model identifies the team most likely to emerge victorious based on win counts.
        """)

        # Unique teams from the dataset
        teams = updated_wc_final_data_df['Team1'].unique()

        # Simulate Matchups
        matchups = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                matchups.append({"Team_A": teams[i], "Team_B": teams[j]})

        simulated_data = pd.DataFrame(matchups)

        # Calculate Historical Averages
        team_stats = updated_wc_final_data_df.groupby('Team1').agg({
            'Team1 Strength Index': 'mean',
            'Batting Disparity': 'mean',
            'Bowling Disparity': 'mean',
            'Normalized Batting Difference': 'mean',
            'Normalized Bowling Difference': 'mean',
            'Rolling Win %': 'mean',
            'Rolling Margin (Runs)': 'mean',
            'Rolling Margin (Wickets)': 'mean',
            'Home Advantage': 'mean'
        }).reset_index()

        # Merge Features for Both Teams
        simulated_data = simulated_data.merge(team_stats, how='left', left_on='Team_A', right_on='Team1')
        simulated_data = simulated_data.merge(
            team_stats, how='left', left_on='Team_B', right_on='Team1', suffixes=('_A', '_B')
        )

        # Create Relative Difference Features
        simulated_data['Team1 Strength Index'] = simulated_data['Team1 Strength Index_A']
        simulated_data['Team2 Strength Index'] = simulated_data['Team1 Strength Index_B']
        simulated_data['Batting Disparity'] = simulated_data['Batting Disparity_A'] - simulated_data['Batting Disparity_B']
        simulated_data['Bowling Disparity'] = simulated_data['Bowling Disparity_A'] - simulated_data['Bowling Disparity_B']
        simulated_data['Normalized Batting Difference'] = simulated_data['Normalized Batting Difference_A'] - simulated_data['Normalized Batting Difference_B']
        simulated_data['Normalized Bowling Difference'] = simulated_data['Normalized Bowling Difference_A'] - simulated_data['Normalized Bowling Difference_B']
        simulated_data['Rolling Win %'] = simulated_data['Rolling Win %_A'] - simulated_data['Rolling Win %_B']
        simulated_data['Rolling Margin (Runs)'] = simulated_data['Rolling Margin (Runs)_A'] - simulated_data['Rolling Margin (Runs)_B']
        simulated_data['Rolling Margin (Wickets)'] = simulated_data['Rolling Margin (Wickets)_A'] - simulated_data['Rolling Margin (Wickets)_B']
        simulated_data['Home Advantage'] = simulated_data['Home Advantage_A'] - simulated_data['Home Advantage_B']

        # Define Features for Prediction
        features = [
            'Team1 Strength Index',
            'Team2 Strength Index',
            'Batting Disparity',
            'Bowling Disparity',
            'Normalized Batting Difference',
            'Normalized Bowling Difference',
            'Rolling Win %',
            'Rolling Margin (Runs)',
            'Rolling Margin (Wickets)',
            'Home Advantage'
        ]

        # Train a Random Forest model
        updated_wc_final_data_df['Target'] = updated_wc_final_data_df['Winner'].apply(
            lambda x: 0 if x == 'Team1' else 1
        )
        X = updated_wc_final_data_df[features]
        y = updated_wc_final_data_df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)

        # Predict Match Outcomes
        simulated_data['Predicted Team_A Win'] = rf_clf.predict(simulated_data[features])

        # Aggregate Results
        win_counts = pd.concat([
            simulated_data.loc[simulated_data['Predicted Team_A Win'] == 1, 'Team_A'],
            simulated_data.loc[simulated_data['Predicted Team_A Win'] == 0, 'Team_B']
        ]).value_counts()

        # Plot the Predicted Win Counts
        predictions_fig = px.bar(
            win_counts,
            x=win_counts.index,
            y=win_counts.values,
            title="Predicted Win Counts for Each Team in ICC Men's T20 World Cup 2026",
            labels={'x': 'Teams', 'y': 'Predicted Wins'},
            color=win_counts.values,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        predictions_fig.update_layout(
            xaxis_title="Teams",
            yaxis_title="Predicted Wins",
            xaxis_tickangle=-45
        )

        # Display Predictions
        st.plotly_chart(predictions_fig)
        st.write(f"### Predictions: The team most likely to win the ICC Men's T20 World Cup 2026 is **{win_counts.idxmax()}**!")






































































