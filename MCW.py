import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import base64
import joblib
from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import RandomOverSampler
from itertools import combinations





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
     1. **🎭 Fan Favorites**  
       Perfect for cricket lovers to discover:
                
        - **Team Battles**: See which teams ruled the field and who could improve.
        - **Ground Chronicles**: Discover stadiums where teams thrived.
        - **Player Highlights**: Check out star performances under pressure.
        - **Forecasting Champions**: Get a glimpse of who might win the next T20 World Cup.
        - **Quick Search**: Find stats on your favorite teams or players.

                
    2. **🧪 Data Insights**  
        Aimed at users who love working with data and want to see the science behind cricket analytics:
       - **About the Data**: Understand the dataset powering the app.
       - **IDA and EDA**: Journey of that data from raw to revelations
       - **Missingness**: Dive into the world of missing values in our cricket dataset to uncover patterns and make informed decisions.
       - **Feature Exploration**: Dive into trends and key metrics.
       - **Feature Engineering**: See how raw data transforms into actionable insights.
       - **Modeling Matches**: Learn how machine learning predicts outcomes.
       - **Champion Forecasts**: Use advanced models to predict the next T20 winner.



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
            "Data Journey: From Raw to Revelations",
            "Cracking the Mystery of Missingness",
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











#############################################################################################################################












#############################################################################################################################


# Matches and Wins by each Team

elif ui_section == "Team Battles":
    st.title("Team Battles")
    st.markdown("""
    Welcome to the **Matches and Wins by Each Team** section—a place where cricket history comes alive! 🏏
    """)

    # Ensure Match Year is calculated
    if 'Match Year' not in wc_final_data_df.columns:
        wc_final_data_df['Match Year'] = pd.to_datetime(wc_final_data_df['Match Date'], errors='coerce').dt.year

    # Function to calculate team stats
    def calculate_team_stats(df, team_col, winner_col):
        """Calculate participation and wins for a specific team column."""
        participation = df.groupby(['Match Year', team_col]).size().reset_index(name='Participation')
        wins = df[df[winner_col] == df[team_col]].groupby(['Match Year', team_col]).size().reset_index(name='Wins')
        return pd.merge(participation, wins, how='left', on=['Match Year', team_col]).fillna(0)

    # Calculate stats for Team1 and Team2
    team1_stats = calculate_team_stats(wc_final_data_df, 'Team1', 'Winner')
    team2_stats = calculate_team_stats(wc_final_data_df, 'Team2', 'Winner')

    # Add hover text for detailed information
    team1_stats['Hover Text'] = team1_stats.apply(
        lambda row: f"Team: {row['Team1']}<br>Year: {row['Match Year']}<br>Participation: {row['Participation']}<br>Wins: {row['Wins']}", 
        axis=1
    )
    team2_stats['Hover Text'] = team2_stats.apply(
        lambda row: f"Team: {row['Team2']}<br>Year: {row['Match Year']}<br>Participation: {row['Participation']}<br>Wins: {row['Wins']}", 
        axis=1
    )

    # Combined Bar and Line Plot
    st.subheader("Head-to-Head Wins Between Teams")
    st.markdown("""
    Dive into the intense rivalries between cricketing giants! This section showcases how each team has fared in direct matchups over the years. From Team 1's performances to Team 2's victories, discover the number of times each team emerged victorious in these head-to-head battles. See detailed breakdowns of participation and wins for each team, year by year, and uncover who truly holds the upper hand in this gripping saga of cricketing clashes.     """)

    # Function to create a combined bar and line plot
    def create_combined_plot(stats, team, colors):
        """Create a combined bar and line plot for participation and wins."""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats['Match Year'], 
            y=stats['Participation'], 
            name=f'{team} Participation', 
            marker_color=colors[0],
            hovertext=stats['Hover Text'], 
            hoverinfo="text"
        ))
        fig.add_trace(go.Scatter(
            x=stats['Match Year'], 
            y=stats['Wins'], 
            mode='lines+markers', 
            name=f'{team} Wins', 
            line=dict(color=colors[1]),
            hovertext=stats['Hover Text'], 
            hoverinfo="text"
        ))
        fig.update_layout(
            title=f'{team} WC Participation (Bar) and Wins (Line) Over the Years',
            xaxis_title='Match Year',
            yaxis=dict(title='Participation'),
            yaxis2=dict(title='Wins', overlaying='y', side='right'),
            barmode='group',
            hovermode='closest',
            width=1000,
            height=600
        )
        return fig

    # Generate plots for Team1 and Team2
    fig_team1 = create_combined_plot(team1_stats, 'Team 1', [px.colors.sequential.Viridis[3], px.colors.sequential.Viridis[6]])
    fig_team2 = create_combined_plot(team2_stats, 'Team 2', [px.colors.sequential.Viridis[1], px.colors.sequential.Viridis[9]])

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_team1, use_container_width=True)
    with col2:
        st.plotly_chart(fig_team2, use_container_width=True)

    st.subheader("Total Matches Played by Each Team")
    st.markdown(" Here, you’ll find out how many matches each team has played over the years. We’ve got a colorful **bar chart** that shows the total matches for every team. Want to know where your favorite  team is from? Check out the interactive **world map**, which highlights the countries where the teams are from.")
    # Calculate total matches for each team
    team1_counts = wc_final_data_df['Team1'].value_counts()
    team2_counts = wc_final_data_df['Team2'].value_counts()
    total_matches = pd.DataFrame({'Team': team1_counts.index, 'Matches': team1_counts.values})
    team2_matches = pd.DataFrame({'Team': team2_counts.index, 'Matches': team2_counts.values})
    total_matches = pd.concat([total_matches, team2_matches], ignore_index=True)
    total_matches = total_matches.groupby('Team', as_index=False).sum()

    # Bar Plot: Total Matches Played by Each Team
    # Sort the data by Matches in descending order
    total_matches_sorted = total_matches.sort_values(by='Matches', ascending=False)

    fig_total_matches = px.bar(
        total_matches,
        x='Team',
        y='Matches',
        color='Matches',  # Color by Matches
        labels={'Matches': 'Number of Matches', 'Team': 'Team'},
        text='Matches',
        template='plotly_white',  # Set template to plotly_white
        color_continuous_scale='Viridis'  # Use Viridis palette
    )
    fig_total_matches.update_layout(xaxis=dict(categoryorder='total descending'))
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
    Want to know which countries are the real champs? We’ve got another **world map** that paints the picture of 
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
    st.subheader("Match Margins")
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
        color='Titles',
        color_continuous_scale='Viridis'  # Use Viridis palette
    )
    fig_titles.update_layout(yaxis=dict(categoryorder='total ascending'))
    st.subheader("T-20 Worldcup Title Holders🏆")
    st.markdown("""
    Step into the Hall of Fame of **T20 Cricket Legends**! 
                
    Who has ruled the cricketing world in the shortest format? This section dives into the **T20 World Cup Title Holders**—the teams that have conquered the cricketing battlefield and lifted the coveted trophy. From thrilling super overs to unforgettable finals, these are the champions who have etched their names in the annals of cricketing history. 
    
    **What’s Inside?**
    - A **bar chart** showcasing the number of T20 titles each team has won.
    - Hover over the bars to relive the exact dates of their epic victories. 
    - Find out who’s leading the charge and who’s gearing up to join the elite club.

    Let’s explore the kings of T20 cricket and their journeys to glory! 
    """)

    st.plotly_chart(fig_titles)






#############################################################################################################################



#Total Matches played at each grounds


elif ui_section == "Ground Chronicles":
    st.title("Ground Chronicles")
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
                            color='Matches',  
                            color_continuous_scale='Viridis',
                            text='Matches')  
    fig_grounds_bar.update_layout(
        xaxis_title='Ground',
        yaxis_title='Number of Matches',)
    fig_grounds_bar.update_layout(
    xaxis={'categoryorder': 'total descending'}
)
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
    

    #Correlation Heatmap - one hot encoding
    st.subheader("Where Winners Rule: The Correlation Heatmap")
    st.markdown("""
        Ever wondered if certain cricket grounds have a magical charm that brings out the best in specific teams? 🤔🏏

        In this section, we dive into a **colorful heatmap** that uncovers the secret sauce of cricketing success! Here’s what it’s all about:

        **What’s Happening Here?**
        - We’ve analyzed how **winning teams** are connected to the **grounds** they played on.
        - Each box in the heatmap shows a **correlation score**—a number that tells us how strongly a winning team’s success is tied to playing on a specific ground. 📈
        - The brighter the color, the stronger the connection! 💡

         **Why Should You Care?**
        - Discover the lucky grounds that seem to favor your favorite teams.
        - Find out if there’s a hidden home-ground advantage lurking in the data. 🏠✨
        - It’s like uncovering the lucky charms of the cricketing world—one ground at a time!

        Take a closer look, hover over the heatmap, and let the colors tell you the story of cricketing success! 🚀
        """)

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

# Player Glory Section
elif ui_section == "Player Glory":
    st.title("Player Glory")
    st.subheader("Unveiling Player Glory")
    st.markdown("""
    
    Explore the incredible contributions of players to the T20 World Cup in this engaging section. From participation and wins across years to standout performances, we've got it all!
    """)

    # Player Participation Trends
    st.subheader("Player Participation Trends")
    st.markdown("""
    Watch how teams have been represented by their total number of players over the years. This section highlights how many players from each country participated in different tournaments, giving a sense of their consistent presence in the competition.
    """)

    # Ensure all years and teams are represented
    all_years = pd.DataFrame({'Year': range(2007, 2025)})
    all_teams = players_df['Team'].unique()
    grid = pd.MultiIndex.from_product([all_years['Year'], all_teams], names=['Year', 'Team']).to_frame(index=False)

    # Merge with player data
    player_data = pd.merge(grid, players_df.groupby(['Year', 'Team']).size().reset_index(name='Players'), how='left')
    player_data['Players'] = player_data['Players'].fillna(0)

    # Summarize data by team
    total_players_by_team = player_data.groupby('Team')['Players'].sum().reset_index()
    total_players_by_team = total_players_by_team.sort_values(by='Players', ascending=False)

    # Bar chart for total player participation by country
    fig_players_bar = px.bar(
        total_players_by_team, 
        x='Players', 
        y='Team', 
        orientation='h',  # Horizontal bar chart
        title='Total Player Participation by Country in T20 World Cups',
        labels={'Players': 'Number of Players', 'Team': 'Country'},
        text='Players',  # Display the number of players on the bars
        template='plotly_white',
        color='Players',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Update layout for better visualization
    fig_players_bar.update_traces(textposition='outside')
    fig_players_bar.update_layout(
        xaxis_title='Number of Players',
        yaxis_title='Country',
        width=800,
        height=600,
        yaxis=dict(categoryorder='total ascending')  # Sort by total players
    )

    st.plotly_chart(fig_players_bar, use_container_width=True)


    # Legends of Longevity
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
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_longest_participation .update_layout(
    xaxis={'categoryorder': 'total descending'}
    )

    st.plotly_chart(fig_longest_participation, use_container_width=True)

    # Captains Who Led the Way
    st.subheader("Captains Who Led the Way")
    st.markdown("""
    Leadership matters! Discover the captains who led their teams for the longest time in World Cups. These individuals not only inspired their teammates but also etched their names in cricketing history.
    """)

    if 'Year' in players_df.columns and 'Year' in captains_df.columns:
        merged_captains = pd.merge(players_df, captains_df, on=['Player Name', 'Team', 'Year'], how='inner')
        captain_durations = merged_captains.groupby(['Player Name', 'Team'])['Year'].nunique().reset_index(name='Captaincy Duration')
        longest_captaincy = captain_durations.loc[captain_durations.groupby('Team')['Captaincy Duration'].idxmax()]

        fig_longest_captains = px.bar(
            longest_captaincy, 
            x='Team', 
            y='Captaincy Duration', 
            color='Player Name', 
            title='Captains with the Longest Duration for Each Team',
            labels={'Captaincy Duration': 'Years as Captain', 'Team': 'Team'},
            text='Player Name',
            template='plotly_white',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_longest_captains .update_layout(
        xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_longest_captains, use_container_width=True)
    else:
        st.error("Required columns missing in players or captains dataset.")

    # Top Match Winners
    st.subheader("Top Match Winners")
    st.markdown("""
    Find out which players secured the most wins for their teams. This section highlights the standout performers who consistently delivered victories, earning them a spot in cricketing glory.
    """)

    merged_data = pd.merge(all_matches_data_df, players_df, how='inner', left_on='Winner', right_on='Team')
    player_wins = merged_data.groupby(['Player Name', 'Team']).size().reset_index(name='Wins')
    top_players_by_wins = player_wins.loc[player_wins.groupby('Team')['Wins'].idxmax()]

    fig_funnel = px.funnel(
        top_players_by_wins, 
        x='Player Name', 
        y='Wins', 
        color='Team', 
        title='Wins by Top Player for Each Team',
        labels={'Wins': 'Number of Wins', 'Player Name': 'Player Name'},
        template='plotly_white',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_funnel .update_layout(
    xaxis={'categoryorder': 'total descending'}
    )
    st.plotly_chart(fig_funnel, use_container_width=True)




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
        st.title("Forecasting the Next Champions")
        st.subheader(" Who Will Reign Supreme?")
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
    st.title("Search Magic")
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
    st.subheader(" Search for a Player")
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

    ## **Here’s What Awaits You:**  """)

    st.markdown(" **📊 About the Data**")
    st.markdown("""
    Peek behind the curtains and understand the foundation of this exploration—our dataset! Learn about the features, variables, and data sources that drive the analytics.  
    """)

    st.markdown("**📈 Cricket Stats**")
    st.markdown("""
    Dive into the **Distribution of Features** and uncover fascinating trends and stories hidden within the data. From player performances to team dynamics, visualize how the numbers come alive to tell the story of the World Cup.  
    """)

    st.markdown(" **🧹 Data Journey: From Raw to Revelations**")
    st.markdown("""
    Our combined **IDA and EDA** section ensured the dataset was cleaned, structured, and explored. Key highlights included extracting match numbers, splitting margins into `Margin (Runs)` and `Margin (Wickets)`, and visualizing trends to uncover the stories behind the numbers.
    """)

    st.markdown(" **🔍 Cracking the Mystery of Missingness**")
    st.markdown("""
    Exploring missing data in the dataset revealed structured patterns of missingness, like the inverse relationship between `Margin (Runs)` and `Margin (Wickets)`. Using heatmaps and a missingness correlation matrix, we identified these patterns and handled the missing values effectively to maintain data integrity.
    """)

    st.markdown("**🛠️ Feature Factory**")
    st.markdown("""
    The lab where the magic happens! This section breaks down the building blocks of cricket analytics, showing you how raw data transforms into meaningful insights. Discover engineered features that enhance predictions and uncover hidden relationships.
    """)

    st.markdown(" **🤖 Modeling the Game: Unveiling Predictions**")
    st.markdown("""
    Take a step into the machine learning arena. See how advanced models like Logistic Regression, Random Forest, and XGBoost are trained to predict match outcomes. Dive into the performance metrics to understand what drives accurate predictions.
    """)

    st.markdown("**🔮 Forecasting the Next Champions**")
    st.markdown("""
    Enter the **Predictor’s Playground** and try your hand at being the ultimate cricket soothsayer. With the power of advanced analytics, see which team has the best odds of taking home the next World Cup trophy.
    """)

    st.markdown("## **Who is This For?**  ")
    st.markdown("""If you love playing with numbers, solving mysteries, or just want to see the “behind-the-scenes” magic of cricket analytics, this is your playground. It’s not just stats; it’s the art of making every number count!

Get ready to blend cricket passion with data brilliance, and let’s create some magic together! 🏆✨  
    """)



        # Footer or call-to-action
    st.markdown("---")
    st.markdown("### 🏏 Let the cricket journey begin! Navigate using the sidebar to explore more insights.")




############################################################################################################################


elif ds_section == "About the Data":
    st.title("About the Data")
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

# IDA and EDA Section in Streamlit
elif ds_section == "Data Journey: From Raw to Revelations":
    st.title( "Data Journey: From Raw to Revelations")
    st.subheader("Welcome to the Data Adventure Zone! 🏏📊")

    st.markdown("""
    Our journey began with **Initial Data Analysis (IDA)**—the cleaning crew of our cricketing dataset. 
    We took raw, messy data and turned it into something sparkling and ready for action. Here's a highlight reel of our efforts:  
    """)

    st.markdown("""
    - **T-20 Match Detective Work**: We channeled our inner Sherlock Holmes to decode match identifiers, extracting cryptic T-20 match numbers from strings (yes, even those tricky hashtags!) and giving them their rightful place in our dataset.  
    - **Time Machine Activated**: We broke down match dates into year, month, and day—unlocking the power to analyze cricketing trends like seasoned analysts.  
    - **Margin Makeover**: Margins of victory were in mixed formats (runs or wickets), so we split them into two separate columns—`Margin (Runs)` and `Margin (Wickets)`—for cleaner and more precise analysis.  
    - **Column Magic**: Unnecessary baggage? Gone! Redundant columns like the original `Match Date` were retired after we extracted all their valuable insights.  
    """)

    st.markdown("""
    With our data spick-and-span, it was time for **Exploratory Data Analysis (EDA)**—the playground of insights! Here’s what we did:  
    """)

    st.markdown("""
    - **Trends Galore**: Through engaging visualizations, we captured team performances across the years, from participation counts to win percentages.  
    - **Battlefields Unveiled**: Interactive maps brought history to life, showing us where teams fought it out—team by team, ground by ground.  
    - **Player Spotlights**: Player statistics became our focus, as we uncovered the stars who outshone the rest and the legends who delivered the most wins for their teams.  
    """)

    st.markdown("""
    All these visualizations and more can be found in the **Fan Favorites** section of this app—crafted just for you!  

    Together, IDA and EDA gave us the foundation and the spark to bring this app to life. 
    Now, it’s your turn to explore and uncover the stories hidden in the numbers. Enjoy the ride! 🚀
    """)





############################################################################################################################

elif ds_section == "Cracking the Mystery of Missingness":
    st.title("🔍 Cracking the Mystery of Missingness")

    st.markdown("""
    Data is rarely perfect, but that's what makes it exciting! Here, we delve into the world of missing values in our cricket dataset to uncover patterns and make informed decisions.  

    ### **No Missingness at the Start!**  
    Initially, we were delighted to find no missing values in our dataset. However, as we dived deeper, analyzing other data files, we encountered some gaps—small but significant. Below is what we explored:
    """)
    missing_values_all_matches = all_matches_data_df.isnull().sum()
    missing_values_wc_final_dataset = wc_final_data_df.isnull().sum()
    missing_values_players = players_df.isnull().sum()
    missing_values_captains = captains_df.isnull().sum()

    missing_values = wc_final_data_df.isnull().sum()

    # Display the missing values in each column
    missing_values

    st.markdown("""
    ### **What We Found:**  
    Our primary dataset (`wc_final_data_df`) showed missing values in two specific columns:  
    - **Margin (Runs)**: 166 missing values  
    - **Margin (Wickets)**: 169 missing values  

    Why the gaps? Matches are either decided by runs or wickets—so when one margin is present, the other is naturally missing. This is structured missingness, tied to how the game was decided. Here's a heatmap highlighting the missing values in the dataset:
    """)
    # Define functions 
    def extract_numeric_value(value):
        match = re.search(r'\d+', str(value))
        return int(match.group()) if match else None
    def extract_t20_int_match(value):
        match = re.search(r'# (\d+)', str(value))  # Extract the number after the hash symbol
        return int(match.group(1)) if match else None

    # Margins Column into Margb (runs) and Margin (Wickets)
    def extract_runs_correct(margin):
        if isinstance(margin, str) and 'runs' in margin:
            return float(margin.split()[0])  # Extract the number for runs
        return None

    def extract_wickets_correct(margin):
        if isinstance(margin, str) and 'wickets' in margin:
            return float(margin.split()[0])  # Extract the number for wickets
        return None

    # 
    wc_final_data_df['Margin (Runs)'] = wc_final_data_df['Margin'].apply(extract_runs_correct)
    wc_final_data_df['Margin (Wickets)'] = wc_final_data_df['Margin'].apply(extract_wickets_correct)

    # Display the updated dataframe with the new columns
    wc_final_data_df[['Margin', 'Margin (Runs)', 'Margin (Wickets)']].head()

    # Save the updated dataframe
    wc_final_data_df.to_csv('updated_wc_final_data_df.csv', index=False)

    # Plot missingness heatmap for wc_final_data_df
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(wc_final_data_df.isnull(), cmap="YlGnBu", cbar=True, yticklabels=False, ax=ax)
    ax.set_title('Missingness Heatmap for the Entire Dataset')
    st.pyplot(fig)

    st.markdown("""
    ### **Investigating Patterns:**  
    When we analyzed the relationship between these columns, we found them to be **highly inversely correlated**. This means:  
    - If `Margin (Runs)` is present, `Margin (Wickets)` is missing, and vice versa.  

    This indicates a **Missing at Random (MAR)** pattern, where the missingness is dependent on the match's outcome type. To handle this, we replaced the NaN values in these columns with `0`, as they represent cases where the respective metric was irrelevant.  

    By addressing missingness in a structured way, we ensured that the integrity of our dataset remains intact, paving the way for accurate analysis and meaningful insights! 🏏
    """)






############################################################################################################################


elif ds_section == "Cricket Stats":
    st.title("Cricket Stats: Hidden Stories Behind the Numbers 🏏")

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
    fig_margin_runs.update_layout(
    xaxis={'categoryorder': 'total descending'}
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
    
    fig_margin_wickets.update_layout(
    xaxis={'categoryorder': 'total descending'}
    )

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
        wc_final_data_df,
        x="Team1 Avg Batting Ranking",
        nbins=20,
        title="Distribution of Team 1 Avg Batting Ranking",
        labels={"x": "Batting Ranking (Team 1)", "y": "Frequency"},
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    fig_batting_team1.update_traces(marker_line_color="black", marker_line_width=1.5)
    fig_batting_team1.update_layout(
        xaxis_title="Batting Ranking (Team 1)",
        yaxis_title="Frequency",
        width=600,
        height=400
)


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


elif ds_section == "Feature Factory":
    st.title("Feature Factory: Cricket Analytics Unveiled")
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

    # Show `updated_wc_final_data_df` with description
    st.write("### The Dataset: `updated_wc_final_data_df`")
    st.write(
        """
        This is the core dataset used for cricket analytics. It includes historical match data, team statistics, 
        and engineered features derived from the raw data. Here are some of its key features:
        - **Winner**: The winning team of the match.
        - **Team1 and Team2**: Competing teams.
        - **Ground**: Venue of the match.
        - **Strength Indices**: Metrics combining batting and bowling performances.
        - **Normalized Differences**: Adjusted values highlighting disparities between teams.
        """
    )
    st.dataframe(updated_wc_final_data_df)

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

# Modeling
import requests
import joblib
from io import BytesIO


# URLs for pre-trained models
LOG_REG_MODEL_URL = "https://github.com/Shamsvi/Streamlit_Mens_T-20_Cricket_WorldCup_2007-2024/raw/main/models/logistic_regression_model.pkl"
RF_MODEL_URL = "https://github.com/Shamsvi/Streamlit_Mens_T-20_Cricket_WorldCup_2007-2024/raw/main/models/random_forest_model.pkl"
XGB_MODEL_URL = "https://github.com/Shamsvi/Streamlit_Mens_T-20_Cricket_WorldCup_2007-2024/raw/main/models/xgboost_model.pkl"

# Function to dynamically load models
@st.cache_resource
def load_models():
    try:
        log_reg = joblib.load(BytesIO(requests.get(LOG_REG_MODEL_URL).content))
        rf_clf = joblib.load(BytesIO(requests.get(RF_MODEL_URL).content))
        xgb_clf = joblib.load(BytesIO(requests.get(XGB_MODEL_URL).content))
        return log_reg, rf_clf, xgb_clf
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Preprocessing Function
@st.cache_data
def preprocess_data(df, required_features):
    missing_features = [feature for feature in required_features if feature not in df.columns]
    for feature in missing_features:
        df[feature] = 0  # Assign default value to missing features

    if df.empty or df.isnull().all().any():
        return None, None, None, None, None, ["Dataset is empty or contains only missing values."]

    X = df[required_features]
    y = df['Winner']

    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if not rare_classes.empty:
        df = df[~df['Winner'].isin(rare_classes)]
        X = df[required_features]
        y = df['Winner']

    if y.nunique() <= 1:
        return None, None, None, None, None, ["Target variable 'Winner' does not have enough variability."]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    try:
        ros = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = ros.fit_resample(X, y_encoded)
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
        )
    except ValueError as e:
        return None, None, None, None, None, [f"Error during balancing or splitting: {e}"]

    return X_train, X_test, y_train, y_test, label_encoder, None

# Confusion Matrix Visualization
def interactive_confusion_matrix(y_test, predictions, model_name, label_encoder):
    cm = confusion_matrix(y_test, predictions)
    labels = label_encoder.classes_

    labeled_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        labeled_cm,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        ax=ax
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Actual Class", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Modeling Section
if ds_section == "Modeling the Game: Unveiling Predictions":
    if 'updated_wc_final_data_df' not in locals():
        st.error("Dataset `updated_wc_final_data_df` is not loaded.")
    else:
        required_features = [
            'Team1 Strength Index', 'Team2 Strength Index',
            'Batting Disparity', 'Bowling Disparity',
            'Rolling Margin (Runs)', 'Rolling Margin (Wickets)',
            'Home Advantage'
        ]

        st.title("Modeling the Game: Unveiling Predictions")
        st.write("""
        **Step into the Analytics Dugout!**
                 
        In this section, we use cutting-edge machine learning models to predict the outcomes of cricket matches. It's like having your own expert cricket analyst, but powered by algorithms! We compare the performances of Logistic Regression, Random Forest, and XGBoost to see which model hits the boundary and predicts match outcomes with the most accuracy.

        But that's not all—this section also introduces Confusion Matrices, a powerful tool for evaluating model performance. A confusion matrix shows how well a model distinguishes between different classes by displaying the actual versus predicted outcomes for each class, with the diagonal values indicating correct predictions and the off-diagonal values representing misclassifications.

        Since we're handling a multiclass classification problem, the confusion matrix displays results for all possible classes (e.g., predictions for Afghanistan, India, Australia, etc.) rather than a simplified 2x2 binary classification matrix. In this scenario, the diagonal values represent correct predictions for each class, while the off-diagonal values represent misclassifications.🚀
        """)

        # Preprocess Data
        X_train, X_test, y_train, y_test, label_encoder, issues = preprocess_data(updated_wc_final_data_df, required_features)
        if issues:
            for issue in issues:
                st.error(issue)
            st.stop()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        log_reg, rf_clf, xgb_clf = load_models()
        if not all([log_reg, rf_clf, xgb_clf]):
            st.error("Models failed to load properly. Please check the URLs.")
            st.stop()

        models = {
            "Logistic Regression": log_reg,
            "Random Forest": rf_clf,
            "XGBoost": xgb_clf
        }

        results = {}
        for model_name, model in models.items():
            y_pred = model.predict(X_test_scaled if model_name == "Logistic Regression" else X_test)
            metrics = {
                "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
                "Precision (%)": round(precision_score(y_test, y_pred, average="weighted") * 100, 2),
                "Recall (%)": round(recall_score(y_test, y_pred, average="weighted") * 100, 2),
                "F1-Score (%)": round(f1_score(y_test, y_pred, average="weighted") * 100, 2)
            }
            results[model_name] = metrics
            st.subheader(f"{model_name} Evaluation Table")
            st.write(pd.DataFrame(metrics, index=["Value"]).T)
            interactive_confusion_matrix(y_test, y_pred, model_name, label_encoder)

            if model_name == "Logistic Regression":
                st.write("""
                **About Logistic Regression:**
                Logistic Regression is a linear model that predicts probabilities of outcomes using a logistic function. 
                It is straightforward, interpretable, and performs well on linearly separable data.

                **Deductions:**
                - Ideal for datasets with a linear relationship between features and target variable.
                - May struggle with complex, non-linear interactions or when the dataset has a large number of classes.
                """)
            elif model_name == "Random Forest":
                st.write("""
                **About Random Forest:**
                Random Forest is an ensemble learning method that creates multiple decision trees and aggregates their results.
                It excels in handling non-linear relationships and complex datasets with noisy features.

                **Deductions:**
                - Provides robust performance by reducing overfitting through ensembling.
                - Particularly useful when feature interactions are important in determining outcomes.
                - Provides feature importance scores for model interpretability.
                """)
            elif model_name == "XGBoost":
                st.write("""
                **About XGBoost:**
                XGBoost is a gradient boosting algorithm known for its speed and performance. 
                It iteratively builds trees to correct errors made in previous iterations.

                **Deductions:**
                - Highly effective for large, complex datasets with many features.
                - Its regularization parameters help reduce overfitting and improve generalization.
                - The iterative learning approach makes it capable of capturing intricate patterns in the data.
                """)

        st.subheader("Model Performance Comparison")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        best_model_name = results_df['F1-Score (%)'].idxmax()
        st.write(f"### Recommendation: The best model for this dataset is **{best_model_name}**, achieving the highest F1-Score.")


    





############################################################################################################################

elif ds_section == "Forecasting the Next Champions":

    # Check for dataset availability
    if 'updated_wc_final_data_df' not in locals():
        st.error("Dataset `updated_wc_final_data_df` is not loaded. Please load the dataset.")
    else:
        # Apply Feature Engineering
        if 'Home Advantage' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Home Advantage'] = updated_wc_final_data_df.apply(
                lambda row: 1 if row['Team1'] in row['Ground'] or row['Team2'] in row['Ground'] else 0, axis=1
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

        if 'Batting Disparity' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Batting Disparity'] = (
                updated_wc_final_data_df['Team1 Avg Batting Ranking'] - updated_wc_final_data_df['Team2 Avg Batting Ranking']
            )

        if 'Bowling Disparity' not in updated_wc_final_data_df.columns:
            updated_wc_final_data_df['Bowling Disparity'] = (
                updated_wc_final_data_df['Team1 Avg Bowling Ranking'] - updated_wc_final_data_df['Team2 Avg Bowling Ranking']
            )

        # Forecasting Section
        st.title("Forecasting the Next Champions")
        st.subheader("Who Will Reign Supreme?")
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
        - Using a trained model, outcomes of each matchup are predicted.
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




























