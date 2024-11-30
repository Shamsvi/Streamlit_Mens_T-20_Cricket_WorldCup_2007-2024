import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go  
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import base64

# Load the dataset
matches_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/all_t20_world_cup_matches_results.csv'
players_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/all_t20_world_cup_players_list.csv'
final_dataset_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/wc_final_dataset.csv'
captains_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/all_captains.csv'
cricket_legends_url = 'https://raw.githubusercontent.com/Shamsvi/CMSE-830/main/MidtermPorject/cricket_legends.csv'

# Load the datasets from the URLs
all_matches_data_df = pd.read_csv(matches_url)
players_df = pd.read_csv(players_url)
wc_final_data_df = pd.read_csv(final_dataset_url)
captains_df = pd.read_csv(captains_url)
cricket_legends_df = pd.read_csv(cricket_legends_url)
# App Title
st.title("🏏 Welcome to the Ultimate Men's T20 World Cup Analysis App! 🏆")





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


# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.selectbox(
    "Go to Section",
    (
        "Welcome",
        "Dataset Overview", #all sections
        "Distribution of Key Numeric Features",
        "Distribution of Ranking Differences",
        "Matches",
        "Wins - Total No. of Wins by Each Team",
        "Grounds",
        "Team1 vs Team2 Participations and Wins",
        "Player Participation",
        "Search For Your Favourite Teams and Players"
    )
)







#############################################################################################################################

if section == "Welcome":
     st.subheader('Cricket Fever: Data Edition') 
     st.image("icc_cricket .png", use_container_width=True)
     st.markdown("""
                Are you ready to dive into the thrilling world of cricket? Whether you’re a die-hard fan, a stats geek, or just someone who loves the spirit of the game, this app is your one-stop destination to explore and analyze everything about the **Men's T20 World Cup**!

                ✨ From nail-biting finishes to record-breaking performances, this app unpacks the data behind the drama. Explore:

                - 🔥 **Team Battles**: Who dominated the field and who needs to up their game?
                - 🌍 **Ground Chronicles**: Which stadiums turned into fortresses for teams?
                - 🌟 **Player Glory**: Discover stars who shone brightest under pressure.
                - 🕵️‍♂️ **Search Magic**: Zero in on your favorite teams or players in an instant!

                🎉 **Why this app?**  
                Because cricket isn’t just a sport—it’s a passion, a science, and a celebration. And with this app, you can experience it all in an interactive, fun, and data-driven way.
                """)
     

    # Footer or call-to-action
     st.markdown("---")
     st.markdown("### 🏏 Let the cricket journey begin! Navigate using the sidebar to explore more insights.")




#############################################################################################################################




#  Display the dataset
elif section == "Dataset Overview":
    st.subheader('Dataset Overview')    
    st.write("The Men's Cricket World Cup is one of the most prestigious tournaments in the cricketing calendar, showcasing the talents of the world's best teams and players. This project aims to analyze and visualize key data from the tournament, focusing on match outcomes, team performances, and individual player statistics. By leveraging advanced data analysis techniques, we will explore trends in match margins, batting and bowling averages, and historical rivalries. Through this comprehensive analysis, we seek to provide valuable insights into the dynamics of the tournament, enhancing our understanding of competitiveness and performance in international cricket.")
    wc_final_data_df = pd.read_csv('updated_wc_final_data_df.csv')
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(wc_final_data_df, key = 'dataframe')
    with col2:
        st.dataframe(players_df, key = 'dataframe')









#############################################################################################################################








#Distribution of Key Numeric Features

elif section == "Distribution of Key Numeric Features":
    st.subheader("Distribution of Key Numeric Features")
    st.write("These plots provide insights into match competitiveness and team performance. The first histogram shows the distribution of match margins, where lower margins indicate closely contested games and higher margins point to dominant victories. The second histogram displays Team 1's average batting rankings, helping to assess their overall batting strength. Together, these visualizations highlight the balance between teams and offer a snapshot of performance trends.")

    # Colorblind-friendly colors
    colorblind_friendly_colors = ['#0072B2', '#D55E00']

    # Bar Plot 1: Distribution of Match Margin
    fig_margin = px.histogram(
        wc_final_data_df, 
        x='Margin', 
        nbins=20, 
        title='Distribution of Match Margin',
        labels={'x': 'Margin', 'y': 'Frequency'},
        template='plotly_white'
    )
    fig_margin.update_traces(
        marker_color=colorblind_friendly_colors[0],  # Use colorblind-friendly color
        marker_line_color='black', 
        marker_line_width=1.5
    )
    fig_margin.update_layout(
        xaxis_title='Margin',
        yaxis_title='Frequency',
        width=600,  
        height=400
    )

    # Bar Plot 2: Distribution of Team1 Avg Batting Ranking
    fig_batting_ranking = px.histogram(
        wc_final_data_df, 
        x='Team1 Avg Batting Ranking', 
        nbins=20, 
        title='Distribution of Team1 Avg Batting Ranking',
        labels={'x': 'Batting Ranking', 'y': 'Frequency'},
        template='plotly_white'
    )
    fig_batting_ranking.update_traces(
        marker_color=colorblind_friendly_colors[1],  # Use colorblind-friendly color
        marker_line_color='black', 
        marker_line_width=1.5
    )
    fig_batting_ranking.update_layout(
        xaxis_title='Batting Ranking',
        yaxis_title='Frequency',
        width=600,  
        height=400
    )

    # Display plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_margin, use_container_width=True)
    with col2:
        st.plotly_chart(fig_batting_ranking, use_container_width=True)






#############################################################################################################################




#Distribution of Ranking Differences

elif section == "Distribution of Ranking Differences":
    st.subheader("Distribution of Ranking Differences")
    st.write("The 2 histograms show the distribution of batting and bowling ranking differences between Team 1 and Team 2.The x-axis represents the ranking difference, while the y-axis shows the frequency of occurrences. A higher frequency at lower differences suggests that most matches have been between teams with similar rankings, while a higher frequency at larger differences indicates matches with greater disparities in rankings. ")
    # Plot 1: Distribution of Batting Ranking Difference with black edges
    fig_batting_ranking_diff = px.histogram(
        wc_final_data_df, 
        x='Batting Ranking Difference', 
        nbins=20, 
        title='Distribution of Batting Ranking Difference',
        labels={'x': 'Batting Ranking Difference', 'y': 'Frequency'},
        template='plotly_white'
    )
    fig_batting_ranking_diff.update_traces(marker_line_color='black', marker_line_width=1.5)
    fig_batting_ranking_diff.update_layout(
        xaxis_title='Batting Ranking Difference',
        yaxis_title='Frequency',
        width=600,  
        height=400
    )
    fig_bowling_ranking_diff = px.histogram(
        wc_final_data_df, 
        x='Bowling Ranking Difference', 
        nbins=20, 
        title='Distribution of Bowling Ranking Difference',
        labels={'x': 'Bowling Ranking Difference', 'y': 'Frequency'},
        template='plotly_white'
    )
    fig_bowling_ranking_diff.update_traces(marker_line_color='black', marker_line_width=1.5)
    fig_bowling_ranking_diff.update_layout(
        xaxis_title='Bowling Ranking Difference',
        yaxis_title='Frequency',
        width=600,  
        height=400
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_batting_ranking_diff, use_container_width=True)
    with col2:
        st.plotly_chart(fig_bowling_ranking_diff, use_container_width=True)
    #Line plot
    st.write("The line plot tracks the changes in batting and bowling ranking differences over time. The x-axis represents the match year, and the y-axis shows the ranking difference.This plot helps visualize how the competitiveness between teams has evolved over time, highlighting trends in batting and bowling strength disparities. ")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wc_final_data_df['Match Year'],
        y=wc_final_data_df['Batting Ranking Difference'],
        mode='lines',
        name='Batting Ranking Difference',
        line=dict(color='coral'),
        hovertext=wc_final_data_df.apply(lambda row: f"Team 1: {row['Team1']}<br>Team 2: {row['Team2']}", axis=1),  # Adding team names in hover text
        hoverinfo='text+y'
    ))
    fig.add_trace(go.Scatter(
        x=wc_final_data_df['Match Year'],
        y=wc_final_data_df['Bowling Ranking Difference'],
        mode='lines',
        name='Bowling Ranking Difference',
        line=dict(color='purple', dash='dash'),
        hovertext=wc_final_data_df.apply(lambda row: f"Team 1: {row['Team1']}<br>Team 2: {row['Team2']}", axis=1),  # Adding team names in hover text
        hoverinfo='text+y'
    ))
    fig.update_layout(
        title='Batting and Bowling Ranking Differences Between Team 1 and Team 2 Over Time',
        xaxis_title='Match Year',
        yaxis_title='Ranking Difference',
        legend_title='Ranking Type',
        hovermode='x unified'
    )
    st.plotly_chart(fig)






#############################################################################################################################







# Total Matches Played by Each Team

elif section == "Matches":
    st.subheader("Matches")
    st.write("The Matches section displays the total number of matches played by each team using a bar plot and a geospatial map. The number of matches for each team is calculated by combining their appearances as both Team1 and Team2. The bar plot shows the total number of matches for each team, and a choropleth map visualizes the geographical distribution of matches by mapping teams to their respective countries. Both visualizations are presented side by side to provide insights into the total matches and their geographic spread.")

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


    





#############################################################################################################################





#Total Wins by Each Team

elif section == "Wins - Total No. of Wins by Each Team":
    st.subheader("Wins - Total No. of Wins by Each Team")
    st.write("In this section, we analyze team performance in the T20 World Cup through various visualizations. First, a bar chart highlights the total wins by each team, providing a clear comparison of the top performers in the tournament. Next, a geospatial plot illustrates the distribution of these wins across different countries, offering a geographic perspective on team success. Following this, a scatter plot examines the match margins, showcasing by how many runs each team won and identifying the losing teams in each case. Finally, a horizontal bar chart displays the total titles won by each team, with hover functionality that reveals the exact final match dates when teams secured their victories. Together, these visualizations offer a comprehensive view of team dominance, performance patterns, and geographic success in the T20 World Cup.")
    

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
    win_counts.columns = ['Team', 'Wins']  # Rename columns for clarity

    # Bar Plot for Wins
    fig_bar_wins = px.bar(
        win_counts, 
        x='Team', 
        y='Wins', 
        labels={'Wins': 'Number of Wins', 'Team': 'Team'},
        text='Wins',  
        title='Number of Wins by Country (Teams)',
        color='Wins',  
        color_continuous_scale='Viridis'
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

    # Choropleth map for total wins
    fig_geo_wins = px.choropleth(
        country_wins,
        locations='Country',
        locationmode='country names',
        color='Wins',
        hover_name='Country',
        hover_data=['Wins'],
        title='Total Wins by Country (Teams)',
        color_continuous_scale='Viridis'
        
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
    wc_final_data_df['Losing Team'] = wc_final_data_df.apply(determine_losing_team, axis=1)

    # Scatter Plot for margins
    wc_final_data_df['Margin (Runs)'] = wc_final_data_df['Margin'].apply(extract_runs_correct)
    wc_final_data_df['Margin (Wickets)'] = wc_final_data_df['Margin'].apply(extract_wickets_correct)

    # Select relevant columns, including 'Winner', 'Losing Team', and 'Margin (Runs)' or 'Margin (Wickets)'
    plot_data = wc_final_data_df[['Winner', 'Losing Team', 'Margin (Runs)', 'Margin (Wickets)']].dropna(subset=['Winner', 'Losing Team'])

    # Create 'Margin Type' column to differentiate between runs and wickets
    plot_data['Margin Type'] = plot_data.apply(lambda row: 'Runs' if pd.notnull(row['Margin (Runs)']) else 'Wickets', axis=1)

    # Create a 'Margin Numeric' column that combines 'Margin (Runs)' and 'Margin (Wickets)'
    plot_data['Margin Numeric'] = plot_data.apply(lambda row: row['Margin (Runs)'] if pd.notnull(row['Margin (Runs)']) else row['Margin (Wickets)'], axis=1)

    # Update the scatter plot to reflect the margin by runs or wickets
    fig_win_margin = px.scatter(
        plot_data,
        x='Winner',
        y='Margin Numeric',
        title='Match Margin by Winner (Scatter Plot)',
        labels={'Margin Numeric': 'Match Margin', 'Winner': 'Winning Team'},
        color='Margin Type',  # Color by whether the margin was in Runs or Wickets
        hover_data={
            'Winner': True,
            'Margin Numeric': True,
            'Losing Team': True,
            'Margin Type': True  # Show the type of margin in hover info
        }
    )

    # Display the scatter plot and data frame side by side
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_win_margin, use_container_width=True)
    with col4:
    # Exclude 'Margin Numeric' column
        st.dataframe(plot_data.drop(columns=['Margin Numeric']), use_container_width=True)

    # Add Match Date column by parsing year, month, and day
    wc_final_data_df['Match Date'] = pd.to_datetime(
        wc_final_data_df[['Match Year', 'Match Month', 'Match Day']].astype(str).agg('-'.join, axis=1), errors='coerce')

    # List of specific final match dates
    final_dates = [
        '2007-09-24',  # September 24, 2007
        '2009-06-21',  # June 21, 2009
        '2010-05-16',  # May 16, 2010
        '2012-10-07',  # October 7, 2012
        '2014-04-06',  # April 6, 2014
        '2016-04-03',  # April 3, 2016
        '2021-11-14',  # November 14, 2021
        '2022-11-13',  # November 13, 2022
        '2024-06-29'   # June 29, 2024
    ]

    final_dates = pd.to_datetime(final_dates)

    # Filter dataset for final match dates only
    final_matches_df = wc_final_data_df[wc_final_data_df['Match Date'].isin(final_dates)]

    # Aggregate number of titles for each team
    winner_titles = final_matches_df.groupby('Winner')['Match Date'].apply(list).reset_index(name='Final Dates')
    winner_titles['Final Dates'] = winner_titles['Final Dates'].apply(lambda dates: [date.strftime('%Y-%m-%d') for date in dates])
    winner_titles['Titles'] = winner_titles['Final Dates'].apply(len)

    # Sort by number of titles
    winner_titles = winner_titles.sort_values(by='Titles', ascending=False)

    # Horizontal bar chart for titles
    fig_titles = px.bar(
        winner_titles, 
        y='Winner', 
        x='Titles', 
        orientation='h',
        text='Titles',
        hover_data={'Final Dates': True},
        labels={'Titles': 'Number of Titles', 'Winner': 'Team'},
        title='T20 World Cup Title Holders'
    )

    # Display the title holders bar chart
    st.plotly_chart(fig_titles)





#############################################################################################################################






#Total Matches played at each grounds


elif section == "Grounds":
    st.subheader("Grounds")
    st.write("In this section, we explore the impact of different cricket grounds on match outcomes. We begin by showcasing the total matches played at each ground through a bar chart, followed by a map to visualize the geographical distribution of these venues. Additionally, a heatmap highlights the winning teams at specific grounds, giving insights into team performance by location. Lastly, a correlation heatmap reveals the relationships between winning teams and the grounds where they tend to succeed.")
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








#Team1 vs Team2 Participations and Wins

elif section == "Team1 vs Team2 Participations and Wins":
    st.subheader("Team1 vs Team2 Participations and Wins")
    st.write("This section provides an in-depth comparison of Team 1 and Team 2 participation and wins across different years. The bar and line chart illustrates how both teams have participated and performed over the years, with bars representing total participation and lines showing total wins. Additionally, we explore the head-to-head performance between Team 1 and Team 2 in terms of victories. Two stacked bar charts visualize the number of wins each team secured against their counterpart, highlighting the competitive nature of these matchups.")
    if 'Match Year' not in wc_final_data_df.columns:
        wc_final_data_df['Match Year'] = pd.to_datetime(wc_final_data_df['Match Date'], errors='coerce').dt.year

    all_years = pd.Series(range(2007, 2025))
    # Team 1 participation and wins
    team1_participation = wc_final_data_df.groupby(['Match Year', 'Team1']).size().reset_index(name='Total Participation')
    team1_wins = wc_final_data_df[wc_final_data_df['Winner'] == wc_final_data_df['Team1']].groupby('Match Year').size().reset_index(name='Total Wins')
    # Merging participation and wins for Team 1
    team1_stats = pd.merge(all_years.to_frame(name='Match Year'), team1_participation, how='left', on='Match Year')
    team1_stats = pd.merge(team1_stats, team1_wins, how='left', on='Match Year')
    team1_stats['Total Participation'] = team1_stats['Total Participation'].fillna(0)  # Fill missing participation with 0
    team1_stats['Total Wins'] = team1_stats['Total Wins'].fillna(0)  # Fill missing wins with 0
    # Team 2 participation and wins
    team2_participation = wc_final_data_df.groupby(['Match Year', 'Team2']).size().reset_index(name='Total Participation')
    team2_wins = wc_final_data_df[wc_final_data_df['Winner'] == wc_final_data_df['Team2']].groupby('Match Year').size().reset_index(name='Total Wins')
    # Merging participation and wins for Team 2
    team2_stats = pd.merge(all_years.to_frame(name='Match Year'), team2_participation, how='left', on='Match Year')
    team2_stats = pd.merge(team2_stats, team2_wins, how='left', on='Match Year')
    team2_stats['Total Participation'] = team2_stats['Total Participation'].fillna(0)  
    team2_stats['Total Wins'] = team2_stats['Total Wins'].fillna(0) 

    fig_team1_team2 = go.Figure()
    fig_team1_team2.add_trace(go.Bar(
        x=team1_stats['Match Year'], 
        y=team1_stats['Total Participation'], 
        name='Team 1 Participation', 
        marker_color='blue',
        yaxis='y'
    ))
    fig_team1_team2.add_trace(go.Scatter(
        x=team1_stats['Match Year'], 
        y=team1_stats['Total Wins'], 
        mode='lines', 
        name='Team 1 Wins', 
        line=dict(color='red'),
        yaxis='y2' 
    ))
    fig_team1_team2.add_trace(go.Bar(
        x=team2_stats['Match Year'], 
        y=team2_stats['Total Participation'], 
        name='Team 2 Participation', 
        marker_color='green',
        yaxis='y'
    ))
    fig_team1_team2.add_trace(go.Scatter(
        x=team2_stats['Match Year'], 
        y=team2_stats['Total Wins'], 
        mode='lines', 
        name='Team 2 Wins', 
        line=dict(color='orange'),
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
    st.plotly_chart(fig_team1_team2)
    #
    win_counts = all_matches_data_df.groupby(['Winner', 'Team1', 'Team2']).size().reset_index(name='Wins')

    win_counts['Hover Text'] = win_counts.apply(lambda row: f"Team 1: {row['Team1']}<br>Team 2: {row['Team2']}<br>Winner: {row['Winner']}<br>Wins: {row['Wins']}", axis=1)
    color_map = {
        'India': 'lightblue',
        'Australia': 'lightgreen',
        'England': 'lightcoral',
        'Pakistan': 'lightpink',
        'South Africa': 'khaki',
        'Sri Lanka': 'lightsalmon',
        'West Indies': 'lavender',
        'Bangladesh': 'lightyellow',
        'Nepal': 'lightgray',
        'Zimbabwe': 'lightsteelblue',
        'Afghanistan': 'peachpuff',
        'New Zealand': 'powderblue',
        'Netherlands': 'lightblue',
        'Scotland': 'lightgreen',
        'U.S.A.': 'mistyrose',
        'Ireland': 'lightgreen',
        'Kenya': 'lightcyan',
        'Oman': 'lightseagreen',
        'United Arab Emirates': 'lightgoldenrodyellow',
        'Hong Kong': 'lightpink',
        'P.N.G': 'lightcoral',
        'Canada': 'lavenderblush',
        'Uganda': 'lightyellow',
        'No Result': 'whitesmoke',  
        'Tied': 'lightgrey'
    }
    fig_team1_over_team2 = px.bar(win_counts, 
                x='Team1', 
                y='Wins', 
                color='Winner', 
                title='Wins of Team 1 Against Team 2',
                labels={'Wins': 'Number of Wins', 'Team1': 'Team 1'},
                text='Wins',
                hover_name='Hover Text',
                color_discrete_map=color_map)
    fig_team1_over_team2.update_layout(
        xaxis_title='Team 1',
        yaxis_title='Number of Wins',
        hovermode='closest',
        barmode='stack',  
        xaxis_tickangle=-45  
    )
    fig_team2_over_team1 = px.bar(win_counts, 
                x='Team2', 
                y='Wins', 
                color='Winner', 
                title='Wins of Team 2 Against Team 1',
                labels={'Wins': 'Number of Wins', 'Team2': 'Team 2'},
                text='Wins',
                hover_name='Hover Text',
                color_discrete_map=color_map)  # Use the defined color map
    fig_team2_over_team1.update_layout(
        xaxis_title='Team 2',
        yaxis_title='Number of Wins',
        hovermode='closest',
        barmode='stack',  
        xaxis_tickangle=-45  
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_team1_over_team2, use_container_width=True)
    with col2:
        st.plotly_chart(fig_team2_over_team1, use_container_width=True)






#############################################################################################################################






#Players


elif section == "Player Participation":
    st.subheader("Player Participation")
    st.write("This section provides a comprehensive analysis of player participation trends over the years...")

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

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_funnel, use_container_width=True)
    with col2:
        st.dataframe(top_players_by_wins, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_longest_captains, use_container_width=True)
    with col2:
        st.dataframe(longest_captaincy, use_container_width=True)







#############################################################################################################################




#search for your favourite teams and players





elif section == "Search For Your Favourite Teams and Players":
    st.subheader("Look For Your Favourite Teams and Players!")

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









































































































