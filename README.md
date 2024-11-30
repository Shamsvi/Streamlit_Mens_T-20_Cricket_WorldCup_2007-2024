**Mens T-20 Worldcup Analysis: 2007-2024**

The Men's Cricket World Cup is one of the most prestigious tournaments in the cricketing calendar, showcasing the talents of the world's best teams and players. This project aims to analyze and visualize key data from the tournament, focusing on match outcomes, team performances, and individual player statistics. By leveraging advanced data analysis techniques, we will explore trends in match margins, batting and bowling averages, and historical rivalries. Through this comprehensive analysis, we seek to provide valuable insights into the dynamics of the tournament, enhancing our understanding of competitiveness and performance in international cricket.

**Key Features:**
1.  Dataset Overview: Summary and exploration of the dataset used in the analysis.
2.  Distribution of Key Numeric Features: Visual representation of key numeric statistics such as match margins, player performance, and win percentages.
3.  Distribution of Ranking Differences: Analyze ranking differences between teams and visualize their distribution.
4.  Matches: Detailed breakdown of the matches played, including team head-to-head statistics.
5.  Wins: Total number of wins by each team, visualized across multiple matches and venues.
6.  Grounds: Analyze performance by venue, providing insights into how teams perform at specific grounds.
7.  Team1 vs Team2 Participations and Wins: Explore participation and win statistics for Team1 vs Team2 matchups.
8.  Player Participation: Detailed visualization of player participation statistics.
9.  Search For Your Favourite Teams and Players: Search functionality to filter the dataset for specific teams and players.
10. The project is deployed as a Streamlit app that allows users to interactively explore the data and derive insights.

**Datasets Used:**
1. all_matches_data_df: Contains data on match results, teams, grounds, dates, and winning margins.
2. wc_final_data_df: Detailed dataset for World Cup matches including win percentages and match-specific statistics, like batting and bowling averages, etc.
3. players_df: Player information for matches, loaded separately from a CSV file.
4. captains_df: Dataset for the captainf of all of the teams in the all_matches_data_df and wc_final_data_df, for the years 2007-2024
    
**Live Demo**

You can access the deployed Streamlit app directly without needing to install anything locally:
````
https://mens-cricket-t20-worldcup2007-2024.streamlit.app
`````

**Local Setup Instructions**

If you want to run the app locally for development or testing purposes, follow the steps below.

**Prerequisites**

Ensure you have the following installed:
      Python 3.8 and the Required Python libraries (listed below)
    
Installation Steps
1. Clone the Repository:
   `````
   git clone https://github.com/Shamsvi/CMSE-830.git
   cd CMSE-830/MidtermPorject

2. Set Up a Virtual Environment (Optional but recommended)

   To avoid dependency conflicts, it’s recommended to use a virtual environment:
   ````
   python3 -m venv .venv
   source .venv/bin/activate   # For Windows: .venv\Scripts\activate
4. Install Required Libraries
   ````
    pip install -r requirements.txt
5. Running the Streamlit App Locally

    Once all dependencies are installed, you can run the Streamlit app with the following command:
   `````
   streamlit run app.py
7. Access the App

    After running the Streamlit command, the app will be accessible locally in your browser. Navigate to:
    `````
    http://localhost:8501
    `````

