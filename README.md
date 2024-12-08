# **Men's Cricket World Cup Analytics and Insights**

The **Men's Cricket World Cup** is one of the most prestigious tournaments in the cricketing calendar, bringing together the best teams and players from around the world. This project dives deep into the rich history of the tournament, leveraging advanced data analysis and visualization techniques to uncover trends, dynamics, and stories that define international cricket. From match outcomes to team performances and individual player statistics, this project offers a comprehensive look at the game.

## **Project Highlights**

### **Interactive Streamlit App**
This project is deployed as a **Streamlit app**, enabling users to interactively explore and visualize cricket data. The app is structured into engaging sections, each designed to showcase different facets of cricket analytics. 

### **Key Features**
1. **Dataset Overview**: Summary and exploration of the dataset used in the analysis.
2. **Distribution of Key Numeric Features**: Visual representation of key numeric statistics such as match margins, player performance, and win percentages.
3. **Distribution of Ranking Differences**: Analyze ranking differences between teams and visualize their distribution.
4. **Matches**: Detailed breakdown of the matches played, including team head-to-head statistics.
5. **Wins**: Total number of wins by each team, visualized across multiple matches and venues.
6. **Grounds**: Analyze performance by venue, providing insights into how teams perform at specific grounds.
7. **Team1 vs Team2 Participations and Wins**: Explore participation and win statistics for Team1 vs Team2 matchups.
8. **Player Participation**: Detailed visualization of player participation statistics.
9. **Search For Your Favourite Teams and Players**: Search functionality to filter the dataset for specific teams and players.
10. **Data Journey: From Raw to Revelations**: Combining **IDA** and **EDA** efforts, including T-20 match number extraction and splitting match margins into `Margin (Runs)` and `Margin (Wickets)`.
11. **Cracking the Mystery of Missingness**: Insights into structured missingness patterns, imputation methods, and MAR analysis.
12. **Modeling Matches**: Learn how machine learning predicts match outcomes using Logistic Regression, Random Forest, and XGBoost.
13. **Champion Forecasts**: Simulate matchups to forecast the next T20 World Cup champion.

---

## **Goals of the Project**
- **Understand Competitiveness**: Analyze team rivalries and performance over time.  
- **Uncover Patterns**: Explore trends in match outcomes, margins, and player contributions.  
- **Predict Outcomes**: Use machine learning to forecast match results and tournament champions.  
- **Enhance Accessibility**: Provide an interactive app for users to engage with the data and uncover their own insights.

---

## **Technical Highlights**
- **Feature Engineering**: Transforming raw data into features like rolling averages, ranking differences, and home advantage.
- **Missingness Handling**: Analyzing and addressing structured missingness using advanced statistical techniques.
- **Interactive Visualizations**: Leveraging Plotly and Streamlit to create engaging and dynamic visuals.
- **Machine Learning Models**: Comparing Logistic Regression, Random Forest, and XGBoost to predict match outcomes.
- **Deployment**: Hosted as a **Streamlit app** for interactive exploration and analysis.

---

Explore the cricketing world like never before with this data-driven journey through the Men’s Cricket World Cup. **Dive in and uncover the stories hidden in the numbers!**


**Datasets Used:**
1. all_matches_data_df: Contains data on match results, teams, grounds, dates, and winning margins.
2. wc_final_data_df: Detailed dataset for World Cup matches including win percentages and match-specific statistics, like batting and bowling averages, etc.
3. players_df: Player information for matches, loaded separately from a CSV file.
4. captains_df: Dataset for the captainf of all of the teams in the all_matches_data_df and wc_final_data_df, for the years 2007-2024
    
### **Live Demo**

You can access the deployed Streamlit app directly without needing to install anything locally:
````
https://cricketfever.streamlit.app/
`````

### **Local Setup Instructions**

If you want to run the app locally for development or testing purposes, follow the steps below.

**Prerequisites**

Ensure you have the following installed:
      Python 3.8 and the Required Python libraries (listed below)
    
Installation Steps
1. Clone the Repository:
   `````
   git clone [https://github.com/Shamsvi/Streamlit_Mens_T-20_Cricket_WorldCup_2007-2024.git]

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
   streamlit run MCW.py
7. Access the App

    After running the Streamlit command, the app will be accessible locally in your browser. Navigate to:
    `````
    http://localhost:8501
    `````

