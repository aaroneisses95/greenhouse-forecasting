"Module with the constants used within the package"
import os

# The path to the csv with the weather data
PATH_WEATHER_CSV = os.path.abspath(f"data/Weather.csv")

# The available teams that can be used to train the data
TEAMS = ["AICU", "Automatoes", "Digilog", "Grower", "IUACAAS", "TheAutomators"]

# The features that have been selected during the exploration. These are based on training on
# the dataset of the Automatoes.
SELECTED_FEATURES = ["Iglob", "Rhout", "Tout", "HumDef", "Tair", "t_heat_sp"]
