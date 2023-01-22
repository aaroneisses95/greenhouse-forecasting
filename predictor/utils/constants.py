"Module with the constants used within the package"
from pathlib import Path

PATH_WEATHER_CSV = (
    Path(__file__).resolve().parent.parent.parent.joinpath("data/Weather.csv")
)

TEAMS = ["AICU", "Automatoes", "Digilog", "Grower", "IUACAAS", "TheAutomators"]
