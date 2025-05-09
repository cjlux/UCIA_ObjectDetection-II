######################################
#   Jean-Luc.Charles@mailo.com
#   2024/11/21 - v1.0
######################################

from ultralytics import settings

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]
