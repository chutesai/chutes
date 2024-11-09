
# 0.17
- Added rich dependency
- Restructed the register command to try to connect to the API before prompting for wallet etc
- Changed config to be a singleton, to prevent errors at import time
- Restructured to use typer for CLI