
# 0.20

### CLI
- Restructured to use typer for CLI
- Restructed the register command to try to connect to the API before prompting for wallet etc
- Change Regex to allow nested directories in chute names (e.g. bish/bash:bosh instead of only bash:bosh)

### Auth
- Removed the 'X-Parachutes-Auth' header, as it is not necessary if we have 'X-Parachutes-Hotkey' and 'X-Parachutes-Signature'.

### Generic
- Changed config to be a singleton, to prevent errors at import time
- Constants for headers for easier use
- Added rich dependency & typer
- Fixed the vllm template to not use chunked prefill
