
# 0.20

### CLI
- Restructured to use typer for CLI
- Restructed the register command to try to connect to the API before prompting for wallet etc

### Auth
- Removed the 'X-Parachutes-Auth' header, as it is not necessary if we have 'X-Parachutes-Hotkey' and 'X-Parachutes-Signature'.

### Generic
- Changed config to be a singleton, to prevent errors at import time
- Constants for headers for easier use
- Added rich dependency
- Fixed the vllm template to not use chunked prefill