# Gambler_Pinata_AI

## Introduction
The **Gambler_Pinata_AI** is an AI-driven automation system designed to interact with online games. It features a modular architecture with components for browser automation, betting optimization, scatter detection, and user interaction.

## Features
- **Scatter Detection**: Uses OpenCV to identify scatter symbols and trigger actions.
- **Bet Optimization**: Dynamically adjusts betting strategies based on game history.
- **Browser Automation**: Automates browser actions using Selenium WebDriver.
- **Custom Configurations**: Supports dynamic rulesets and configurations via JSON files.
- **User Alerts**: Provides audio/visual alerts for critical events.
- **Auto Ingestion**: Monitors directories for new files and processes them.

## File Structure

### Core Functionality
- `gambler_pinata_ai/ai_gambler.py`: Main AI script for scatter detection and game control.
- `gambler_pinata_ai/ai_gambler_customized.py`: Loads custom configurations for the AI.
- `gambler_pinata_ai/main.py`: Entry point for the AI system.

### Betting Management
- `gambler_pinata_ai/bet_adjuster.py`: Automates bet adjustments.
- `gambler_pinata_ai/bet_optimizer.py`: Optimizes betting decisions based on history.

### Game Control
- `gambler_pinata_ai/hot_mode.py`: Toggles "Hot Mode" for enhanced features.
- `gambler_pinata_ai/alert_engine.py`: Sends alerts for special events.

### User Interface
- `gambler_pinata_ai/cortex_ui.py`: Terminal-based UI for interacting with the system.

### Automation Tools
- `gambler_pinata_ai/auto_ingest.py`: Watches directories and processes new files.
- `gambler_pinata_ai/auto_sentinel.py`: Manages surveillance tasks.

### Configurations and Assets
- `ruleset/tactic.json`: Defines game tactics and thresholds.
- `assets_config.json`: Maps game assets like symbols and buttons.

### Visual Assets
- `symbol_A_letter.png`, `symbol_shawarma_taco.png`, etc.: Images used for scatter detection and UI interactions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/1-ManArmy/Gambler_Pinata_AI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Gambler_Pinata_AI
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the following commands to use different components:
- Launch AI:
  ```bash
  python gambler_pinata_ai/main.py
  ```
- Adjust Bets:
  ```bash
  python gambler_pinata_ai/bet_adjuster.py
  ```

## Contributing
Contributions are welcome! Fork this repository, create a branch, and submit a pull request.

## License
[MIT License](LICENSE)
