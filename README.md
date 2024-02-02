# AWS Deepracer Reward Function Visualizer

**This is a work in progress. Checkout the TODOs and Feel free to contribute to this project by opening a pull request.**

This project aims to validate the reward function used in AWS Deepracer by visualizing the path taken by the race car given the reward function. The aim is to reduce unnecessary training when the reward function might have issues that can be ironed out by visualizing the "probable" path taken by the car when the agent is fully trained. **TAKE THE RESULT WITH A GRAIN OF SALT. This is not a replacement for actual training as there is more variables and interactions that this visualizer does not take into account.**

## Installation

Install the required packages using the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Other setups

Make sure the track you want to visualize is in the `tracks` directory. Download the tracks from [AWS Deepracer Community](https://github.com/aws-deepracer-community/deepracer-race-data/blob/main/raw_data/tracks/README.md)

## Usage

Run the following command to start the visualizer:

```bash
python main.py
```

## Liability

This project is not affiliated with AWS Deepracer. The visualizer is provided as is and should not be used as a replacement for actual training. The visualizer is a tool to help you understand the reward function better and should be used as a tool to help you debug the reward function. The visualizer does not take into account all the variables and interactions that the car might encounter in the real world. Use the visualizer at your own risk.
