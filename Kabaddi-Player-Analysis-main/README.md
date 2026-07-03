# Overview

Kabaddi Player Analysis is a robust tool designed to facilitate in-depth analysis of kabaddi players through data-driven modeling. It provides a clear architectural overview with a comprehensive predictive pipeline, enabling developers and analysts to build, understand, and evaluate models that reveal key performance indicators.

## Why Kabaddi Player Analysis?

This project streamlines sports analytics by providing a modular framework for player impact prediction. Core features include:

### 🔬 Model Pipeline: 
Aggregates season-level data, engineers key features, and trains regression models to predict player efficiency.

### 📊 Model Evaluation: 
Supports hyperparameter tuning and feature importance analysis to optimize model performance.

### 🏗️ System Architecture: 
Offers detailed documentation for easy onboarding and maintenance.

### 🚀 Strategic Insights: 
Enables data-driven decision-making for team strategies and player development.

### 🛠️ Developer-Friendly: 
Modular design for customization and seamless integration into larger analytics workflows.

# Getting Started
## Prerequisites
### Make sure you have the following installed:

 Programming Language: Python (>=3.8 recommended)

 Package Manager: Conda (recommended) or pip

## Installation
### Build Kabaddi Player Analysis from the source and install dependencies:

1. Clone the repository:

        git clone https://github.com/omraut111/Kabaddi-Player-Analysis
   
2. Navigate to the project directory:
   
        cd Kabaddi-Player-Analysis
   
## Install the dependencies:

## Using Conda:

        conda env create -f conda.yml
        
## Or using pip:

        pip install -r requirements.txt

        
# Usage
### Run the project:

### (Conda recommended)


        conda activate venv
        python <entrypoint_script>.py
        
Replace <entrypoint_script>.py with the main python file (e.g., kabaddi_player_impact_predictor_complete.py).

## Testing
This project uses the pytest framework. To run tests:


        conda activate venv
        pytest
