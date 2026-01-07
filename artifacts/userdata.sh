#!/bin/bash

##########################


##############################
# User Data Script to set up Virtual Machines in AWS Cloud infrastucture 
# Author: Firstname Lastname
# Date: 2023-06-01  
# Version: 1.0
# Description: This script sets up the virtual machines in AWS cloud infrastucture, installs the required dependencies, and configures the necessary services.
# Usage: bash userdata.sh   
# Example: bash userdata.sh
# Note: Ensure the script is executable by running "chmod +x userdata.sh" before executing it.
##############################

Git Clone the code repository
update dependencies
install python on VM
Setup Python virtual environment
python requirements installation
python build model train.py
pythin run app app.py
Create WSGI as linux systemd service
Create NGNIX as linux systemd service
Enable these services to run on startup
Start the services