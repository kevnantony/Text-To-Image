# Project Overview

This project involves the development of a machine learning model and an API to serve and interact with this model. The project includes multiple components, such as model loading, API handling, and a Gradio demo interface for user interaction. Below is a detailed overview of how each part of the project works.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Model Loading](#model-loading)
4. [API Endpoints](#api-endpoints)
5. [Gradio Interface](#gradio-interface)
6. [Docker Integration](#docker-integration)

# Project Structure

The project is organized into several key directories and files:

```plaintext
src/
  api/
    app.py
    main.py
    handlers.py
    api.py
    req.py
    Dockerfile
    requirement.txt
  model/
    model_loading.py
    inference.py
  utils/
    helper_functions.py
gradio_demo/
  dem0-app.py
```

# Setup and Installation

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.11 or higher**: Make sure Python is installed on your system.
- **Docker**: Ensure Docker is installed and running on your machine.
- **Virtual Environment**: It is recommended to use a virtual environment to manage dependencies.

## Clone the Repository

Start by cloning the project repository from GitHub:

```bash
git clone <repository-url>
cd <repository-folder>
```

## Clone and Activate Virtual Environment
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
```
## Install dependencies
```bash
pip install -r src/requirements.txt
```
# Model Loading
The model is loaded using the scripts in the model/ directory.
- model_loading.py: This script is responsible for loading the machine learning model into memory. It ensures the model is available for inference when the API is called.
- inference.py: This script handles the inference logic, taking input data, processing it, and returning predictions.

# API Endpoints
The API is designed to interact with the loaded model and handle requests from users.
main.py: The entry point of the API. It sets up the necessary routes and initializes the FastAPI server.
- handlers.py: Contains the logic for handling different API requests, including prediction requests, and error handling.
- api.py: Defines the API routes and connects them to the corresponding handler functions.

# Gradio Interface
The gradio_demo/ directory contains a demo application using Gradio for easy interaction with the model.
- demoapp.py: This script sets up a Gradio interface that allows users to interact with the model through a web interface. Users can upload images, enter text, or provide other inputs, and receive real-time predictions or responses.

# Docker Integration
- Build the Docker Image:
```bash
docker build -t my-app .
```
- Run the Docker Container:
```bash
docker run --gpus all --ipc=host -p <PORT_User>:<PORT_Docker> my-app
```

### This project integrates a machine learning model with a FastAPI-based API and a Gradio interface, all packaged within a Docker container for easy deployment. The structure allows for modular development and easy expansion of features.