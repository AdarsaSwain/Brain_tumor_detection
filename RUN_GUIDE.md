# Brain Tumor Detection - Installation and Run Guide

This document provides step-by-step instructions to set up and run the Brain Tumor Detection application on your local machine.

## Prerequisites

1.  **Python 3.9 - 3.11**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
    *   *Note: Python 3.12+ might have compatibility issues with some older TensorFlow versions used in this project.*
2.  **Windows OS**: These instructions are tailored for Windows.

---

## Quick Start (Automatic)

We have provided a script that handles everything (environment setup, dependency installation, and running the server).

1.  Open the project folder.
2.  Double-click on **`run_app.bat`**.
3.  Wait for the dependencies to install (this may take 5-10 minutes the first time due to TensorFlow's size).
4.  Once the server is running, open your browser and go to:
    [http://localhost:5000](http://localhost:5000)

---

## Manual Installation

If you prefer to set up the environment manually, follow these steps:

### 1. Create a Virtual Environment
Open a terminal (PowerShell or Command Prompt) in the project's `src` directory and run:
```powershell
python -m venv venv
```

### 2. Activate the Environment
```powershell
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Application
```powershell
python app.py
```

---

## Troubleshooting

### 1. "Long Path" Error during Installation
If you see an error related to "Long Paths" while installing TensorFlow:
*   **Solution A**: Move the project folder to a shorter path (e.g., `C:\projects\BrainTumor`).
*   **Solution B**: Enable Long Paths in Windows (Advanced):
    *   Press `Win + R`, type `regedit`, and hit Enter.
    *   Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
    *   Set `LongPathsEnabled` to `1`.

### 2. Model Files Not Found
Ensure that the `.h5` model files are present in the `src` directory:
*   `epoch10_sgd_acc96Point76.h5`
*   `multi-model-30K-epouch20.h5`

---

## Project Structure
*   **`src/`**: Contains the Flask application and models.
*   **`src/templates/`**: HTML files for the web interface.
*   **`src/static/`**: CSS and JS files.
*   **`Model Codes/`**: Jupyter notebooks used for training the models.
