# pypsa-fbmc

This package adds FBMC capabilities for the pypsa package. It further introduces several methods for GSK calculation.

WARNING: While the notebooks work, at this stage of development, the package will contain bugs and is not meant for production.

## Setup Instructions

To run the FBMC module, follow these steps:

1. **Install Python 3.11.x** (e.g. from https://www.python.org/downloads/release/python-3118/)

2. **Clone the repository:**

```bash
git clone https://github.com/ameldekok/pypsa-fbmc.git
cd pypsa-fbmc
```

3. **Create and activate a virtual environment:**

- On **Windows (Command Prompt)**:

```cmd
py -3.11 -m venv venv
venv\Scripts\activate.bat
```

- On **macOS/Linux**:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

4. **Install dependencies:**

```bash
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

5. **Activate gurobi (or another solver)**

Package included in dependencies; Instructions for licence and further installation can be found here: https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer.

It is also possible to choose another optimiser, but you will have to update the `solver =`to other options within the code.

6. **Run the code:**

You can now import and use the module in your Python scripts or Jupyter notebooks. For example, start with one of the jupyter notebooks.