# About 
This repository contains a convolutional neural network that is able to detect buildings on satellite imagery from SENTINEL-2. 


# Requirements

- We recommend a computer with a GPU 
- at least 24 GB RAM and additionally 16 GB swap memory available
- Python 3.x installed on your system. You can download it from the official [Python website](https://www.python.org/).
- The connection to copernicus.eu requires a registration which must be done manually.


# Setup Instructions

1. Clone this repository to your local machine

```bash
git clone https://github.com/christianstubbe/architecture-of-ml-systems
cd architecture-of-ml-systems
```

2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate # On Windows, use `venv\Scripts\activate`
```

3. Install Dependencies
    
```bash
pip install -r requirements.txt 
```

# Logging

Logs for this pipepline are stored in ```main.log```.