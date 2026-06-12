Bibliometric Insights Web System

A web-based system for bibliometric analysis using BERTopic and Topological Data Analysis (TDA). The system enables researchers to upload bibliographic datasets, perform topic modelling, visualize topic evolution, and explore relationships between research topics through interactive dashboards.

Features:
- Topic modelling using BERTopic
- Topic label generation
- Interactive visualization dashboard
- Topic evolution analysis
- Mapper graph generation using TDA
- Bibliographic data upload and processing
- Research trend exploration

Technologies Used:
- Python
- Flask
- BERTopic
- Sentence Transformers
- Plotly
- Pandas
- HTML/CSS/JavaScript

Project Structure:

BibliometricProject/
│
├── data/                   # Input and processed datasets
├── static/                 # CSS, JS, images, generated visualizations
├── templates/              # HTML templates
│
├── app.py                  # Main Flask application
├── bib_reader.py           # Reads bibliographic data
├── preprocess_bib.py       # Data preprocessing
├── bertopic_run.py         # BERTopic modelling
├── apply_topic_labels.py   # Topic label generation
├── topic_evolution.py      # Topic evolution analysis
├── run_mapper.py           # TDA Mapper visualization
└── persistence_run.py      # Persistent homology analysis

Installation:

1. Clone Repository
git clone https://github.com/akmalwafi/BibliometricProject.git
cd BibliometricProject

2. Create Virtual Environment
python -m venv .venv

Activate environment:

Windows:
.venv\Scripts\activate

Linux/Mac:
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

Workflow (PyCharm Users):

Since this project is run inside PyCharm, start by running app.py. The Flask web application will handle the rest of the steps automatically when you interact with the web interface.

Step 1: Run the Flask App
python app.py

Open your browser:
http://127.0.0.1:5000

The web app interface allows you to:
- Upload a bibliographic dataset (.bib or .csv)
- Preprocess the data
- Generate BERTopic topics
- Apply topic labels
- Perform topic evolution analysis
- Run TDA (Mapper and Persistent Homology)
- Explore results through interactive visualizations

System Flow:

Run app.py → Open Web Interface
           │
           ▼
       Upload Dataset
           │
           ▼
     Preprocessing & Cleaning
           │
           ▼
     BERTopic Modelling
           │
           ▼
     Topic Label Generation
           │
           ▼
     Topic Evolution Analysis
           │
           ▼
     TDA Analysis (Mapper & Persistent Homology)
           │
           ▼
Interactive Dashboard & Visualization

Note: You don’t need to run the other Python scripts manually. They are called automatically by the web interface when you perform actions like uploading a dataset or generating topics.

Author:
Muhammad Wafi
Bachelor of Computer Science (Hons.)
Universiti Teknologi MARA (UiTM)
