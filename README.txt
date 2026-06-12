# Bibliometric Insights Web System

A web-based system for bibliometric analysis using BERTopic and Topological Data Analysis (TDA). The system enables researchers to upload bibliographic datasets, perform topic modelling, visualize topic evolution, and explore relationships between research topics through interactive dashboards.

---

## Features

- Topic modelling using BERTopic
- Topic label generation
- Interactive visualization dashboard
- Topic evolution analysis
- Mapper graph generation using TDA
- Bibliographic data upload and processing
- Research trend exploration

---

## Technologies Used

- Python
- Flask
- BERTopic
- Sentence Transformers
- Plotly
- Pandas
- HTML/CSS/JavaScript

---

## Project Structure

```text
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
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/akmalwafi/BibliometricProject.git
cd BibliometricProject
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

Windows

```bash
.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Workflow

### Step 1: Prepare Bibliographic Dataset

Place your bibliographic dataset inside:

```text
data/
```

Supported formats:

- BibTeX (.bib)
- CSV

---

### Step 2: Preprocess Data

Run:

```bash
python preprocess_bib.py
```

This step:

- Cleans records
- Removes missing values
- Prepares text for topic modelling

---

### Step 3: Generate Topics Using BERTopic

Run:

```bash
python bertopic_run.py
```

This step:

- Creates document embeddings
- Clusters documents
- Generates topics

Output:

```text
data/processed/
```

---

### Step 4: Apply Topic Labels

Run:

```bash
python apply_topic_labels.py
```

This step:

- Generates human-readable topic labels
- Improves topic interpretation

---

### Step 5: Generate Topic Evolution Analysis

Run:

```bash
python topic_evolution.py
```

This step:

- Tracks topic changes over time
- Produces trend visualizations

---

### Step 6: Generate TDA Visualizations

Mapper Graph:

```bash
python run_mapper.py
```

Persistent Homology:

```bash
python persistence_run.py
```

This step:

- Reveals hidden structures in the dataset
- Identifies relationships between research topics

---

### Step 7: Launch Web Application

Run:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## System Flow

```text
Upload Dataset
       │
       ▼
Data Preprocessing
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
TDA Analysis
 ├── Mapper Graph
 └── Persistent Homology
       │
       ▼
Interactive Dashboard
```

---

## Sample Outputs

- Topic Distribution
- Topic Evolution Charts
- BERTopic Visualizations
- Mapper Graph
- Persistent Homology Diagram

---

## Future Improvements

- Real-time analysis
- Additional topic modelling techniques
- Advanced filtering options
- Multi-dataset comparison

---

## Author

**Muhammad Wafi**  
Bachelor of Computer Science (Hons.)  
Universiti Teknologi MARA (UiTM)
