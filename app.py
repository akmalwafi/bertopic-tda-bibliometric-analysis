import os
import uuid
import json
import threading
import traceback
import time
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename

from pipeline import run_all, run_tda_only
from bib_reader import load_multiple_bibs

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RUNTIME_HISTORY_PATH = os.path.join(BASE_DIR, "data", "runtime_history.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"bib"}
MAX_FILES = 5
MAX_TOTAL_ENTRIES = 3000

# job_id -> {
#   "status": "...",
#   "error": None|str,
#   "outputs": dict,
#   "dataset_id": str|None,
#   "message": str,
#   "total_entries": int,
#   "start_time": float,
#   "estimated_seconds": int,
#   "progress": int,
#   "last_update_time": float,
#   "job_type": "bertopic"|"tda"
# }
JOBS = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠ Failed to read JSON: {path} -> {e}")
    return default


def save_runtime_history(total_entries: int, elapsed_seconds: int):
    """
    Save real completed runtime so future estimated time can become more accurate.
    """
    try:
        os.makedirs(os.path.dirname(RUNTIME_HISTORY_PATH), exist_ok=True)

        history = load_json_safe(RUNTIME_HISTORY_PATH, [])

        history.append({
            "total_entries": total_entries,
            "elapsed_seconds": elapsed_seconds,
            "time": time.time()
        })

        # Keep only latest 20 successful runs
        history = history[-20:]

        with open(RUNTIME_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    except Exception as e:
        print(f"⚠ Failed to save runtime history: {e}")


def estimate_from_history(total_entries: int):
    """
    Estimate runtime based on previous successful runs with similar dataset sizes.
    """
    history = load_json_safe(RUNTIME_HISTORY_PATH, [])

    if not history:
        return None

    similar_runs = []

    for item in history:
        old_entries = item.get("total_entries", 0)
        old_seconds = item.get("elapsed_seconds", 0)

        if old_entries <= 0 or old_seconds <= 0:
            continue

        lower = total_entries * 0.5
        upper = total_entries * 1.5

        if lower <= old_entries <= upper:
            seconds_per_entry = old_seconds / old_entries
            similar_runs.append(seconds_per_entry)

    if not similar_runs:
        return None

    avg_seconds_per_entry = sum(similar_runs) / len(similar_runs)
    return int(avg_seconds_per_entry * total_entries)


def estimate_pipeline_seconds(total_entries: int) -> int:
    """
    Estimate processing time for the first stage:
    preprocess + SBERT embeddings + BERTopic + topic labeling.
    """
    if total_entries <= 0:
        return 120

    history_estimate = estimate_from_history(total_entries)

    if history_estimate:
        return max(60, min(1800, history_estimate))

    # Fallback estimate for first-time runs
    if total_entries <= 100:
        estimated = 90
    elif total_entries <= 500:
        estimated = int(total_entries * 0.45)
    elif total_entries <= 1500:
        estimated = int(total_entries * 0.55)
    else:
        estimated = int(total_entries * 0.65)

    return max(60, min(1800, estimated))


def estimate_tda_seconds(dataset_id: str) -> int:
    """
    Estimate processing time for TDA Mapper + persistence.
    """
    try:
        cleaned_csv = os.path.join(PROCESSED_DIR, dataset_id, "cleaned_docs.csv")

        if not os.path.exists(cleaned_csv):
            return 180

        with open(cleaned_csv, "r", encoding="utf-8", errors="ignore") as f:
            total_rows = max(0, sum(1 for _ in f) - 1)

        if total_rows <= 100:
            return 90
        elif total_rows <= 500:
            return 180
        elif total_rows <= 1500:
            return 360
        else:
            return 600

    except Exception:
        return 180


def is_valid_dataset_id(dataset_id: str) -> bool:
    """
    Make sure dataset_id points only inside data/processed.
    """
    if not dataset_id:
        return False

    dataset_root = os.path.abspath(os.path.join(PROCESSED_DIR, dataset_id))
    allowed_base = os.path.abspath(PROCESSED_DIR)

    if os.path.commonpath([dataset_root, allowed_base]) != allowed_base:
        return False

    return os.path.exists(dataset_root)


def get_dataset_paths(dataset_id: str) -> dict:
    """
    Build paths for the existing data/processed/<dataset_id> structure.
    """
    root = os.path.join(PROCESSED_DIR, dataset_id)

    bertopic_dir = os.path.join(root, "bertopic")
    tda_dir = os.path.join(root, "tda")
    embeddings_dir = os.path.join(root, "embeddings")

    return {
        "dataset_id": dataset_id,
        "root": root,

        "cleaned_csv": os.path.join(root, "cleaned_docs.csv"),

        "embeddings_dir": embeddings_dir,
        "embeddings_npy": os.path.join(embeddings_dir, "sbert_embeddings.npy"),

        "bertopic_dir": bertopic_dir,
        "topic_info_csv": os.path.join(bertopic_dir, "topic_info.csv"),
        "doc_topics_csv": os.path.join(bertopic_dir, "doc_topics.csv"),
        "topic_labels_csv": os.path.join(bertopic_dir, "topic_labels.csv"),
        "barchart_html": os.path.join(bertopic_dir, "barchart_all_topics.html"),
        "topics_map_html": os.path.join(bertopic_dir, "topics_map.html"),
        "topics_keywords_txt": os.path.join(bertopic_dir, "topics_keywords.txt"),

        "tda_dir": tda_dir,
        "mapper_html": os.path.join(tda_dir, "mapper_graph.html"),
        "mapper_node_summaries_json": os.path.join(tda_dir, "mapper_node_summaries.json"),
        "mapper_global_summary_json": os.path.join(tda_dir, "mapper_global_summary.json"),
        "persistence_diagram_png": os.path.join(tda_dir, "persistence_diagram.png"),
        "persistence_barcode_png": os.path.join(tda_dir, "persistence_barcode.png"),
    }


def get_bertopic_state(dataset_id: str):
    """
    Check whether BERTopic output exists.
    """
    if not dataset_id:
        return False

    paths = get_dataset_paths(dataset_id)

    return (
        os.path.exists(paths["topic_info_csv"])
        and os.path.exists(paths["doc_topics_csv"])
        and (
            os.path.exists(paths["barchart_html"])
            or os.path.exists(paths["topics_map_html"])
        )
    )


def get_tda_state(dataset_id: str):
    """
    Check whether TDA output exists.
    """
    if not dataset_id:
        return False, [], {}

    paths = get_dataset_paths(dataset_id)

    mapper_html = paths["mapper_html"]
    node_summary_path = paths["mapper_node_summaries_json"]
    global_summary_path = paths["mapper_global_summary_json"]

    tda_done = (
        os.path.exists(mapper_html)
        and os.path.exists(node_summary_path)
        and os.path.exists(global_summary_path)
    )

    if tda_done:
        node_summaries = load_json_safe(node_summary_path, [])
        global_summary = load_json_safe(global_summary_path, {})
    else:
        node_summaries = []
        global_summary = {}

    return tda_done, node_summaries, global_summary


def get_dashboard_outputs(dataset_id: str) -> dict:
    """
    Only basic output state for dashboard.
    Actual graph display is moved to separate pages.
    """
    paths = get_dataset_paths(dataset_id)

    outputs = {}

    if os.path.exists(paths["barchart_html"]):
        outputs["BERTopic - All Topics Barchart"] = paths["barchart_html"]

    if os.path.exists(paths["topics_map_html"]):
        outputs["BERTopic - Topics Map"] = paths["topics_map_html"]

    if os.path.exists(paths["mapper_html"]):
        outputs["TDA Mapper Graph"] = paths["mapper_html"]

    if os.path.exists(paths["persistence_diagram_png"]):
        outputs["TDA - Persistence Diagram"] = paths["persistence_diagram_png"]

    if os.path.exists(paths["persistence_barcode_png"]):
        outputs["TDA - Persistence Barcode"] = paths["persistence_barcode_png"]

    return outputs


def get_topic_percentage_data(dataset_id: str):
    """
    Build topic percentage data from BERTopic topic_info.csv.
    Excludes Topic -1 because -1 means outlier/unclassified documents.
    """
    paths = get_dataset_paths(dataset_id)

    topic_info_path = paths["topic_info_csv"]
    topic_labels_path = paths["topic_labels_csv"]

    if not os.path.exists(topic_info_path):
        return []

    try:
        topic_info = pd.read_csv(topic_info_path)
    except Exception as e:
        print("⚠ Failed to read topic_info.csv:", e)
        return []

    if "Topic" not in topic_info.columns or "Count" not in topic_info.columns:
        return []

    # Exclude outlier topic -1
    topic_info = topic_info[topic_info["Topic"] != -1].copy()

    if topic_info.empty:
        return []

    # Load readable topic labels if available
    label_map = {}

    if os.path.exists(topic_labels_path):
        try:
            labels_df = pd.read_csv(topic_labels_path)

            if "Topic" in labels_df.columns and "Topic_Label" in labels_df.columns:
                labels_df["Topic"] = labels_df["Topic"].astype(int)
                label_map = dict(zip(labels_df["Topic"], labels_df["Topic_Label"]))
        except Exception as e:
            print("⚠ Failed to load topic labels for percentage chart:", e)

    total_docs = topic_info["Count"].sum()

    if total_docs <= 0:
        return []

    chart_data = []

    for _, row in topic_info.iterrows():
        try:
            topic_id = int(row["Topic"])
            count = int(row["Count"])
            percentage = round((count / total_docs) * 100, 2)

            topic_name = label_map.get(topic_id)

            if not topic_name:
                if "CustomName" in topic_info.columns and pd.notna(row.get("CustomName")):
                    topic_name = str(row.get("CustomName"))
                elif "Name" in topic_info.columns and pd.notna(row.get("Name")):
                    topic_name = str(row.get("Name"))
                else:
                    topic_name = f"Topic {topic_id}"

            chart_data.append({
                "topic_id": topic_id,
                "topic_name": topic_name,
                "count": count,
                "percentage": percentage
            })

        except Exception:
            continue

    chart_data = sorted(chart_data, key=lambda x: x["percentage"], reverse=True)

    return chart_data


def get_topic_timeline_data(dataset_id: str):
    """
    Build topic timeline data using:
    - doc_topics.csv: topic assigned to each paper
    - cleaned_docs.csv: publication year of each paper

    Output:
    topic_id, topic_name, oldest_year, newest_year, year_range, count
    """
    paths = get_dataset_paths(dataset_id)

    doc_topics_path = paths["doc_topics_csv"]
    cleaned_docs_path = paths["cleaned_csv"]
    topic_labels_path = paths["topic_labels_csv"]

    if not os.path.exists(doc_topics_path) or not os.path.exists(cleaned_docs_path):
        return []

    try:
        doc_topics = pd.read_csv(doc_topics_path)
        cleaned_docs = pd.read_csv(cleaned_docs_path)
    except Exception as e:
        print("⚠ Failed to read topic timeline files:", e)
        return []

    topic_col = "topic" if "topic" in doc_topics.columns else "Topic"
    year_col = "year" if "year" in cleaned_docs.columns else "Year"

    if topic_col not in doc_topics.columns or year_col not in cleaned_docs.columns:
        return []

    df = cleaned_docs.copy()

    # Align by row order. This matches your pipeline because doc_topics.csv
    # is created from the same cleaned_docs order.
    df["topic"] = doc_topics[topic_col]
    df["year"] = pd.to_numeric(df[year_col], errors="coerce")

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Exclude BERTopic outlier/unclassified topic
    df = df[df["topic"] != -1]

    if df.empty:
        return []

    label_map = {}

    if os.path.exists(topic_labels_path):
        try:
            labels_df = pd.read_csv(topic_labels_path)

            if "Topic" in labels_df.columns and "Topic_Label" in labels_df.columns:
                labels_df["Topic"] = labels_df["Topic"].astype(int)
                label_map = dict(zip(labels_df["Topic"], labels_df["Topic_Label"]))
        except Exception as e:
            print("⚠ Failed to load topic labels for timeline chart:", e)

    timeline_data = []

    for topic_id, group in df.groupby("topic"):
        try:
            topic_id = int(topic_id)
            oldest_year = int(group["year"].min())
            newest_year = int(group["year"].max())
            count = int(len(group))
            year_range = newest_year - oldest_year

            timeline_data.append({
                "topic_id": topic_id,
                "topic_name": label_map.get(topic_id, f"Topic {topic_id}"),
                "oldest_year": oldest_year,
                "newest_year": newest_year,
                "year_range": year_range,
                "count": count
            })

        except Exception:
            continue

    # Sort: newest/active topics first
    timeline_data = sorted(
        timeline_data,
        key=lambda x: (x["newest_year"], x["oldest_year"]),
        reverse=True
    )

    return timeline_data


# ============================================================
# MAIN ROUTES
# ============================================================

@app.get("/")
def index():
    return render_template("index.html")


def _run_job(job_id: str, bib_paths):
    """
    First stage job:
    Run BERTopic only through pipeline.py run_all().
    """
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["message"] = "BERTopic pipeline is starting..."
        JOBS[job_id]["progress"] = 3
        JOBS[job_id]["last_update_time"] = time.time()

        def progress_callback(percent, message):
            JOBS[job_id]["progress"] = int(percent)
            JOBS[job_id]["message"] = message
            JOBS[job_id]["last_update_time"] = time.time()

        result = run_all(bib_paths, progress_callback=progress_callback)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["error"] = None
        JOBS[job_id]["outputs"] = result.get("outputs", {})
        JOBS[job_id]["dataset_id"] = result.get("dataset_id")
        JOBS[job_id]["message"] = "BERTopic analysis completed successfully."
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["last_update_time"] = time.time()

        elapsed_seconds = int(time.time() - JOBS[job_id].get("start_time", time.time()))
        total_entries = JOBS[job_id].get("total_entries", 0)

        save_runtime_history(total_entries, elapsed_seconds)

    except Exception:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = traceback.format_exc()
        JOBS[job_id]["outputs"] = {}
        JOBS[job_id]["dataset_id"] = None
        JOBS[job_id]["message"] = "BERTopic pipeline failed."
        JOBS[job_id]["last_update_time"] = time.time()


def _run_tda_job(job_id: str, dataset_id: str):
    """
    Second stage job:
    Run TDA Mapper + persistence only after BERTopic exists.
    """
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["message"] = "TDA analysis is starting..."
        JOBS[job_id]["progress"] = 3
        JOBS[job_id]["last_update_time"] = time.time()

        def progress_callback(percent, message):
            JOBS[job_id]["progress"] = int(percent)
            JOBS[job_id]["message"] = message
            JOBS[job_id]["last_update_time"] = time.time()

        result = run_tda_only(dataset_id, progress_callback=progress_callback)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["error"] = None
        JOBS[job_id]["outputs"] = result.get("outputs", {})
        JOBS[job_id]["dataset_id"] = result.get("dataset_id")
        JOBS[job_id]["message"] = "TDA analysis completed successfully."
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["last_update_time"] = time.time()

    except Exception:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = traceback.format_exc()
        JOBS[job_id]["outputs"] = {}
        JOBS[job_id]["dataset_id"] = dataset_id
        JOBS[job_id]["message"] = "TDA analysis failed."
        JOBS[job_id]["last_update_time"] = time.time()


@app.post("/run")
def run_pipeline():
    files = request.files.getlist("bibfile")

    if not files or all(f.filename == "" for f in files):
        return "No files selected", 400

    if len(files) > MAX_FILES:
        return f"You can upload at most {MAX_FILES} .bib files", 400

    saved_paths = []

    for f in files:
        if f.filename == "":
            continue

        if not allowed_file(f.filename):
            return f"Invalid file type: {f.filename}. Please upload only .bib files", 400

        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
        f.save(save_path)
        saved_paths.append(save_path)

    if not saved_paths:
        return "No valid .bib files uploaded", 400

    try:
        entries = load_multiple_bibs(saved_paths)
        total_entries = len(entries)

        print(f"📊 Total BibTeX entries: {total_entries}")

        if total_entries == 0:
            return "No BibTeX entries found in uploaded files.", 400

        if total_entries > MAX_TOTAL_ENTRIES:
            return (
                f"Too many papers ({total_entries}). "
                f"Limit is {MAX_TOTAL_ENTRIES}. "
                f"Please upload fewer files or split your dataset.",
                400
            )

    except Exception as e:
        return f"Failed to read BibTeX files: {e}", 400

    job_id = str(uuid.uuid4())
    estimated_seconds = estimate_pipeline_seconds(total_entries)

    JOBS[job_id] = {
        "status": "queued",
        "error": None,
        "outputs": {},
        "dataset_id": None,
        "message": f"Queued BERTopic analysis. {len(saved_paths)} file(s), {total_entries} total entries.",
        "total_entries": total_entries,
        "start_time": time.time(),
        "estimated_seconds": estimated_seconds,
        "progress": 3,
        "last_update_time": time.time(),
        "job_type": "bertopic"
    }

    t = threading.Thread(target=_run_job, args=(job_id, saved_paths), daemon=True)
    t.start()

    return redirect(url_for("status", job_id=job_id))


@app.post("/run-tda/<dataset_id>")
def run_tda(dataset_id):
    """
    Triggered by the Run TDA button.
    """
    if not is_valid_dataset_id(dataset_id):
        return "Dataset not found", 404

    if not get_bertopic_state(dataset_id):
        return "BERTopic output not found. Please run BERTopic first.", 400

    job_id = str(uuid.uuid4())
    estimated_seconds = estimate_tda_seconds(dataset_id)

    JOBS[job_id] = {
        "status": "queued",
        "error": None,
        "outputs": {},
        "dataset_id": dataset_id,
        "message": "Queued TDA Mapper analysis.",
        "total_entries": 0,
        "start_time": time.time(),
        "estimated_seconds": estimated_seconds,
        "progress": 3,
        "last_update_time": time.time(),
        "job_type": "tda"
    }

    t = threading.Thread(target=_run_tda_job, args=(job_id, dataset_id), daemon=True)
    t.start()

    return redirect(url_for("status", job_id=job_id))


@app.get("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)

    if not job:
        return "Job not found", 404

    if job["status"] in {"queued", "running"}:
        elapsed_seconds = int(time.time() - job.get("start_time", time.time()))
        estimated_seconds = int(job.get("estimated_seconds", 120))

        # Keep below 95% until job is truly done.
        progress = int(job.get("progress", 8))
        progress = max(3, min(95, progress))

        remaining_seconds = max(0, estimated_seconds - elapsed_seconds)

        return render_template(
            "status.html",
            job_id=job_id,
            status=job["status"],
            message=job.get("message", ""),
            total_entries=job.get("total_entries", 0),
            progress=progress,
            elapsed_seconds=elapsed_seconds,
            remaining_seconds=remaining_seconds,
            estimated_seconds=estimated_seconds
        )

    if job["status"] == "error":
        return render_template(
            "status.html",
            job_id=job_id,
            status="error",
            error=job["error"],
            message=job.get("message", "")
        )

    dataset_id = job.get("dataset_id")

    if dataset_id:
        return redirect(url_for("results_page", dataset_id=dataset_id))

    return "Dataset ID not found after job completed.", 500


# ============================================================
# DASHBOARD + SEPARATE SAME-TAB PAGES
# ============================================================

@app.get("/results/<dataset_id>")
def results_page(dataset_id):
    """
    Dashboard page only.
    This page should navigate to BERTopic, Mapper+Summary, and Persistence pages.
    """
    if not is_valid_dataset_id(dataset_id):
        return "Dataset not found", 404

    bertopic_done = get_bertopic_state(dataset_id)
    tda_done, node_summaries, global_summary = get_tda_state(dataset_id)
    outputs = get_dashboard_outputs(dataset_id)

    return render_template(
        "results.html",
        dataset_id=dataset_id,
        outputs=outputs,
        bertopic_done=bertopic_done,
        tda_done=tda_done,
        node_summaries=node_summaries,
        global_summary=global_summary
    )


@app.get("/bertopic/<dataset_id>")
def bertopic_page(dataset_id):
    """
    BERTopic visualizations page.
    Same tab navigation.
    """
    if not is_valid_dataset_id(dataset_id):
        return "Dataset not found", 404

    paths = get_dataset_paths(dataset_id)

    bertopic_outputs = {}

    if os.path.exists(paths["barchart_html"]):
        bertopic_outputs["BERTopic - All Topics Barchart"] = paths["barchart_html"]

    if os.path.exists(paths["topics_map_html"]):
        bertopic_outputs["BERTopic - Topics Map"] = paths["topics_map_html"]

    topic_percentage_data = get_topic_percentage_data(dataset_id)
    topic_timeline_data = get_topic_timeline_data(dataset_id)

    return render_template(
        "bertopic.html",
        dataset_id=dataset_id,
        outputs=bertopic_outputs,
        topic_percentage_data=topic_percentage_data,
        topic_timeline_data=topic_timeline_data
    )


@app.get("/tda-mapper/<dataset_id>")
def tda_mapper_page(dataset_id):
    """
    TDA Mapper graph + TDA summary page.
    Same tab navigation.
    """
    if not is_valid_dataset_id(dataset_id):
        return "Dataset not found", 404

    paths = get_dataset_paths(dataset_id)
    tda_done, node_summaries, global_summary = get_tda_state(dataset_id)

    if not tda_done:
        return redirect(url_for("results_page", dataset_id=dataset_id))

    return render_template(
        "tda_mapper.html",
        dataset_id=dataset_id,
        mapper_path=paths["mapper_html"],
        node_summaries=node_summaries,
        global_summary=global_summary
    )


@app.get("/persistence/<dataset_id>")
def persistence_page(dataset_id):
    """
    Persistence diagram + barcode page.
    Same tab navigation.
    """
    if not is_valid_dataset_id(dataset_id):
        return "Dataset not found", 404

    paths = get_dataset_paths(dataset_id)

    persistence_outputs = {}

    if os.path.exists(paths["persistence_diagram_png"]):
        persistence_outputs["Persistence Diagram"] = paths["persistence_diagram_png"]

    if os.path.exists(paths["persistence_barcode_png"]):
        persistence_outputs["Persistence Barcode"] = paths["persistence_barcode_png"]

    if not persistence_outputs:
        return redirect(url_for("results_page", dataset_id=dataset_id))

    return render_template(
        "persistence.html",
        dataset_id=dataset_id,
        outputs=persistence_outputs
    )


# ============================================================
# SAFE FILE VIEWER
# ============================================================

@app.get("/view")
def view_file():
    path = request.args.get("path", "").strip()

    if not path:
        return "Missing path parameter", 400

    requested_path = os.path.abspath(path)
    allowed_base = os.path.abspath(PROCESSED_DIR)

    if os.path.commonpath([requested_path, allowed_base]) != allowed_base:
        return "Not allowed", 403

    if not os.path.exists(requested_path):
        return "File not found", 404

    ext = os.path.splitext(requested_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]:
        return send_file(requested_path)

    if ext == ".html":
        with open(requested_path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".json":
        return send_file(requested_path, mimetype="application/json")

    return "Unsupported file type", 415


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)