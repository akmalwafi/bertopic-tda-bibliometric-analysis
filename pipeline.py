# pipeline.py
import hashlib
from pathlib import Path

from preprocess_bib import preprocess_bib
from sbert_embed import make_embeddings
from bertopic_run import run_bertopic
from run_mapper import main as run_mapper_main
from persistence_run import run_persistence
from gpt_topic_explainer import run_generatetopic
from apply_topic_labels import apply_topic_labels as run_labels


BASE_PROCESSED_DIR = Path("data/processed")


def get_file_hash(filepaths: list[str], chunk_size: int = 8192) -> str:
    hasher = hashlib.md5()

    for filepath in sorted(filepaths):
        path = Path(filepath)
        hasher.update(path.name.encode("utf-8"))

        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)

    return hasher.hexdigest()


def build_paths_from_dataset_id(dataset_id: str) -> dict:
    """
    Build the same dataset folder structure using an existing dataset_id.
    This is needed when user clicks Run TDA later.
    """
    root = BASE_PROCESSED_DIR / dataset_id
    embeddings_dir = root / "embeddings"
    bertopic_dir = root / "bertopic"
    tda_dir = root / "tda"

    root.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    bertopic_dir.mkdir(parents=True, exist_ok=True)
    tda_dir.mkdir(parents=True, exist_ok=True)

    return {
        "dataset_id": dataset_id,
        "root": root,

        # preprocess output
        "cleaned_csv": root / "cleaned_docs.csv",

        # embeddings
        "embeddings_dir": embeddings_dir,
        "embeddings_npy": embeddings_dir / "sbert_embeddings.npy",
        "embeddings_meta_csv": embeddings_dir / "meta.csv",

        # BERTopic
        "bertopic_dir": bertopic_dir,
        "topic_info_csv": bertopic_dir / "topic_info.csv",
        "doc_topics_csv": bertopic_dir / "doc_topics.csv",
        "topic_labels_csv": bertopic_dir / "topic_labels.csv",
        "bertopic_model_dir": bertopic_dir / "bertopic_model",
        "barchart_html": bertopic_dir / "barchart_all_topics.html",
        "topics_map_html": bertopic_dir / "topics_map.html",
        "topics_keywords_txt": bertopic_dir / "topics_keywords.txt",
        "labels_applied_marker": bertopic_dir / ".labels_applied",

        # TDA
        "tda_dir": tda_dir,
        "mapper_html": tda_dir / "mapper_graph.html",
        "mapper_node_summaries_json": tda_dir / "mapper_node_summaries.json",
        "mapper_global_summary_json": tda_dir / "mapper_global_summary.json",
        "mapper_meta_csv": tda_dir / "mapper_meta.csv",

        # persistence
        "persistence_diagram_png": tda_dir / "persistence_diagram.png",
        "persistence_barcode_png": tda_dir / "persistence_barcode.png",
        "persistence_dgms_npy": tda_dir / "persistence_dgms.npy",
    }


def create_dataset_structure(input_bib_paths: list[str]) -> dict:
    dataset_id = get_file_hash(input_bib_paths)
    return build_paths_from_dataset_id(dataset_id)


def run_all(input_bib_paths: list[str], progress_callback=None) -> dict:
    """
    IMPORTANT:
    This now runs BERTopic stage only.

    Flow:
    1. Preprocess BibTeX
    2. Generate SBERT embeddings
    3. Run BERTopic
    4. Generate readable topic labels
    5. Apply topic labels

    TDA Mapper and persistence are no longer run here.
    They will run later when user clicks the Run TDA button.
    """
    paths = create_dataset_structure(input_bib_paths)

    def update_progress(percent, message):
        if progress_callback:
            progress_callback(percent, message)
        print(f"📊 {percent}% - {message}")

    update_progress(5, "Preparing dataset structure...")

    print(f"📁 Dataset ID: {paths['dataset_id']}")
    print(f"📁 Output root: {paths['root']}")

    # 1) preprocess
    update_progress(10, "Preprocessing BibTeX records...")

    if not paths["cleaned_csv"].exists():
        preprocess_bib(
            input_bib_paths=input_bib_paths,
            output_csv=str(paths["cleaned_csv"]),
        )
    else:
        print("✅ Skip preprocess (cached)")

    update_progress(25, "Preprocessing completed.")

    # 2) embeddings
    update_progress(30, "Generating SBERT embeddings...")

    if not paths["embeddings_npy"].exists():
        make_embeddings(
            input_csv=str(paths["cleaned_csv"]),
            out_dir=str(paths["embeddings_dir"]),
        )
    else:
        print("✅ Skip embeddings (cached)")

    update_progress(50, "SBERT embeddings completed.")

    # 3) BERTopic
    update_progress(55, "Checking BERTopic topic modeling outputs...")

    bertopic_ready = (
        paths["topic_info_csv"].exists()
        and paths["doc_topics_csv"].exists()
        and paths["bertopic_model_dir"].exists()
    )

    update_progress(60, "Running BERTopic topic modeling...")

    if not bertopic_ready:
        run_bertopic(
            cleaned_csv=str(paths["cleaned_csv"]),
            embeddings_path=str(paths["embeddings_npy"]),
            out_dir=str(paths["bertopic_dir"]),
        )

        # New BERTopic output means labels should be reapplied
        if paths["labels_applied_marker"].exists():
            paths["labels_applied_marker"].unlink()
    else:
        print("✅ Skip BERTopic (cached)")

    update_progress(75, "BERTopic topic modeling completed.")

    # 4) topic label generation
    update_progress(80, "Generating readable topic labels...")

    if not paths["topic_labels_csv"].exists():
        run_generatetopic(
            input_file=str(paths["topic_info_csv"]),
            output_file=str(paths["topic_labels_csv"]),
        )

        if paths["labels_applied_marker"].exists():
            paths["labels_applied_marker"].unlink()
    else:
        print("✅ Skip topic label generation (cached)")

    update_progress(88, "Topic label generation completed.")

    # 5) apply labels at least once per dataset
    update_progress(92, "Applying topic labels to BERTopic model...")

    if not paths["labels_applied_marker"].exists():
        run_labels(
            model_dir=str(paths["bertopic_model_dir"]),
            labels_csv=str(paths["topic_labels_csv"]),
            out_dir=str(paths["bertopic_dir"]),
        )
        paths["labels_applied_marker"].write_text("ok", encoding="utf-8")
        print("✅ Topic labels applied")
    else:
        print("✅ Skip apply labels (already applied)")

    update_progress(98, "Preparing BERTopic result dashboard...")
    update_progress(100, "BERTopic analysis completed successfully.")

    return {
        "dataset_id": paths["dataset_id"],
        "root": str(paths["root"]),
        "tda_done": False,
        "outputs": {
            "BERTopic - All Topics Barchart": str(paths["barchart_html"]),
            "BERTopic - Topics Map": str(paths["topics_map_html"]),
        },
    }


def run_tda_only(dataset_id: str, progress_callback=None) -> dict:
    """
    Run TDA Mapper and persistence only after BERTopic already exists.
    This function will be called later from app.py when user clicks Run TDA.
    """
    paths = build_paths_from_dataset_id(dataset_id)

    def update_progress(percent, message):
        if progress_callback:
            progress_callback(percent, message)
        print(f"📊 {percent}% - {message}")

    update_progress(5, "Preparing TDA analysis...")

    required_files = [
        paths["cleaned_csv"],
        paths["embeddings_npy"],
        paths["topic_info_csv"],
        paths["doc_topics_csv"],
        paths["topic_labels_csv"],
    ]

    missing_files = [str(p) for p in required_files if not p.exists()]

    if missing_files:
        raise FileNotFoundError(
            "BERTopic must be completed before running TDA. Missing files:\n"
            + "\n".join(missing_files)
        )

    # 1) mapper + summaries
    update_progress(20, "Checking TDA Mapper graph outputs...")

    mapper_ready = (
        paths["mapper_html"].exists()
        and paths["mapper_node_summaries_json"].exists()
        and paths["mapper_global_summary_json"].exists()
    )

    update_progress(35, "Building TDA Mapper graph and summaries...")

    if not mapper_ready:
        run_mapper_main(
            doc_topics_csv=str(paths["doc_topics_csv"]),
            embeddings_path=str(paths["embeddings_npy"]),
            out_dir=str(paths["tda_dir"]),
            topic_info_csv=str(paths["topic_info_csv"]),
            cleaned_docs_csv=str(paths["cleaned_csv"]),
            topic_labels_csv=str(paths["topic_labels_csv"]),
        )
    else:
        print("✅ Skip Mapper (cached)")

    update_progress(70, "TDA Mapper graph completed.")

    # 2) persistence
    update_progress(75, "Checking persistence diagram outputs...")

    persistence_ready = (
        paths["persistence_diagram_png"].exists()
        and paths["persistence_barcode_png"].exists()
        and paths["persistence_dgms_npy"].exists()
    )

    update_progress(85, "Generating persistence diagram and barcode...")

    if not persistence_ready:
        run_persistence(
            embeddings_path=str(paths["embeddings_npy"]),
            out_dir=str(paths["tda_dir"]),
        )
    else:
        print("✅ Skip persistence (cached)")

    update_progress(98, "Preparing TDA result dashboard...")
    update_progress(100, "TDA analysis completed successfully.")

    return {
        "dataset_id": paths["dataset_id"],
        "root": str(paths["root"]),
        "tda_done": True,
        "outputs": {
            "BERTopic - All Topics Barchart": str(paths["barchart_html"]),
            "BERTopic - Topics Map": str(paths["topics_map_html"]),
            "TDA Mapper Graph": str(paths["mapper_html"]),
            "TDA - Persistence Diagram": str(paths["persistence_diagram_png"]),
            "TDA - Persistence Barcode": str(paths["persistence_barcode_png"]),
        },
    }