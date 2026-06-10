# run_mapper.py
import os
import json
import ast
import re
import html
import numpy as np
import pandas as pd
import kmapper as km

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_read_csv(path):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def parse_representation(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    parts = [x.strip().strip("'").strip('"') for x in text.split(",")]
    return [p for p in parts if p]


def normalize_label_text(text):
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"^\d+[_\-\s]*", "", text)
    text = text.replace("_", " ")
    return text.strip()


def get_column(df, candidates):
    if df is None or df.empty:
        return None

    for col in candidates:
        if col in df.columns:
            return col

    lower = {c.lower(): c for c in df.columns}

    for col in candidates:
        if col.lower() in lower:
            return lower[col.lower()]

    return None


def safe_node_id(node_id):
    return (
        str(node_id)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


# ============================================================
# CLUSTER NAME MAP
# ============================================================

def build_cluster_name_map(graph):
    cluster_name_map = {}
    reverse_cluster_name_map = {}

    for idx, node_id in enumerate(graph["nodes"].keys(), start=1):
        cluster_name = f"Cluster {idx}"
        cluster_name_map[node_id] = cluster_name
        reverse_cluster_name_map[cluster_name] = node_id

    return cluster_name_map, reverse_cluster_name_map


# ============================================================
# TOPIC LABEL HELPERS
# ============================================================

def build_topic_label_map(topic_labels_df):
    if topic_labels_df.empty:
        return {}

    topic_col = get_column(topic_labels_df, ["Topic", "topic"])
    label_col = get_column(
        topic_labels_df,
        [
            "Label",
            "label",
            "Generated_Label",
            "generated_label",
            "Topic_Label",
            "topic_label",
            "CustomName",
            "Name",
        ],
    )

    if topic_col is None or label_col is None:
        return {}

    label_map = {}

    for _, row in topic_labels_df.iterrows():
        try:
            topic_id = int(row[topic_col])
            label_text = normalize_label_text(row[label_col])
            if label_text:
                label_map[topic_id] = label_text
        except Exception:
            continue

    return label_map


def get_topic_label(topic_id, topic_info_df, topic_label_map=None):
    topic_id = int(topic_id)

    if topic_id == -1:
        return "Outlier / Mixed Topics"

    if topic_label_map and topic_id in topic_label_map and topic_label_map[topic_id]:
        return topic_label_map[topic_id]

    if topic_info_df.empty:
        return f"Topic {topic_id}"

    if "Topic" not in topic_info_df.columns:
        return f"Topic {topic_id}"

    row = topic_info_df[topic_info_df["Topic"] == topic_id]

    if row.empty:
        return f"Topic {topic_id}"

    for col in ["CustomName", "Name"]:
        if col in row.columns:
            value = normalize_label_text(row.iloc[0][col])
            if value and value.lower() != "nan":
                return value

    if "Representation" in row.columns:
        kws = parse_representation(row.iloc[0]["Representation"])
        if kws:
            return ", ".join(kws[:3])

    return f"Topic {topic_id}"


def get_topic_keywords(topic_id, topic_info_df):
    topic_id = int(topic_id)

    if topic_id == -1:
        return ["outlier", "mixed", "miscellaneous"]

    if topic_info_df.empty:
        return []

    if "Topic" not in topic_info_df.columns:
        return []

    row = topic_info_df[topic_info_df["Topic"] == topic_id]

    if row.empty:
        return []

    if "Representation" in row.columns:
        return parse_representation(row.iloc[0]["Representation"])[:5]

    return []


# ============================================================
# GRAPH STRUCTURE HELPERS
# ============================================================

def build_degree_map(graph):
    degree_map = {node_id: 0 for node_id in graph["nodes"].keys()}
    seen_edges = set()

    for src, dsts in graph["links"].items():
        for dst in dsts:
            edge = tuple(sorted((src, dst)))

            if edge not in seen_edges:
                seen_edges.add(edge)
                degree_map[src] += 1
                degree_map[dst] += 1

    return degree_map


def build_overlap_map(graph):
    node_members = {
        node_id: set(map(int, doc_indices))
        for node_id, doc_indices in graph["nodes"].items()
    }

    overlap_map = {node_id: [] for node_id in graph["nodes"].keys()}
    node_ids = list(node_members.keys())

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            a = node_ids[i]
            b = node_ids[j]

            if node_members[a] & node_members[b]:
                overlap_map[a].append(b)
                overlap_map[b].append(a)

    return overlap_map


def build_shared_document_details(graph, cleaned_docs_df, cluster_name_map=None):
    """
    Document-level sharing summary.

    For each Mapper node/cluster, find the exact documents inside that node
    that also appear in another Mapper node.

    This avoids saying the whole cluster overlaps when only some documents overlap.
    """
    title_col = get_column(cleaned_docs_df, ["title", "Title"])
    year_col = get_column(cleaned_docs_df, ["year", "Year"])

    doc_to_nodes = {}

    for node_id, doc_indices in graph["nodes"].items():
        for idx in doc_indices:
            doc_to_nodes.setdefault(int(idx), []).append(node_id)

    shared_map = {node_id: [] for node_id in graph["nodes"].keys()}

    for doc_idx, node_ids in doc_to_nodes.items():
        if len(node_ids) <= 1:
            continue

        title_text = f"Document {doc_idx}"
        year_text = "N/A"

        if title_col is not None and 0 <= doc_idx < len(cleaned_docs_df):
            value = cleaned_docs_df.iloc[doc_idx][title_col]
            if pd.notna(value):
                title_text = str(value)

        if year_col is not None and 0 <= doc_idx < len(cleaned_docs_df):
            value = cleaned_docs_df.iloc[doc_idx][year_col]
            if pd.notna(value):
                try:
                    year_text = str(int(float(value)))
                except Exception:
                    year_text = str(value)

        for current_node in node_ids:
            other_nodes = [n for n in node_ids if n != current_node]

            for other_node in other_nodes:
                other_cluster_name = (
                    cluster_name_map.get(other_node, other_node)
                    if cluster_name_map
                    else other_node
                )

                shared_map[current_node].append(
                    {
                        "doc_index": doc_idx,
                        "title": title_text,
                        "year": year_text,
                        "shared_with_node_id": other_node,
                        "shared_with_safe_node_id": safe_node_id(other_node),
                        "shared_with_cluster_name": other_cluster_name,
                    }
                )

    return shared_map


def infer_structure(cluster_size, degree):
    if degree == 0:
        return "isolated"

    if degree <= 2 and cluster_size < 10:
        return "weakly connected niche cluster"

    if degree <= 2:
        return "weakly connected cluster"

    if cluster_size >= 20 and degree >= 3:
        return "dense dominant cluster"

    return "moderately connected cluster"


def structure_meaning(structure):
    mapping = {
        "isolated": "This suggests an outlier or niche area with very limited relation to other topics.",
        "weakly connected niche cluster": "This suggests a small specialized topic with limited overlap with the rest of the dataset.",
        "weakly connected cluster": "This suggests partial overlap with nearby themes, but the topic remains fairly distinct.",
        "dense dominant cluster": "This suggests a well-established and cohesive research area with strong semantic similarity.",
        "moderately connected cluster": "This suggests a meaningful topic with some overlap or transition to neighboring themes.",
    }

    return mapping.get(
        structure,
        "This reflects the structural role of the cluster in the Mapper graph.",
    )


def summarize_years(year_values):
    valid = []

    for y in year_values:
        try:
            if pd.notna(y):
                valid.append(int(float(y)))
        except Exception:
            continue

    if not valid:
        return [], "", None

    counts = pd.Series(valid).value_counts().sort_index()
    year_list = counts.index.tolist()

    if len(year_list) == 1:
        year_text = str(year_list[0])
    else:
        year_text = f"{min(year_list)} to {max(year_list)}"

    top_year = int(counts.idxmax())

    return year_list, year_text, top_year


def user_friendly_structure_summary(topic_name, structure, cluster_size, degree, year_text=""):
    year_part = f" The papers mainly come from {year_text}." if year_text else ""

    if topic_name == "Outlier / Mixed Topics":
        return (
            f"This cluster represents outlier or mixed documents. It does not strongly match one single topic, "
            f"so it should be interpreted as a miscellaneous or less-cohesive cluster. "
            f"It contains {cluster_size} document(s) and has {degree} connection(s).{year_part}"
        )

    return (
        f"This cluster represents {topic_name}. It is a {structure}. "
        f"It contains {cluster_size} document(s) and has {degree} connection(s).{year_part}"
    )


# ============================================================
# DOCUMENT TOOLTIP
# FIXED:
# - No repeated button inside every document/member.
# - The button is inserted only once at the top by patch_mapper_html().
# ============================================================

def build_document_tooltips(
    doc_topics_df,
    topic_info_df,
    cleaned_docs_df,
    graph,
    topic_label_map=None,
    cluster_name_map=None,
):
    n_docs = max(len(doc_topics_df), len(cleaned_docs_df))
    tooltips = []

    topic_col = get_column(doc_topics_df, ["topic", "Topic"])
    title_col = get_column(cleaned_docs_df, ["title", "Title"])
    year_col = get_column(cleaned_docs_df, ["year", "Year"])

    for i in range(n_docs):
        topic_name = "Unknown Topic"
        year_text = "N/A"
        title_text = f"Document {i}"
        topic_id = None

        if topic_col is not None and i < len(doc_topics_df):
            try:
                topic_id = int(doc_topics_df.iloc[i][topic_col])
                topic_name = get_topic_label(
                    topic_id,
                    topic_info_df,
                    topic_label_map=topic_label_map,
                )
            except Exception:
                pass

        if title_col is not None and i < len(cleaned_docs_df):
            value = cleaned_docs_df.iloc[i][title_col]
            if pd.notna(value):
                title_text = str(value)

        if year_col is not None and i < len(cleaned_docs_df):
            value = cleaned_docs_df.iloc[i][year_col]
            if pd.notna(value):
                try:
                    year_text = str(int(float(value)))
                except Exception:
                    year_text = str(value)

        topic_id_text = topic_id if topic_id is not None else "N/A"

        tooltip = "<br>".join(
            [
                f"<b>Topic:</b> {html.escape(str(topic_name))}",
                f"<b>Topic ID:</b> {html.escape(str(topic_id_text))}",
                f"<b>Year:</b> {html.escape(str(year_text))}",
                f"<b>Title:</b> {html.escape(str(title_text))}",
            ]
        )

        tooltips.append(tooltip)

    return np.array(tooltips, dtype=object)


# ============================================================
# NODE SUMMARIES FOR results.html / tda_mapper.html
# ============================================================

def build_node_summaries(
    graph,
    doc_topics_df,
    topic_info_df,
    cleaned_docs_df,
    topic_label_map=None,
    cluster_name_map=None,
):
    degree_map = build_degree_map(graph)

    shared_document_map = build_shared_document_details(
        graph=graph,
        cleaned_docs_df=cleaned_docs_df,
        cluster_name_map=cluster_name_map,
    )

    node_summaries = []

    topic_col = get_column(doc_topics_df, ["topic", "Topic"])
    title_col = get_column(cleaned_docs_df, ["title", "Title"])
    year_col = get_column(cleaned_docs_df, ["year", "Year"])

    for node_id, doc_indices in graph["nodes"].items():
        doc_indices = list(map(int, doc_indices))

        cluster_size = len(doc_indices)
        degree = degree_map.get(node_id, 0)

        cluster_name = cluster_name_map.get(node_id, node_id) if cluster_name_map else node_id

        cluster_topics = []
        cluster_topic_details = []
        sample_titles = []
        cluster_papers = []
        year_values = []

        valid_topic_idx = []
        valid_doc_idx = []

        if not doc_topics_df.empty and topic_col is not None:
            valid_topic_idx = [i for i in doc_indices if 0 <= i < len(doc_topics_df)]
            if valid_topic_idx:
                cluster_topics = doc_topics_df.iloc[valid_topic_idx][topic_col].tolist()

        if not cleaned_docs_df.empty:
            valid_doc_idx = [i for i in doc_indices if 0 <= i < len(cleaned_docs_df)]

            if valid_doc_idx and title_col is not None:
                all_titles = (
                    cleaned_docs_df.iloc[valid_doc_idx][title_col]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                sample_titles = all_titles[:5]

            if valid_doc_idx and year_col is not None:
                year_values = cleaned_docs_df.iloc[valid_doc_idx][year_col].tolist()

            for doc_idx in valid_doc_idx:
                title_text = f"Document {doc_idx}"
                year_text = "N/A"
                paper_topic_id = None
                paper_topic_name = "Unknown Topic"

                if title_col is not None:
                    value = cleaned_docs_df.iloc[doc_idx][title_col]
                    if pd.notna(value) and str(value).strip():
                        title_text = str(value).strip()

                if year_col is not None:
                    value = cleaned_docs_df.iloc[doc_idx][year_col]
                    if pd.notna(value):
                        try:
                            year_text = str(int(float(value)))
                        except Exception:
                            year_text = str(value)

                if (
                    not doc_topics_df.empty
                    and topic_col is not None
                    and 0 <= doc_idx < len(doc_topics_df)
                ):
                    try:
                        paper_topic_id = int(doc_topics_df.iloc[doc_idx][topic_col])
                        paper_topic_name = get_topic_label(
                            paper_topic_id,
                            topic_info_df,
                            topic_label_map=topic_label_map,
                        )
                    except Exception:
                        paper_topic_id = None
                        paper_topic_name = "Unknown Topic"

                cluster_papers.append(
                    {
                        "doc_index": doc_idx,
                        "title": title_text,
                        "year": year_text,
                        "topic_id": paper_topic_id,
                        "topic_name": paper_topic_name,
                    }
                )

        if cluster_topics:
            topic_counts = pd.Series(cluster_topics).value_counts()
            non_outlier = topic_counts[topic_counts.index != -1]

            if not non_outlier.empty:
                main_topic = int(non_outlier.index[0])
            else:
                main_topic = int(topic_counts.index[0])

            topic_name = get_topic_label(main_topic, topic_info_df, topic_label_map=topic_label_map)
            keywords = get_topic_keywords(main_topic, topic_info_df)

            for topic_id, count in topic_counts.items():
                try:
                    topic_id_int = int(topic_id)
                except Exception:
                    continue

                cluster_topic_details.append(
                    {
                        "topic_id": topic_id_int,
                        "topic_name": get_topic_label(
                            topic_id_int,
                            topic_info_df,
                            topic_label_map=topic_label_map,
                        ),
                        "keywords": get_topic_keywords(topic_id_int, topic_info_df),
                        "document_count": int(count),
                    }
                )
        else:
            main_topic = -1
            topic_name = "Outlier / Mixed Topics"
            keywords = ["outlier", "mixed", "miscellaneous"]
            cluster_topic_details = [
                {
                    "topic_id": -1,
                    "topic_name": "Outlier / Mixed Topics",
                    "keywords": ["outlier", "mixed", "miscellaneous"],
                    "document_count": cluster_size,
                }
            ]

        year_list, year_text, dominant_year = summarize_years(year_values)
        structure = infer_structure(cluster_size, degree)

        if main_topic == -1:
            content_summary = (
                "This cluster is associated with outlier or mixed documents that do not strongly "
                "belong to a single coherent BERTopic topic."
            )
        elif keywords:
            keyword_text = ", ".join(keywords[:3])
            content_summary = (
                f"This cluster is mainly associated with {topic_name}, "
                f"characterized by keywords such as {keyword_text}."
            )
        else:
            content_summary = f"This cluster is mainly associated with {topic_name}."

        if year_text:
            content_summary += f" The documents mainly span {year_text}."

        structural_summary = (
            f"Structurally, this cluster forms a {structure} with {cluster_size} document(s) "
            f"and degree {degree}. {structure_meaning(structure)}"
        )

        user_summary = user_friendly_structure_summary(
            topic_name=topic_name,
            structure=structure,
            cluster_size=cluster_size,
            degree=degree,
            year_text=year_text,
        )

        shared_documents = shared_document_map.get(node_id, [])

        unique_shared_doc_ids = sorted(
            set(item.get("doc_index") for item in shared_documents if item.get("doc_index") is not None)
        )

        node_summaries.append(
            {
                "node_id": node_id,
                "safe_node_id": safe_node_id(node_id),
                "cluster_name": cluster_name,
                "cluster_size": cluster_size,
                "degree": degree,
                "main_topic": main_topic,
                "topic_name": topic_name,
                "keywords": keywords,
                "sample_titles": sample_titles,

                "cluster_topics": cluster_topic_details,
                "cluster_papers": cluster_papers,
                "cluster_paper_count": len(cluster_papers),

                "years": year_list,
                "year_text": year_text,
                "dominant_year": dominant_year,
                "structure_type": structure,
                "content_summary": content_summary,
                "structural_summary": structural_summary,
                "user_summary": user_summary,
                "full_summary": f"{content_summary} {structural_summary}",

                "shared_documents": shared_documents,
                "shared_document_count": len(unique_shared_doc_ids),

                "overlapped_nodes": [],
            }
        )

    return node_summaries


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def build_global_summary(graph, node_summaries):
    total_nodes = len(graph["nodes"])

    seen_edges = set()

    for src, dsts in graph["links"].items():
        for dst in dsts:
            seen_edges.add(tuple(sorted((src, dst))))

    total_edges = len(seen_edges)
    isolated_count = sum(1 for n in node_summaries if n["degree"] == 0)
    largest_cluster = max((n["cluster_size"] for n in node_summaries), default=0)

    disconnected_ratio = (isolated_count / total_nodes) if total_nodes else 0

    if disconnected_ratio > 0.3:
        connectivity_text = "largely fragmented"
        user_text = "The graph is highly fragmented, meaning the dataset is divided into many separate topic regions."
    elif disconnected_ratio > 0.1:
        connectivity_text = "moderately fragmented"
        user_text = "The graph has several separate groups, suggesting the dataset contains multiple distinct themes."
    else:
        connectivity_text = "relatively connected"
        user_text = "The graph is fairly connected, suggesting many topics are related or transition into one another."

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "isolated_nodes": isolated_count,
        "largest_cluster_size": largest_cluster,
        "connectivity_type": connectivity_text,
        "summary_text": (
            f"The Mapper graph contains {total_nodes} clusters and {total_edges} connections. "
            f"It is {connectivity_text}, with {isolated_count} isolated clusters. "
            f"The largest cluster contains {largest_cluster} document(s), indicating a dominant topic region."
        ),
        "user_summary": (
            f"This TDA graph contains {total_nodes} cluster(s). {user_text} "
            f"There are {isolated_count} isolated cluster(s), and the largest cluster contains "
            f"{largest_cluster} document(s), which may represent one of the main topic areas."
        ),
    }


# ============================================================
# SAFE PATCH MAPPER HTML
# FIXED:
# - One "Current Cluster" badge only.
# - One "View Cluster Summary" button only.
# - No repeated button under every document/member.
# ============================================================

def patch_mapper_html(html_path, cluster_name_map):
    if not os.path.exists(html_path):
        return

    cluster_map_json = json.dumps(cluster_name_map, ensure_ascii=False)

    patch_code = f"""
<script>
window.CLUSTER_NAME_MAP = {cluster_map_json};
window.CURRENT_MAPPER_NODE_ID = null;

function getNodeIdFromData(d) {{
    if (!d) return null;
    return d.name || d.node_id || d.id || (d.tooltip ? d.tooltip.node_id : null);
}}

function removeOldClusterControls() {{
    document.querySelectorAll(".current-cluster-badge-wrap").forEach(function(el) {{
        el.remove();
    }});

    document.querySelectorAll(".current-cluster-badge").forEach(function(el) {{
        el.remove();
    }});
}}

function updateCurrentClusterBadge(currentRawId) {{
    if (!currentRawId) return;

    window.CURRENT_MAPPER_NODE_ID = currentRawId;

    removeOldClusterControls();

    const clusterName = window.CLUSTER_NAME_MAP[currentRawId] || currentRawId;

    const divs = Array.from(document.querySelectorAll("div"));
    const topicDiv = divs.find(function(el) {{
        const text = (el.innerText || "").trim();
        return text.startsWith("Topic:");
    }});

    if (!topicDiv || !topicDiv.parentNode) return;

    const wrap = document.createElement("div");
    wrap.className = "current-cluster-badge-wrap";
    wrap.style.margin = "10px 0 14px 0";

    const badge = document.createElement("div");
    badge.className = "current-cluster-badge";
    badge.style.margin = "0 0 8px 0";
    badge.style.padding = "7px 12px";
    badge.style.borderRadius = "999px";
    badge.style.background = "#0ea5e9";
    badge.style.color = "white";
    badge.style.fontSize = "13px";
    badge.style.fontWeight = "700";
    badge.style.display = "inline-block";
    badge.style.boxShadow = "0 0 6px rgba(14,165,233,0.6)";
    badge.style.letterSpacing = "0.3px";
    badge.innerText = "Current Cluster: " + clusterName;

    const btn = document.createElement("button");
    btn.className = "mapper-node-btn";
    btn.setAttribute("data-node-id", currentRawId);
    btn.setAttribute("data-cluster-name", clusterName);

    btn.style.display = "block";
    btn.style.padding = "6px 10px";
    btn.style.border = "none";
    btn.style.borderRadius = "6px";
    btn.style.background = "#2563eb";
    btn.style.color = "white";
    btn.style.cursor = "pointer";
    btn.style.fontSize = "12px";
    btn.style.marginTop = "6px";

    btn.innerText = "View " + clusterName + " Summary";

    btn.onclick = function() {{
        window.parent.openNodeDetailsFromGraph(currentRawId);
    }};

    wrap.appendChild(badge);
    wrap.appendChild(btn);

    topicDiv.parentNode.insertBefore(wrap, topicDiv);
}}

function highlightMapperNode(nodeId) {{
    const mapperNodes = Array.from(document.querySelectorAll("path.circle"));

    mapperNodes.forEach(function(node) {{
        node.style.stroke = "";
        node.style.strokeWidth = "";
        node.style.filter = "";
    }});

    let targetNode = null;

    mapperNodes.forEach(function(node) {{
        const d = node.__data__;
        const currentId = getNodeIdFromData(d);
        if (String(currentId) === String(nodeId)) {{
            targetNode = node;
        }}
    }});

    if (!targetNode && typeof graph !== "undefined" && graph.nodes) {{
        const nodeIndex = graph.nodes.findIndex(function(n) {{
            return String(getNodeIdFromData(n)) === String(nodeId);
        }});

        if (nodeIndex >= 0 && mapperNodes[nodeIndex]) {{
            targetNode = mapperNodes[nodeIndex];
        }}
    }}

    if (!targetNode) return;

    targetNode.style.stroke = "#facc15";
    targetNode.style.strokeWidth = "10px";
    targetNode.style.filter = "drop-shadow(0px 0px 10px #facc15)";
}}

function syncCurrentNodeFromMapperClick(nodeElement) {{
    const d = nodeElement.__data__;
    const nodeId = getNodeIdFromData(d);

    if (!nodeId) return;

    window.CURRENT_MAPPER_NODE_ID = nodeId;

    setTimeout(function() {{
        updateCurrentClusterBadge(nodeId);
        highlightMapperNode(nodeId);
    }}, 80);
}}

function attachMapperNodeListeners() {{
    const mapperNodes = Array.from(document.querySelectorAll("path.circle"));

    mapperNodes.forEach(function(node) {{
        if (node.dataset.syncAttached === "1") return;

        node.dataset.syncAttached = "1";

        node.addEventListener("click", function() {{
            syncCurrentNodeFromMapperClick(node);
        }});

        node.addEventListener("mousedown", function() {{
            syncCurrentNodeFromMapperClick(node);
        }});
    }});
}}

window.focusMapperNodeFromSummary = function(nodeId) {{
    try {{
        if (!nodeId) return;

        window.CURRENT_MAPPER_NODE_ID = nodeId;

        highlightMapperNode(nodeId);
        updateCurrentClusterBadge(nodeId);

        const svg = document.querySelector("svg");
        if (svg) {{
            svg.scrollIntoView({{
                behavior: "smooth",
                block: "center",
                inline: "center"
            }});
        }}

        console.log("Highlighted mapper node:", nodeId);

    }} catch (e) {{
        console.error("focusMapperNodeFromSummary error:", e);
    }}
}};

document.addEventListener("DOMContentLoaded", function() {{
    attachMapperNodeListeners();

    const observer = new MutationObserver(function() {{
        attachMapperNodeListeners();
    }});

    observer.observe(document.body, {{
        childList: true,
        subtree: true
    }});
}});
</script>
"""

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(
        r"<script>\s*window\.CLUSTER_NAME_MAP[\s\S]*?</script>",
        "",
        content
    )

    if "</body>" in content:
        content = content.replace("</body>", patch_code + "\n</body>", 1)
    else:
        content += patch_code

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


# ============================================================
# MAIN
# ============================================================

def main(
    doc_topics_csv,
    embeddings_path,
    out_dir,
    topic_info_csv=None,
    cleaned_docs_csv=None,
    topic_labels_csv=None,
    n_cubes=10,
    perc_overlap=0.3,
    dbscan_eps=0.8,
    dbscan_min_samples=3,
):
    os.makedirs(out_dir, exist_ok=True)

    X = np.load(embeddings_path)

    doc_topics_df = safe_read_csv(doc_topics_csv)
    topic_info_df = safe_read_csv(topic_info_csv) if topic_info_csv else pd.DataFrame()
    cleaned_docs_df = safe_read_csv(cleaned_docs_csv) if cleaned_docs_csv else pd.DataFrame()
    topic_labels_df = safe_read_csv(topic_labels_csv) if topic_labels_csv else pd.DataFrame()

    topic_label_map = build_topic_label_map(topic_labels_df)

    mapper = km.KeplerMapper(verbose=1)

    lens = mapper.fit_transform(
        X,
        projection=PCA(n_components=2),
    )

    graph = mapper.map(
        lens,
        X,
        cover=km.Cover(
            n_cubes=n_cubes,
            perc_overlap=perc_overlap,
        ),
        clusterer=DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        ),
    )

    cluster_name_map, reverse_cluster_name_map = build_cluster_name_map(graph)

    node_summaries = build_node_summaries(
        graph=graph,
        doc_topics_df=doc_topics_df,
        topic_info_df=topic_info_df,
        cleaned_docs_df=cleaned_docs_df,
        topic_label_map=topic_label_map,
        cluster_name_map=cluster_name_map,
    )

    global_summary = build_global_summary(
        graph=graph,
        node_summaries=node_summaries,
    )

    document_tooltips = build_document_tooltips(
        doc_topics_df=doc_topics_df,
        topic_info_df=topic_info_df,
        cleaned_docs_df=cleaned_docs_df,
        graph=graph,
        topic_label_map=topic_label_map,
        cluster_name_map=cluster_name_map,
    )

    html_path = os.path.join(out_dir, "mapper_graph.html")

    mapper.visualize(
        graph,
        path_html=html_path,
        title="Mapper Graph of SBERT Embeddings",
        custom_tooltips=document_tooltips,
    )

    patch_mapper_html(
        html_path=html_path,
        cluster_name_map=cluster_name_map,
    )

    node_summary_path = os.path.join(out_dir, "mapper_node_summaries.json")
    global_summary_path = os.path.join(out_dir, "mapper_global_summary.json")
    cluster_map_path = os.path.join(out_dir, "mapper_cluster_name_map.json")

    with open(node_summary_path, "w", encoding="utf-8") as f:
        json.dump(node_summaries, f, indent=2, ensure_ascii=False)

    with open(global_summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    with open(cluster_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cluster_name_map": cluster_name_map,
                "reverse_cluster_name_map": reverse_cluster_name_map,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Mapper graph saved to:", html_path)
    print("Node summaries saved to:", node_summary_path)
    print("Global summary saved to:", global_summary_path)
    print("Cluster name map saved to:", cluster_map_path)

    return {
        "graph": graph,
        "html_path": html_path,
        "node_summary_path": node_summary_path,
        "global_summary_path": global_summary_path,
        "cluster_map_path": cluster_map_path,
    }


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    main(
        doc_topics_csv=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "bertopic",
            "doc_topics.csv",
        ),
        embeddings_path=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "embeddings",
            "sbert_embeddings.npy",
        ),
        out_dir=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "tda",
        ),
        topic_info_csv=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "bertopic",
            "topic_info.csv",
        ),
        cleaned_docs_csv=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "cleaned_docs.csv",
        ),
        topic_labels_csv=os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "bertopic",
            "topic_labels.csv",
        ),
    )