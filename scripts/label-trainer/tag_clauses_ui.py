import os
import re
import json
import gradio as gr

# CONFIG
DATA_DIR = "Selected Files/Civil Appeal"
OUTPUT_DIR = "Newly Annotated/Civil Appeal"
LABELS = ["Claim", "Premise", "Opposition", "None"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# UTILITIES
def natural_key(value: str):
    parts = re.split(r"(\d+)", value)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_clauses_from_file(filepath: str):
    """Reads metadata and body clauses (after '=== BODY ===')."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    body_index = None
    for i, line in enumerate(lines):
        if line.replace(" ", "").lower() == "===body===":
            body_index = i
            break

    if body_index is None:
        return [], lines

    metadata = lines[:body_index]
    body = lines[body_index + 1 :]
    return metadata, body


def _response(
    status,
    doc_id,
    metadata,
    clauses,
    annotations,
    idx,
    current_clause,
    current_label,
    split_clause_text="",
    split_label_choice="None",
):
    return (
        status,
        doc_id,
        metadata,
        clauses,
        annotations,
        idx,
        current_clause,
        current_label,
        split_clause_text,
        split_label_choice,
    )


def _store_current_clause(label, text, clauses, annotations, idx):
    if not clauses:
        return False, "No file loaded."

    clean_text = (text or "").strip()
    if not clean_text:
        return False, "❌ Clause text cannot be empty."

    clauses[idx] = clean_text
    annotations[idx]["text"] = clean_text
    annotations[idx]["label"] = label if label else "None"
    return True, clean_text


def save_annotations(doc_id, annotations, metadata):
    outpath = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    payload = {"doc_id": doc_id, "metadata": metadata, "clauses": []}
    for i, ann in enumerate(annotations):
        payload["clauses"].append(
            {
                "clause_id": i + 1,
                "text": ann["text"],
                "label": ann["label"],
                "prev_clause": annotations[i - 1]["text"] if i > 0 else None,
                "next_clause": (
                    annotations[i + 1]["text"] if i < len(annotations) - 1 else None
                ),
            }
        )

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return outpath


def _load_initial_state(selected_display_name):
    if not selected_display_name:
        return _response(
            "No file selected.",
            None,
            None,
            None,
            None,
            None,
            "",
            "None",
        )

    real_filename = file_display_map.get(selected_display_name, selected_display_name)
    doc_id = os.path.splitext(real_filename)[0]
    input_path = os.path.join(DATA_DIR, real_filename)
    annotated_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")

    if not os.path.exists(input_path):
        return _response(
            f"❌ File not found in {DATA_DIR}: {real_filename}",
            None,
            None,
            None,
            None,
            None,
            "",
            "None",
        )

    if os.path.exists(annotated_path):
        with open(annotated_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        clauses = [c["text"] for c in data.get("clauses", [])]
        annotations = [
            {"text": c["text"], "label": c.get("label", "None")}
            for c in data.get("clauses", [])
        ]
        metadata = data.get("metadata", [])
        status = f"✅ Resumed {real_filename} ({len(clauses)} clauses)"
    else:
        metadata, clauses = load_clauses_from_file(input_path)
        annotations = [{"text": c, "label": "None"} for c in clauses]
        status = f"Annotating {real_filename} ({len(clauses)} clauses)"

    if not clauses:
        return _response(
            "❌ No clauses detected in file.",
            doc_id,
            metadata,
            clauses,
            annotations,
            None,
            "",
            "None",
        )

    idx = 0
    current_clause = annotations[idx]["text"]
    current_label = annotations[idx]["label"] or "None"
    return _response(
        status,
        doc_id,
        metadata,
        clauses,
        annotations,
        idx,
        current_clause,
        current_label,
    )


# ACTION HANDLERS
def start_annotation(selected_file):
    return _load_initial_state(selected_file)


def annotate_next(label, current_text, doc_id, metadata, clauses, annotations, idx):
    if not clauses or idx is None:
        return _response(
            "No file loaded.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
        )

    ok, result = _store_current_clause(label, current_text, clauses, annotations, idx)
    if not ok:
        return _response(
            result, doc_id, metadata, clauses, annotations, idx, current_text, label
        )

    if idx + 1 >= len(clauses):
        outpath = save_annotations(doc_id, annotations, metadata)
        return _response(
            f"✅ Done! Saved annotations to {outpath}",
            None,
            None,
            None,
            None,
            None,
            "",
            "None",
        )

    new_idx = idx + 1
    current_clause = clauses[new_idx]
    current_label = annotations[new_idx]["label"] or "None"
    status = f"Clause {new_idx+1}/{len(clauses)}"
    return _response(
        status,
        doc_id,
        metadata,
        clauses,
        annotations,
        new_idx,
        current_clause,
        current_label,
    )


def annotate_prev(label, current_text, doc_id, metadata, clauses, annotations, idx):
    if not clauses or idx is None:
        return _response(
            "No file loaded.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
        )

    ok, result = _store_current_clause(label, current_text, clauses, annotations, idx)
    if not ok:
        return _response(
            result, doc_id, metadata, clauses, annotations, idx, current_text, label
        )

    new_idx = max(idx - 1, 0)
    current_clause = clauses[new_idx]
    current_label = annotations[new_idx]["label"] or "None"
    status = f"Clause {new_idx+1}/{len(clauses)}"
    return _response(
        status,
        doc_id,
        metadata,
        clauses,
        annotations,
        new_idx,
        current_clause,
        current_label,
    )


def update_clause(label, current_text, doc_id, metadata, clauses, annotations, idx):
    if not clauses or idx is None:
        return _response(
            "No file loaded.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
        )

    ok, result = _store_current_clause(label, current_text, clauses, annotations, idx)
    if not ok:
        return _response(
            result, doc_id, metadata, clauses, annotations, idx, current_text, label
        )

    status = f"✏️ Updated clause {idx+1}"
    return _response(
        status,
        doc_id,
        metadata,
        clauses,
        annotations,
        idx,
        clauses[idx],
        annotations[idx]["label"],
    )


def split_clause(
    label,
    current_text,
    split_text,
    split_label,
    doc_id,
    metadata,
    clauses,
    annotations,
    idx,
):
    if not clauses or idx is None:
        return _response(
            "No file loaded.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
            split_text,
            split_label,
        )

    ok, msg = _store_current_clause(label, current_text, clauses, annotations, idx)
    if not ok:
        return _response(
            msg,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
            split_text,
            split_label,
        )

    clean_split = (split_text or "").strip()
    if not clean_split:
        return _response(
            "❌ Provide text for the new clause.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
            split_text,
            split_label,
        )

    insert_idx = idx + 1
    clauses.insert(insert_idx, clean_split)
    annotations.insert(
        insert_idx,
        {"text": clean_split, "label": split_label if split_label else "None"},
    )

    status = f"✂️ Split into clauses {idx+1} & {insert_idx+1}"
    return _response(
        status,
        doc_id,
        metadata,
        clauses,
        annotations,
        insert_idx,
        clean_split,
        annotations[insert_idx]["label"],
        "",
        "None",
    )


def go_to_clause(
    goto_value,
    label,
    current_text,
    doc_id,
    metadata,
    clauses,
    annotations,
    idx,
):
    if not clauses or idx is None:
        return _response(
            "No file loaded.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
        )

    ok, msg = _store_current_clause(label, current_text, clauses, annotations, idx)
    if not ok:
        return _response(
            msg, doc_id, metadata, clauses, annotations, idx, current_text, label
        )

    try:
        goto_idx = int(goto_value) - 1
    except (TypeError, ValueError):
        return _response(
            "❌ Enter a valid clause number.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
        )

    if goto_idx < 0 or goto_idx >= len(clauses):
        return _response(
            f"❌ Choose between 1 and {len(clauses)}.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
        )

    status = f"Jumped to clause {goto_idx+1}/{len(clauses)}"
    return _response(
        status,
        doc_id,
        metadata,
        clauses,
        annotations,
        goto_idx,
        clauses[goto_idx],
        annotations[goto_idx]["label"],
    )


def save_now(label, current_text, doc_id, metadata, clauses, annotations, idx):
    if not clauses or idx is None:
        return _response(
            "No file loaded.",
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_text,
            label,
        )

    ok, msg = _store_current_clause(label, current_text, clauses, annotations, idx)
    if not ok:
        return _response(
            msg, doc_id, metadata, clauses, annotations, idx, current_text, label
        )

    outpath = save_annotations(doc_id, annotations, metadata)
    status = f"💾 Saved annotations to {outpath}"
    return _response(
        status,
        doc_id,
        metadata,
        clauses,
        annotations,
        idx,
        clauses[idx],
        annotations[idx]["label"],
    )


# GRADIO INTERFACE
all_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")], key=natural_key
)
annotated_files = {
    os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")
}

file_display_map = {}
files_display = []
for fname in all_files:
    base = os.path.splitext(fname)[0]
    display_name = f"✅ {fname}" if base in annotated_files else fname
    file_display_map[display_name] = fname
    files_display.append(display_name)

with gr.Blocks() as demo:
    gr.Markdown("## 🏷️ Editable Legal Clause Annotation Tool")
    with gr.Row():
        selected_file = gr.Dropdown(
            files_display,
            label="Choose a file",
            info="✅ indicates an existing JSON annotation",
        )
        load_btn = gr.Button("Load")

    with gr.Row():
        goto_input = gr.Textbox(label="Go to clause #", placeholder="", scale=1)
        goto_btn = gr.Button("Go", scale=0)

    status = gr.Textbox(label="Status", interactive=False)

    doc_id = gr.State()
    metadata = gr.State()
    clauses = gr.State()
    annotations = gr.State()
    idx = gr.State()

    with gr.Row():
        current_clause = gr.Textbox(
            label="Clause (editable)", lines=6, interactive=True, scale=4
        )
        copy_btn = gr.Button("📋 Copy", scale=1)

    label_choice = gr.Radio(LABELS, label="Label")

    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        next_btn = gr.Button("➡️ Next")
        update_btn = gr.Button("✏️ Update Clause")
        save_btn = gr.Button("💾 Save JSON")

    split_clause_box = gr.Textbox(
        label="New clause text (for split)",
        lines=4,
        placeholder="Enter text for next clause",
    )
    split_label_choice = gr.Radio(LABELS, label="Label for new clause", value="None")
    split_btn = gr.Button("✂️ Split & Insert")

    copy_btn.click(
        lambda x: None,
        inputs=[current_clause],
        outputs=[],
        js="""
        (text) => {
            navigator.clipboard.writeText(text).then(() => {
                const toast = document.createElement('div');
                toast.textContent = '✅ Copied to clipboard';
                toast.style.position = 'fixed';
                toast.style.bottom = '20px';
                toast.style.right = '20px';
                toast.style.background = '#4caf50';
                toast.style.color = '#fff';
                toast.style.padding = '8px 12px';
                toast.style.borderRadius = '8px';
                toast.style.fontSize = '14px';
                toast.style.zIndex = 9999;
                document.body.appendChild(toast);
                setTimeout(() => toast.remove(), 1200);
            });
        }
        """,
    )

    load_btn.click(
        start_annotation,
        inputs=[selected_file],
        outputs=[
            status,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_clause,
            label_choice,
            split_clause_box,
            split_label_choice,
        ],
    )

    next_btn.click(
        annotate_next,
        inputs=[
            label_choice,
            current_clause,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
        ],
        outputs=[
            status,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_clause,
            label_choice,
            split_clause_box,
            split_label_choice,
        ],
    )

    prev_btn.click(
        annotate_prev,
        inputs=[
            label_choice,
            current_clause,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
        ],
        outputs=[
            status,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_clause,
            label_choice,
            split_clause_box,
            split_label_choice,
        ],
    )

    update_btn.click(
        update_clause,
        inputs=[
            label_choice,
            current_clause,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
        ],
        outputs=[
            status,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_clause,
            label_choice,
            split_clause_box,
            split_label_choice,
        ],
    )

    split_btn.click(
        split_clause,
        inputs=[
            label_choice,
            current_clause,
            split_clause_box,
            split_label_choice,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
        ],
        outputs=[
            status,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_clause,
            label_choice,
            split_clause_box,
            split_label_choice,
        ],
    )

    save_btn.click(
        save_now,
        inputs=[
            label_choice,
            current_clause,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
        ],
        outputs=[
            status,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_clause,
            label_choice,
            split_clause_box,
            split_label_choice,
        ],
    )

    goto_btn.click(
        go_to_clause,
        inputs=[
            goto_input,
            label_choice,
            current_clause,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
        ],
        outputs=[
            status,
            doc_id,
            metadata,
            clauses,
            annotations,
            idx,
            current_clause,
            label_choice,
            split_clause_box,
            split_label_choice,
        ],
    )

demo.launch()
