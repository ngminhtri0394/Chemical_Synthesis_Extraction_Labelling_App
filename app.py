import streamlit as st
import json
import base64
import fitz  # PyMuPDF
import tempfile
import os
import re

# --- Configuration ---
st.set_page_config(page_title="Materials Synthesis Annotator", layout="wide")

# --- Fuzzy Matching & Highlighting Engine ---
def make_fuzzy_regex(term):
    """Builds a punctuation-agnostic regex for complex chemical formulas."""
    chars = []
    for char in term:
        if char.isalnum() or char in "()[]":
            chars.append(re.escape(char))
        else:
            chars.append(r'.{1,3}') # Wildcard for mojibake (e.g. Â· instead of ·)
    return r'[\s\n]*'.join(chars)

def extract_categorized_highlights(data):
    """Extracts terms and verbatim quotes from the JSON for color-coded highlighting."""
    config = {
        "targets": {"color": (1.0, 0.6, 0.6), "texts": set()},        # Light Red
        "chemicals": {"color": (0.6, 1.0, 0.6), "texts": set()},      # Light Green
        "synthesis": {"color": (0.6, 0.8, 1.0), "texts": set()},      # Light Blue
        "characterization": {"color": (0.9, 0.6, 1.0), "texts": set()},# Light Purple
        "ambiguities": {"color": (1.0, 0.9, 0.4), "texts": set()}     # Light Yellow
    }

    # 1. Targets & Chemicals
    for target in data.get("targets", []):
        if target.get("compound_name"): config["targets"]["texts"].add(target["compound_name"])
        if target.get("molecular_formula"): config["targets"]["texts"].add(target["molecular_formula"])

    for chem in data.get("chemicals", []):
        if chem.get("name"): config["chemicals"]["texts"].add(chem["name"])
        if chem.get("molecular_formula"): config["chemicals"]["texts"].add(chem["molecular_formula"])

    # 2. Synthesis Quotes (handling single, double, and smart quotes)
    quote_pattern = r"['\u2018\u2019\"](.*?)['\u2018\u2019\"]"
    for step in data.get("synthesis", {}).get("steps", []):
        for prov_str in step.get("prov", []):
            config["synthesis"]["texts"].update(re.findall(quote_pattern, prov_str))
        for op in step.get("operations", []):
            for prov_str in op.get("prov", []):
                config["synthesis"]["texts"].update(re.findall(quote_pattern, prov_str))

    # 3. Characterization
    for char in data.get("characterization", []):
        if char.get("method"): config["characterization"]["texts"].add(char["method"])

    # 4. Ambiguities
    for amb in data.get("workflow", {}).get("ambiguities", []):
        excerpt = amb.get("excerpt", "")
        # Strip the outer smart quotes to get the raw string
        clean_excerpt = re.sub(r"^['\u2018\u2019\"]|['\u2018\u2019\"]$", "", excerpt.strip())
        if clean_excerpt:
            config["ambiguities"]["texts"].add(clean_excerpt)

    # Convert sets to lists
    for key in config:
        config[key]["texts"] = list(config[key]["texts"])

    return config

@st.cache_data(show_spinner="Generating highlighted PDF...")
def create_highlighted_pdf(pdf_bytes, highlights_config):
    """Applies highlights to the PDF, using whole-word matching for acronyms."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        page_text = page.get_text("text")
        # Extract the page's structural word list for exact matching
        page_words = page.get_text("words")

        for category, config in highlights_config.items():
            rgb_color = config["color"]
            for text in config["texts"]:
                if not text or len(text) < 2: continue

                actual_strings = set()

                # 1. Fuzzy matching for complex formulas and long quotes
                if any(char.isdigit() or char in "()[]" for char in text) or len(text) > 4:
                    regex_pattern = make_fuzzy_regex(text)
                    matches = re.findall(regex_pattern, page_text, flags=re.IGNORECASE)
                    actual_strings.update(matches)
                else:
                    actual_strings.add(text)

                # 2. Apply the highlights
                for actual_string in actual_strings:
                    actual_string = actual_string.strip()
                    if not actual_string: continue

                    # --- THE SMART WHOLE-WORD FIX ---
                    # If it's a short alphanumeric string (like TEM, SEM, XRD)
                    if actual_string.isalnum() and len(actual_string) <= 4:
                        for w in page_words:
                            # Strip punctuation from the PDF word (e.g., "TEM," -> "TEM")
                            clean_word = re.sub(r'[^a-zA-Z0-9]', '', w[4])

                            # Enforce exact case-sensitive match
                            if clean_word == actual_string:
                                annot = page.add_highlight_annot(fitz.Rect(w[:4]))
                                annot.set_colors(stroke=rgb_color)
                                annot.update()
                    else:
                        # Fallback to standard search for long phrases and spaced formulas
                        text_instances = page.search_for(actual_string)
                        for inst in text_instances:
                            annot = page.add_highlight_annot(inst)
                            annot.set_colors(stroke=rgb_color)
                            annot.update()

    temp_dir = tempfile.gettempdir()
    temp_pdf_path = os.path.join(temp_dir, "highlighted_master.pdf")
    doc.save(temp_pdf_path)
    doc.close()

    return temp_pdf_path

# --- UI Helper Functions ---
def display_pdf(file_path):
    try:
        from streamlit_pdf_viewer import pdf_viewer
        pdf_viewer(file_path, height=850)
    except ImportError:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="850px" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)

def save_data(filename="Corrected_extraction.json"):
    """Save the corrected data to a file and provide download."""
    json_str = json.dumps(st.session_state.data, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Corrected JSON",
        data=json_str,
        file_name=filename,
        mime="application/json"
    )

def render_prov(prov_list, key_prefix):
    st.markdown("**Provenance:**")
    prov_str = "\n".join(prov_list) if prov_list else ""
    new_prov = st.text_area("Edit Provenance", value=prov_str, key=f"{key_prefix}_prov", height=68, label_visibility="collapsed")
    return [p.strip() for p in new_prov.split("\n") if p.strip()]

def render_llm_flag(flag_data):
    """Renders the AI Verification flag UI component."""
    if not flag_data:
        return # Do nothing if the LLM hasn't verified this field yet

    if not flag_data.get('is_supported') or flag_data.get('confidence', 1.0) < 0.8:
        st.warning(f"""
        **AI Verification Flag** (Confidence: {flag_data.get('confidence', 0.0):.2f})
        * **Reasoning:** {flag_data.get('reasoning')}
        * **Suggested Fix:** {flag_data.get('suggested_fix', 'N/A')}
        """)
    else:
        st.success(f"AI Verified (Confidence: {flag_data.get('confidence', 1.0):.2f})")

# --- Main App ---
st.title("Materials Synthesis Annotator")

# --- File Upload Section ---
st.sidebar.header("File Selection")

# JSON file upload
json_file = st.sidebar.file_uploader("Upload JSON Extraction", type=["json"], help="Upload the JSON file containing extracted data")

# PDF file upload
pdf_file = st.sidebar.file_uploader("Upload Source PDF", type=["pdf"], help="Upload the source PDF document")

# Load data from uploaded JSON
if json_file is not None:
    try:
        st.session_state.data = json.load(json_file)
        st.sidebar.success(f"Loaded: {json_file.name}")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON file")
        st.session_state.data = {}
else:
    if 'data' not in st.session_state:
        st.session_state.data = {}

# Store PDF bytes in session state
if pdf_file is not None:
    st.session_state.pdf_bytes = pdf_file.read()
    st.session_state.pdf_name = pdf_file.name
    st.sidebar.success(f"Loaded: {pdf_file.name}")
else:
    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_name = None

data = st.session_state.data

# Check if files are loaded
if not data:
    st.info("Please upload a JSON extraction file using the sidebar.")
elif st.session_state.pdf_bytes is None:
    st.info("Please upload a source PDF file using the sidebar.")
else:
    allowed_units = [str(v) for k, v in data.get("units_policy", {}).items() if isinstance(v, str)]
    col_pdf, col_form = st.columns([1.2, 1])

    # --- LEFT PANE: PDF Viewer ---
    with col_pdf:
        st.subheader("Source Document")

        # 1. Prepare Highlights
        highlights_config = extract_categorized_highlights(data)

        # 2. Legend
        st.markdown("**Legend:** 🔴 Targets | 🟢 Chemicals | 🔵 Synthesis | 🟣 Characterization | 🟡 Ambiguities")

        # 3. Generate and Display
        highlighted_pdf_path = create_highlighted_pdf(st.session_state.pdf_bytes, highlights_config)
        display_pdf(highlighted_pdf_path)

    # --- RIGHT PANE: JSON Form ---
    with col_form:
        st.subheader("Extracted Entity Form")
        if allowed_units:
            st.info(f"**Active Unit Policy:** {', '.join(allowed_units)}")

        with st.expander("Source & Metadata", expanded=False):
            st.text_input("Title", value=data.get("source", {}).get("title", ""))

        # 2. Targets Section
        with st.expander("Target Compounds", expanded=False):
            for i, target in enumerate(data.get("targets", [])):
                st.markdown(f"**Target {i+1}: {target.get('target_id')}**")
                target['compound_name'] = st.text_input("Name", value=target.get("compound_name", ""), key=f"t_name_{i}")
                target['molecular_formula'] = st.text_input("Formula", value=target.get("molecular_formula", ""), key=f"t_form_{i}")

                # --- AI VERIFICATION ---
                render_llm_flag(target.get('_llm_flag'))

                target['prov'] = render_prov(target.get("prov", []), f"t_{i}")
                st.divider()

        # 3. Chemicals Section
        with st.expander("Chemicals List", expanded=False):
            for i, chem in enumerate(data.get("chemicals", [])):
                st.markdown(f"**{chem.get('chemical_id')}: {chem.get('name')}**")
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1: chem['name'] = st.text_input("Name", value=chem.get('name', ''), key=f"c_name_{i}")
                with col2:
                    val = chem.get('amount', {}).get('value')
                    chem['amount']['value'] = st.number_input("Amount", value=float(val) if val else 0.0, key=f"c_amt_{i}")
                with col3:
                    u = chem.get('amount', {}).get('unit', '')
                    chem['amount']['unit'] = st.text_input("Unit", value=u, key=f"c_unit_{i}")

                if u and u not in allowed_units: st.error(f"Policy Violation: '{u}' is not allowed.")

                # --- AI VERIFICATION ---
                render_llm_flag(chem.get('_llm_flag'))

                chem['prov'] = render_prov(chem.get("prov", []), f"c_{i}")
                st.divider()

        # 4. Synthesis Section
        with st.expander("Synthesis Steps", expanded=False):
            for i, step in enumerate(data.get("synthesis", {}).get("steps", [])):
                st.markdown(f"**Step {i+1}: {step.get('step_id')}**")
                step['description'] = st.text_input("Description", value=step.get("description", ""), key=f"syn_desc_{i}")

                # --- Synthesis Conditions ---
                conditions = step.get("conditions", {})
                if conditions:
                    st.markdown("*Conditions:*")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        temp = conditions.get("temperature", {})
                        temp_val = temp.get("value") if temp else None
                        conditions.setdefault("temperature", {})["value"] = st.number_input(
                            "Temp", value=float(temp_val) if temp_val is not None else 0.0, key=f"syn_temp_{i}"
                        )
                        conditions["temperature"]["unit"] = st.text_input(
                            "Temp Unit", value=temp.get("unit", "K") if temp else "K", key=f"syn_temp_u_{i}"
                        )
                    with col2:
                        pressure = conditions.get("pressure", {})
                        pres_val = pressure.get("value") if pressure else None
                        conditions.setdefault("pressure", {})["value"] = st.number_input(
                            "Pressure", value=float(pres_val) if pres_val is not None else 0.0, key=f"syn_pres_{i}"
                        )
                        conditions["pressure"]["unit"] = st.text_input(
                            "Pres Unit", value=pressure.get("unit", "bar") if pressure else "bar", key=f"syn_pres_u_{i}"
                        )
                    with col3:
                        duration = conditions.get("duration", {})
                        dur_val = duration.get("value") if duration else None
                        conditions.setdefault("duration", {})["value"] = st.number_input(
                            "Duration", value=float(dur_val) if dur_val is not None else 0.0, key=f"syn_dur_{i}"
                        )
                        conditions["duration"]["unit"] = st.text_input(
                            "Dur Unit", value=duration.get("unit", "min") if duration else "min", key=f"syn_dur_u_{i}"
                        )
                    with col4:
                        atm = conditions.get("atmosphere", "")
                        conditions["atmosphere"] = st.text_input("Atmosphere", value=atm, key=f"syn_atm_{i}")

                    # --- AI VERIFICATION for Conditions ---
                    render_llm_flag(conditions.get('_llm_flag'))

                for j, op in enumerate(step.get("operations", [])):
                    st.markdown(f"*Operation {j+1}:* `{op.get('action')}`")
                    col1, col2 = st.columns(2)
                    with col1:
                        op['action'] = st.text_input("Action", value=op.get("action", ""), key=f"syn_action_{i}_{j}")
                    with col2:
                        op['notes'] = st.text_input("Notes", value=op.get("notes", ""), key=f"syn_notes_{i}_{j}")

                    # --- AI VERIFICATION ---
                    render_llm_flag(op.get('_llm_flag'))

                    op['prov'] = render_prov(op.get("prov", []), f"syn_op_{i}_{j}")

                step['prov'] = render_prov(step.get("prov", []), f"syn_step_{i}")
                st.divider()

        # 5. Characterization Section
        with st.expander("Characterization", expanded=False):
            for i, char in enumerate(data.get("characterization", [])):
                st.markdown(f"**{char.get('method')}**")
                char['method'] = st.text_input("Method", value=char.get('method', ''), key=f"char_m_{i}")

                # Editable results - iterate all results, not just first
                for j, res in enumerate(char.get("results", [])):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        res['metric'] = st.text_input("Metric", value=res.get('metric', ''), key=f"char_met_{i}_{j}")
                    with col2:
                        res['value'] = st.text_input("Value", value=str(res.get('value', '')), key=f"char_val_{i}_{j}")
                    with col3:
                        res['unit'] = st.text_input("Unit", value=res.get('unit', ''), key=f"char_unit_{i}_{j}")

                # --- AI VERIFICATION ---
                render_llm_flag(char.get('_llm_flag'))

                char['prov'] = render_prov(char.get("prov", []), f"char_{i}")
                st.divider()

        # 6. Analysis Section
        with st.expander("Analysis", expanded=False):
            methods_list = data.get("analysis", {}).get("methods", [])
            for i, method in enumerate(methods_list):
                if isinstance(method, dict):
                    # Editable method name
                    method_name_key = 'method_name' if 'method_name' in method else ('name' if 'name' in method else 'method')
                    current_name = method.get(method_name_key, '')
                    method[method_name_key] = st.text_input("Method Name", value=current_name, key=f"ana_method_{i}")

                    # --- AI VERIFICATION for Analysis Method ---
                    render_llm_flag(method.get('_llm_flag'))

                    # Editable results
                    for j, res in enumerate(method.get("results", [])):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            res['sample'] = st.text_input("Sample", value=res.get('sample', ''), key=f"ana_s_{i}_{j}")
                        with col2:
                            val = res.get('value', 0)
                            res['value'] = st.number_input("Value", value=float(val) if val else 0.0, key=f"ana_v_{i}_{j}")
                        with col3:
                            res['unit'] = st.text_input("Unit", value=res.get('unit', ''), key=f"ana_u_{i}_{j}")

                        # --- AI VERIFICATION ---
                        render_llm_flag(res.get('_llm_flag'))

                    # Editable provenance for analysis method
                    method['prov'] = render_prov(method.get("prov", []), f"ana_{i}")
                elif isinstance(method, str):
                    # Convert string to dict for editability
                    new_name = st.text_input("Method Name", value=method, key=f"ana_method_{i}")
                    methods_list[i] = {"method_name": new_name, "results": [], "prov": []}
                st.divider()

        with st.expander("Workflow & Ambiguities", expanded=True):
            workflow = data.get("workflow", {})

            # --- Timeline Section ---
            timeline = workflow.get("timeline", [])
            if timeline:
                st.markdown("**Timeline:**")
                for i, entry in enumerate(timeline):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        entry['step_ref'] = st.text_input("Step Ref", value=entry.get('step_ref', ''), key=f"wf_step_{i}")
                    with col2:
                        entry['description'] = st.text_input("Description", value=entry.get('description', ''), key=f"wf_desc_{i}")

                    # --- AI VERIFICATION for Timeline ---
                    render_llm_flag(entry.get('_llm_flag'))
                st.divider()

            # --- Ambiguities Section ---
            st.markdown("**Ambiguities Identified:**")
            ambiguities_list = workflow.get("ambiguities", [])
            for i, amb in enumerate(ambiguities_list):
                if isinstance(amb, dict):
                    col1, col2 = st.columns(2)
                    with col1:
                        amb['issue'] = st.text_input("Issue", value=amb.get('issue', ''), key=f"amb_issue_{i}")
                    with col2:
                        amb['excerpt'] = st.text_input("Excerpt", value=amb.get('excerpt', ''), key=f"amb_excerpt_{i}")
                    # --- AI VERIFICATION for Ambiguity ---
                    render_llm_flag(amb.get('_llm_flag'))
                elif isinstance(amb, str):
                    # Convert string to dict for editability
                    new_issue = st.text_input("Ambiguity", value=amb, key=f"amb_issue_{i}")
                    ambiguities_list[i] = {"issue": new_issue, "excerpt": ""}

        # 8. Final Outcomes Section
        with st.expander("Final Outcomes", expanded=False):
            # Ensure final_outcomes exists in data
            if "final_outcomes" not in data:
                data["final_outcomes"] = {}
            final_outcomes = data["final_outcomes"]

            col1, col2, col3 = st.columns(3)

            # Yield
            with col1:
                st.markdown("**Yield**")
                if "yield" not in final_outcomes:
                    final_outcomes["yield"] = {"value": None, "unit": "%"}
                yield_data = final_outcomes["yield"]
                yield_val = yield_data.get("value")
                yield_data["value"] = st.number_input(
                    "Yield Value", value=float(yield_val) if yield_val is not None else 0.0, key="fo_yield_val"
                )
                yield_data["unit"] = st.text_input(
                    "Yield Unit", value=yield_data.get("unit", "%"), key="fo_yield_unit"
                )

            # Capacity
            with col2:
                st.markdown("**Capacity**")
                if "capacity" not in final_outcomes:
                    final_outcomes["capacity"] = {"value": None, "unit": "mAh g-1"}
                capacity_data = final_outcomes["capacity"]
                cap_val = capacity_data.get("value")
                capacity_data["value"] = st.number_input(
                    "Capacity Value", value=float(cap_val) if cap_val is not None else 0.0, key="fo_cap_val"
                )
                capacity_data["unit"] = st.text_input(
                    "Capacity Unit", value=capacity_data.get("unit", "mAh g-1"), key="fo_cap_unit"
                )

            # Cycle Life
            with col3:
                st.markdown("**Cycle Life**")
                cycle_life = final_outcomes.get("cycle_life")
                final_outcomes["cycle_life"] = st.number_input(
                    "Cycles", value=int(cycle_life) if cycle_life is not None else 0, key="fo_cycle"
                )

            # --- AI VERIFICATION for Final Outcomes ---
            render_llm_flag(final_outcomes.get('_llm_flag'))

            # Provenance for final outcomes
            final_outcomes['prov'] = render_prov(final_outcomes.get("prov", []), "fo")

        st.markdown("<br>", unsafe_allow_html=True)

        # Save/Download Section
        col_save1, col_save2 = st.columns(2)
        with col_save1:
            # Generate filename based on source
            base_name = st.session_state.get('pdf_name', 'extraction')
            if base_name.endswith('.pdf'):
                base_name = base_name[:-4]
            output_filename = f"Corrected_{base_name}.json"

            json_str = json.dumps(st.session_state.data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download Corrected JSON",
                data=json_str,
                file_name=output_filename,
                mime="application/json",
                type="primary",
                use_container_width=True
            )
