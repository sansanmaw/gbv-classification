import os, re, pickle
import torch, torch.nn as nn
import numpy as np, pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from groq import Groq
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ARTIFACT_DIR = 'gbv_mtl_roberta_model'

MODEL_SOURCES = {
    "Baseline Model": {
        "type":          "hub",
        "repo_id":       "sansanmaw/gbv-baseline-model",
        "has_intensity": False,
    },
    "MTL RoBERTa": {
        "type":          "local",
        "path":          ARTIFACT_DIR,
        "has_intensity": True,
    },
}

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "llama3-70b-8192",
]

PROTOCOLS = {
    'sexual_violence':              ["Initiate Clinical Forensic Examination protocol", "Refer to specialized SGBV counselor", "Legal advocacy for protective orders"],
    'Physical_violence':            ["Immediate Safety Planning (DV-specific)", "Shelter coordination", "Mandatory reporting review"],
    'economic_violence':            ["Financial dependency assessment", "Link to economic empowerment NGOs", "Asset protection advice"],
    'emotional_violence':           ["Psychosocial support (PSS) mapping", "Chronic stress assessment", "Support group referral"],
    'Harmful_Traditional_practice': ["Community mediation/advocacy", "Human rights legal support", "Child protection services (if minor involved)"],
    'Non-GBV':                      ["No GBV indicators detected", "Signpost to general community services if needed", "Document and close case"],
}

SEVERITY_COLOR = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
LABEL_ICON = {
    'sexual_violence': '🚨', 'Physical_violence': '🩹',
    'economic_violence': '💸', 'emotional_violence': '💬',
    'Harmful_Traditional_practice': '⚠️', 'Non-GBV': '✅',
}

# ──────────────────────────────────────────────
# SUPABASE FEEDBACK
# ──────────────────────────────────────────────
def _get_supabase():
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None

def save_feedback(text, model_prediction, correct_label, intensity, confidence, model_used):
    row = {
        "created_at":       datetime.utcnow().isoformat(),
        "text":             text[:500],
        "model_prediction": model_prediction,
        "correct_label":    correct_label,
        "intensity":        intensity if intensity else "N/A",
        "confidence":       round(float(confidence), 4),
        "model_used":       model_used,
    }
    client = _get_supabase()
    if client:
        try:
            client.table("human_feedback").insert(row).execute()
            return True
        except Exception as e:
            st.warning(f"Supabase error: {e}. Saving locally instead.")
    path = os.path.join(ARTIFACT_DIR, "human_feedback_local.csv")
    fb_df = pd.DataFrame([row])
    fb_df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return True

# ──────────────────────────────────────────────
# MTL MODEL CLASS (used for MTL model only)
# ──────────────────────────────────────────────
class MultiTaskGBVModel(nn.Module):
    def __init__(self, model_name, num_gbv_labels, num_intensity_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.gbv_classifier       = nn.Sequential(nn.Linear(h, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, num_gbv_labels))
        self.intensity_classifier = nn.Sequential(nn.Linear(h, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, num_intensity_labels))

    def forward(self, input_ids=None, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.gbv_classifier(cls), self.intensity_classifier(cls)

# ──────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────
def preprocess_for_roberta(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r'[^\w\s\.\?\!]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(source_key: str):
    cfg = MODEL_SOURCES[source_key]

    if cfg["type"] == "local":
        # MTL model — loads from GitHub repo folder
        path = cfg["path"]
        with open(os.path.join(path, 'label_mappings.pkl'), 'rb') as f:
            m = pickle.load(f)
        tok = AutoTokenizer.from_pretrained(path)
        mdl = MultiTaskGBVModel(m['model_name'], len(m['id2label']), 3)
        mdl.load_state_dict(
            torch.load(os.path.join(path, 'multitask_roberta_model.pt'), map_location='cpu'),
            strict=False
        )
        mdl.eval()
    else:
        # Baseline model — downloads from Hugging Face automatically
        from huggingface_hub import hf_hub_download
        repo = cfg["repo_id"]
        pkl_path = hf_hub_download(repo_id=repo, filename="label_mappings.pkl")
        with open(pkl_path, 'rb') as f:
            m = pickle.load(f)
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForSequenceClassification.from_pretrained(repo)
        mdl.eval()

    return tok, mdl, m

# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
def predict_single(text, tok, mdl, mappings, has_intensity=True):
    enc = tok(preprocess_for_roberta(text), return_tensors='pt',
               truncation=True, padding='max_length', max_length=128)

    with torch.no_grad():
        if mappings.get("model_type") == "sequence_classification":
            # Baseline — one output (GBV type only)
            logits    = mdl(enc['input_ids'], enc['attention_mask']).logits
            probs     = torch.softmax(logits[0], dim=0).numpy()
            int_label = None
        else:
            # MTL — two outputs (GBV type + severity)
            g_logits, i_logits = mdl(enc['input_ids'], enc['attention_mask'])
            probs     = torch.softmax(g_logits[0], dim=0).numpy()
            int_label = mappings['intensity_id2label'][int(torch.argmax(i_logits))] if has_intensity else None

    pred_id = int(np.argmax(probs))
    label   = mappings['id2label'][pred_id]
    scores  = dict(sorted(
        {mappings['id2label'][i]: float(probs[i]) for i in range(len(probs))}.items(),
        key=lambda x: x[1], reverse=True
    ))
    return label, int_label, float(np.max(probs)), scores

# ──────────────────────────────────────────────
# AI ADVISOR
# ──────────────────────────────────────────────
def get_casework_ai_advice(label, intensity, query):
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if not key:
            return "⚠️ GROQ_API_KEY not found in Streamlit Secrets."
        client = Groq(api_key=key)
        severity_line = f"- Severity Level: {intensity}" if intensity else ""
        prompt = f"""You are a professional GBV (Gender-Based Violence) casework advisor supporting humanitarian and community workers in the field.

A case has been flagged with the following details:
- GBV Classification: {label}
{severity_line}
- Incident Description: {query}

Provide structured, trauma-informed casework guidance using this exact format:

**Situation Assessment**
A concise 2–3 sentence professional reading of the case.

**Immediate Actions**
- Action 1
- Action 2
- Action 3

**Referral Pathways**
- Referral 1
- Referral 2

**Safety Considerations**
Key risks to monitor and steps to protect the survivor.

Keep your response practical, compassionate, and suitable for a field caseworker with limited resources."""
        last_error = None
        for model_id in GROQ_MODELS:
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_error = e
                continue
        return f"⚠️ All AI models currently unavailable. Last error: {last_error}"
    except Exception as e:
        return f"⚠️ Casework Assistant unavailable. (Error: {str(e)})"

# ──────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────
st.set_page_config(
    page_title='GBV Case Management Dashboard',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stProgress > div > div { border-radius: 6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; }
    .ethical-note {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Model Settings")
    selected_model = st.selectbox(
        "Active Model",
        options=list(MODEL_SOURCES.keys()),
        help="MTL RoBERTa: GBV type + severity. Baseline: GBV type only."
    )
    has_intensity = MODEL_SOURCES[selected_model]["has_intensity"]
    if not has_intensity:
        st.info("Baseline model shows GBV type only. Switch to MTL RoBERTa for intensity scoring.")
    st.divider()
    st.caption("GBV Case Management Dashboard\n | Capstone Project")

tokenizer, model, mappings = load_model(selected_model)

# ── Header ──
st.markdown("## 🛡️ GBV Case Management Dashboard")
st.markdown("*An AI-assisted tool to support humanitarian and community workers handling Gender-Based Violence cases.*")
st.markdown("""
<div class="ethical-note">
⚠️ <strong> Notice :</strong> This tool uses AI to assist with case classification and is intended to support — not replace — professional judgment.
Always verify the classification and apply your own expertise before taking action.
</div>
""", unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
t1, t2 = st.tabs(["📝 Single Case Assessment", "📂 Batch File Analysis"])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE CASE ASSESSMENT
# ══════════════════════════════════════════════
with t1:
    st.markdown("### Incident Narrative")
    st.caption("Enter the incident text (tweets) below. The model will classify the GBV type and provide casework guidance.")

    user_input = st.text_area(
        label='Incident Narrative',
        height=140,
        placeholder="Describe the incident or paste a social media post here...",
        label_visibility='collapsed',
        key='narrative'
    )

    col_btn, _ = st.columns([3, 3])
    with col_btn:
        analyze_clicked = st.button('🔍 Analyze Case', type='primary', use_container_width=True)

    if analyze_clicked:
        if user_input.strip():
            with st.spinner("Analyzing..."):
                label, int_label, top_confidence, confidence_scores = predict_single(
                    user_input, tokenizer, model, mappings, has_intensity=has_intensity
                )
                ai_advice = get_casework_ai_advice(label, int_label, user_input)
            st.session_state['last_result'] = {
                'label':             label,
                'int_label':         int_label,
                'has_intensity':     has_intensity,
                'top_confidence':    top_confidence,
                'user_input':        user_input,
                'ai_advice':         ai_advice,
                'confidence_scores': confidence_scores,
            }
            st.session_state.pop('feedback_saved', None)
        else:
            st.warning("Please enter an incident narrative before analyzing.")

    if 'last_result' in st.session_state:
        r = st.session_state['last_result']
        st.divider()

        left, right = st.columns([1, 1], gap="large")

        with left:
            icon = LABEL_ICON.get(r['label'], '📌')
            st.markdown(f"#### {icon} Classification Result")
            st.success(f"**{r['label'].replace('_', ' ')}**")

            if r['has_intensity'] and r['int_label']:
                sicon = SEVERITY_COLOR.get(r['int_label'], '⚪')
                st.markdown(f"{sicon} **Severity:** {r['int_label']}   &nbsp;&nbsp; 🎯 **Confidence:** {r['top_confidence']*100:.1f}%")
            else:
                st.markdown(f"🎯 **Confidence:** {r['top_confidence']*100:.1f}%")
                st.caption("Severity scoring is only available with MTL RoBERTa.")

            st.markdown("---")
            st.markdown("**Model Confidence Across All Classes**")
            for class_name, score in r['confidence_scores'].items():
                display = class_name.replace('_', ' ')
                bold = "**" if class_name == r['label'] else ""
                st.markdown(f"{bold}{display}{bold}")
                st.progress(float(score), text=f"{score*100:.1f}%")

            st.markdown("---")
            st.markdown("**📋 Recommended Protocols**")
            for p in PROTOCOLS.get(r['label'], ["Apply general GBV support guidelines"]):
                st.markdown(f"- {p}")

            st.markdown("---")
            st.markdown("**🗳️ Caseworker Feedback**")
            st.caption("Your feedback is saved privately and used only to improve the model.")
            feedback_correct = st.radio(
                "Is this classification correct?",
                ("Yes", "No — it should be something else"),
                horizontal=True,
                key='fb_radio'
            )
            correct_label = r['label']
            if feedback_correct == "No — it should be something else":
                correct_label = st.selectbox(
                    "Select the correct label:",
                    options=list(PROTOCOLS.keys()),
                    key='fb_correction'
                )
            if st.button("💾 Save Feedback", use_container_width=True):
                save_feedback(
                    text=r['user_input'],
                    model_prediction=r['label'],
                    correct_label=correct_label,
                    intensity=r['int_label'],
                    confidence=r['top_confidence'],
                    model_used=selected_model,
                )
                st.session_state['feedback_saved'] = True
            if st.session_state.get('feedback_saved'):
                st.success("✅ Feedback recorded. Thank you!")

        with right:
            st.markdown("#### 🤖 AI Casework Advisor")
            st.caption("Structured guidance generated by AI to assist your response planning.")
            st.markdown(r['ai_advice'])

# ══════════════════════════════════════════════
# TAB 2 — BATCH FILE ANALYSIS
# ══════════════════════════════════════════════
with t2:
    st.markdown("### Batch File Analysis")
    st.caption("Upload a CSV or Excel file with a **'tweet'** column. The model will classify each row.")

    uploaded_file = st.file_uploader("Upload file", type=['csv', 'xlsx', 'xls'], label_visibility='collapsed')

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

            if 'tweet' not in batch_df.columns:
                st.error("❌ The file must contain a column named **'tweet'**.")
            else:
                st.success(f"✅ File loaded — **{len(batch_df)} rows** found.")
                st.dataframe(batch_df[['tweet']].head(5), use_container_width=True)

                if st.button("▶️ Run Batch Analysis", type='primary'):
                    results  = []
                    progress = st.progress(0, text="Processing rows...")
                    status   = st.empty()

                    for i, row in batch_df.iterrows():
                        tweet = str(row.get('tweet', ''))
                        if not tweet.strip():
                            results.append({'tweet': tweet, 'predicted_label': 'N/A', 'intensity': 'N/A', 'confidence_%': 'N/A'})
                            continue
                        lbl, i_lbl, conf, _ = predict_single(tweet, tokenizer, model, mappings, has_intensity=has_intensity)
                        results.append({
                            'tweet':           tweet,
                            'predicted_label': lbl,
                            'intensity':       i_lbl if i_lbl else 'N/A',
                            'confidence_%':    round(conf * 100, 1),
                        })
                        progress.progress((i + 1) / len(batch_df), text=f"Row {i+1} of {len(batch_df)}...")
                        status.caption(f"Latest: **{lbl}** ({conf*100:.1f}%)")

                    progress.empty(); status.empty()
                    results_df = pd.DataFrame(results)
                    st.success(f"✅ Done — {len(results_df)} rows processed.")
                    st.dataframe(results_df, use_container_width=True)

                    label_counts = results_df['predicted_label'].value_counts().reset_index()
                    label_counts.columns = ['Label', 'Count']
                    st.markdown("**Distribution of Predicted Labels**")
                    st.dataframe(label_counts, use_container_width=True)

                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=results_df.to_csv(index=False),
                        file_name="gbv_batch_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")

st.markdown("---")
st.caption("GBV Case Management Dashboard | Built for humanitarian and community workers | Capstone Project")
