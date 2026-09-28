import os
import time
import streamlit as st
from hindsight_client import Hindsight
from groq import Groq

# Page Config
st.set_page_config(
    page_title="Deal Intelligence Agent — Hindsight vs Stateless Demo",
    page_icon="🧠",
    layout="wide"
)

# Custom Light Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .stApp header {
        background-color: transparent;
    }
    .badge-stateless {
        background-color: #fffbe6;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-hindsight {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .memory-box {
        background-color: #f0fdf4;
        border: 1px solid #a7f3d0;
        border-left: 4px solid #059669;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# API Keys & Setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY", "")
HINDSIGHT_BASE_URL = os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")
BANK_ID = "deal_intelligence_demo"
DEFAULT_MODEL = "llama3-8b-8192"

@st.cache_resource
def get_clients():
    groq_c = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    hindsight_c = Hindsight(base_url=HINDSIGHT_BASE_URL, api_key=HINDSIGHT_API_KEY) if HINDSIGHT_API_KEY else None
    if hindsight_c:
        try:
            hindsight_c.create_bank(
                bank_id=BANK_ID,
                name="Deal Intelligence Assistant",
                mission="Remember client objections, past deal context, and technical hurdles."
            )
        except Exception:
            pass
    return groq_c, hindsight_c


groq_client, hindsight = get_clients()

def generate_groq_completion(client, prompt):
    models = [DEFAULT_MODEL, "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "qwen/qwen3.8-27b"]
    for m in models:
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return res.choices[0].message.content, m
        except Exception as e:
            if any(err in str(e).lower() for err in ["not_found", "does not exist", "decommissioned"]):
                continue
            raise e
    raise RuntimeError("No operational Groq model found.")

# Streamlit Title & Header
st.title("🧠 Deal Intelligence Agent")
st.caption("Demonstrating Hindsight Memory Layer vs. Stateless LLM Chatbot")

st.markdown("---")

# Sidebar for Retaining New Memory
with st.sidebar:
    st.header("📥 Memory Retention")
    st.markdown("Retain past sales calls, CTO objections, or technical requirements into Hindsight.")
    
    context_input = st.text_input("Context / Call Label", "Technical Discovery Call #3 with Acme Corp CTO")
    content_input = st.text_area("Client Note / Objection", "During call #3, the Acme Corp CTO raised a strong objection regarding SOC2 Type II certification proof and data residency guarantees.")
    
    if st.button("💾 Retain into Hindsight"):
        if content_input.strip():
            try:
                hindsight.retain(bank_id=BANK_ID, content=content_input.strip(), context=context_input)
                st.success("Memory retained into Hindsight bank successfully!")
            except Exception as e:
                st.error(f"Retain error: {e}")
        else:
            st.warning("Please enter note content.")

# Main Input Section
col_q, col_p = st.columns([3, 1])

with col_q:
    sales_query = st.text_area(
        "Sales Rep Scenario / Query:",
        "We have our next high-stakes sales meeting with Acme Corp tomorrow. How should our sales rep pitch them and structure the discussion?",
        height=100
    )

with col_p:
    st.write("**Quick Presets:**")
    if st.button("🏢 Acme Corp (SOC2)"):
        st.session_state["query"] = "We have our next meeting with Acme Corp tomorrow. How should our sales rep pitch them?"
    if st.button("⚡ TechCorp (SSO)"):
        st.session_state["query"] = "TechCorp's VP of Engineering is joining our demo. What technical prerequisites should we present?"

run_button = st.button("⚡ Run Side-by-Side Comparison", type="primary", use_container_width=True)

if run_button:
    start_t = time.time()
    
    # 1. Stateless Prompt
    stateless_prompt = f"You are a sales deal assistant. Current situation: {sales_query}. Provide a concise, high-impact sales pitch and meeting agenda."
    with st.spinner("Querying Stateless Groq LLM..."):
        stateless_out, model_used = generate_groq_completion(groq_client, stateless_prompt)
    
    # 2. Hindsight Recall
    recall_t = time.time()
    with st.spinner("Querying Hindsight Memory Bank..."):
        recalled_snippets = []
        try:
            resp = hindsight.recall(bank_id=BANK_ID, query=sales_query)
            if hasattr(resp, 'results') and resp.results:
                for r in resp.results:
                    txt = getattr(r, 'text', str(r)).strip()
                    if txt and txt not in recalled_snippets:
                        recalled_snippets.append(txt)
        except Exception as e:
            recalled_snippets = [f"Recall note: {e}"]
    recall_dur = round(time.time() - recall_t, 2)
    
    # 3. Hindsight Prompt
    memories_str = "\n- ".join(recalled_snippets[:4]) if recalled_snippets else "No past memories found."
    stateful_prompt = f"""You are an expert deal intelligence assistant for enterprise sales teams.

HISTORICAL MEMORIES RECALLED FROM HINDSIGHT:
- {memories_str}

CURRENT SITUATION / QUESTION:
{sales_query}

INSTRUCTIONS:
Provide a highly tailored, high-conversion sales strategy and meeting agenda. 
Directly address the client's past objections (such as SOC2 compliance or data residency) retrieved from memory to win the deal."""

    with st.spinner("Generating Hindsight-Augmented Strategy..."):
        hindsight_out, _ = generate_groq_completion(groq_client, stateful_prompt)

    total_dur = round(time.time() - start_t, 2)

    st.success(f"Execution complete in {total_dur}s (Hindsight Recall: {recall_dur}s) | Model: {model_used}")

    # Side-by-Side Comparison Columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<span class="badge-stateless">⚡ STATELESS CHATBOT (NO MEMORY)</span>', unsafe_allow_html=True)
        st.subheader("Standard LLM Response")
        st.markdown(stateless_out)

    with col2:
        st.markdown('<span class="badge-hindsight">🧠 HINDSIGHT-POWERED AGENT</span>', unsafe_allow_html=True)
        st.subheader("Memory-Augmented Response")
        
        if recalled_snippets:
            st.markdown(f"""
            <div class="memory-box">
                <strong>Recalled Context from Hindsight:</strong><br>
                {"<br>".join([f"• <em>{m}</em>" for m in recalled_snippets[:3]])}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(hindsight_out)
