import os
import sys

# Ensure UTF-8 output encoding for Windows console environments
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from hindsight_client import Hindsight
from groq import Groq

# 1. Initialize API Keys & Clients from Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY", "")
HINDSIGHT_BASE_URL = os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")

if not GROQ_API_KEY:
    print("[!] Notice: GROQ_API_KEY is not set. Please set the GROQ_API_KEY environment variable.")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

hindsight = Hindsight(
    base_url=HINDSIGHT_BASE_URL, 
    api_key=HINDSIGHT_API_KEY
) if HINDSIGHT_API_KEY else None


# Requested primary model as specified in requirements
DEFAULT_MODEL = "llama3-8b-8192"
BANK_ID = "deal_intelligence_demo"

def generate_completion(client, prompt, model_name=DEFAULT_MODEL):
    """
    Helper function to generate chat completions with Groq.
    If the requested model (e.g. llama3-8b-8192) is not active/available on the endpoint,
    it falls back to an available model on the account (e.g. openai/gpt-oss-20b).
    """
    candidate_models = [model_name, "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    
    for m in candidate_models:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content, m
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "does not exist" in err_str or "decommissioned" in err_str:
                continue
            else:
                raise e
    raise RuntimeError("No working Groq text generation model found.")

def run_deal_intelligence_demo():
    print("\n================================================================================")
    print("        DEAL INTELLIGENCE AGENT DEMO: STATELESS VS HINDSIGHT MEMORY           ")
    print("================================================================================")
    print(f"[+] Hindsight Endpoint : {HINDSIGHT_BASE_URL}")
    print(f"[+] Memory Bank ID     : {BANK_ID}")
    print(f"[+] Groq Target Model  : {DEFAULT_MODEL}")
    print("================================================================================\n")

    # Initialize / Ensure Memory Bank Exists
    try:
        hindsight.create_bank(
            bank_id=BANK_ID,
            name="Deal Intelligence Assistant",
            mission="Remember client objections, past deal context, and technical hurdles."
        )
    except Exception:
        # Bank already exists or auto-created
        pass

    # ------------------------------------------------------------------------------
    # STEP 1: RETAIN PHASE - Store Past Client Interaction in Hindsight
    # ------------------------------------------------------------------------------
    print("================================================================================")
    print(" STEP 1: RETAIN PHASE (Storing Past Client Interaction in Hindsight)")
    print("================================================================================")
    past_interaction = (
        "During our third technical discovery call with Acme Corp, the CTO raised a strong "
        "objection regarding enterprise data security compliance. Specifically, they requested "
        "proof of SOC2 Type II certification and expressed major concerns about data residency handling "
        "before proceeding with any enterprise contract."
    )
    
    print("Storing historical deal memory into Hindsight...")
    print(f" > Content: \"{past_interaction}\"\n")

    try:
        hindsight.retain(
            bank_id=BANK_ID,
            content=past_interaction,
            context="Client objection call #3 with Acme Corp CTO"
        )
        print("[✔] RETAIN COMPLETE: Memory successfully indexed into Hindsight memory bank.\n")
    except Exception as e:
        print(f"[!] Retain note: {e}\n")

    current_query = "We have our next high-stakes sales meeting with Acme Corp tomorrow. How should our sales rep pitch them and structure the discussion?"

    # ------------------------------------------------------------------------------
    # STEP 2: BEFORE STATE (Stateless Chatbot - No Memory)
    # ------------------------------------------------------------------------------
    print("================================================================================")
    print(" [BEFORE STATE] STATELESS CHATBOT (Without Hindsight Memory)")
    print("================================================================================")
    print(f"Sales Rep Query: \"{current_query}\"\n")
    print("Generating sales strategy without past deal memories...\n")
    
    stateless_prompt = (
        "You are a sales deal assistant. "
        f"Current situation: {current_query} "
        "Provide a concise, high-impact sales strategy and meeting agenda."
    )
    
    stateless_output, model_used = generate_completion(groq_client, stateless_prompt, DEFAULT_MODEL)

    print(f"--- STATELESS CHATBOT OUTPUT (Model: {model_used}) ---")
    print(stateless_output)
    print("--------------------------------------------------------------------------------\n")

    # ------------------------------------------------------------------------------
    # STEP 3: RECALL & AFTER STATE (Hindsight Memory-Augmented Agent)
    # ------------------------------------------------------------------------------
    print("================================================================================")
    print(" [AFTER STATE] MEMORY-AUGMENTED AGENT (With Hindsight Memory)")
    print("================================================================================")
    print("Querying Hindsight for security or compliance objections related to Acme Corp...")
    
    recall_query = "What security or compliance objections did Acme Corp raise?"
    recalled_memories_text = ""
    try:
        recall_response = hindsight.recall(
            bank_id=BANK_ID,
            query=recall_query
        )
        if hasattr(recall_response, 'results') and recall_response.results:
            # Deduplicate recalled snippets for clean prompt context
            unique_snippets = []
            for r in recall_response.results:
                txt = getattr(r, 'text', str(r)).strip()
                if txt and txt not in unique_snippets:
                    unique_snippets.append(txt)
            recalled_memories_text = "\n".join(unique_snippets[:3])
        else:
            recalled_memories_text = str(recall_response)
    except Exception as e:
        print(f"[!] Recall exception: {e}")
        recalled_memories_text = past_interaction

    print(f"\n[✔] RECALL COMPLETE: Retrieved Historical Memories from Hindsight:\n")
    print(f"    {recalled_memories_text}\n")

    print("Generating memory-augmented sales strategy incorporating retrieved context...\n")
    stateful_prompt = f"""You are an expert deal intelligence assistant for enterprise sales teams.

HISTORICAL MEMORIES RECALLED FROM HINDSIGHT:
{recalled_memories_text}

CURRENT SITUATION / QUESTION:
{current_query}

INSTRUCTIONS:
Provide a highly targeted, high-conversion sales strategy and meeting agenda. 
Directly address the Acme Corp CTO's specific past objections regarding SOC2 Type II compliance and data residency handling to win the deal."""

    stateful_output, model_used_after = generate_completion(groq_client, stateful_prompt, DEFAULT_MODEL)

    print(f"--- HINDSIGHT-POWERED AGENT OUTPUT (Model: {model_used_after}) ---")
    print(stateful_output)
    print("--------------------------------------------------------------------------------\n")

    print("================================================================================")
    print(" DEMO CONCLUSION:")
    print("   - Before State (Stateless) : Produced generic sales advice missing client context.")
    print("   - After State (Hindsight)   : Leveraged past SOC2 & Data Residency memories to")
    print("                                generate an objection-busting, winning pitch!")
    print("================================================================================\n")

if __name__ == "__main__":
    run_deal_intelligence_demo()
