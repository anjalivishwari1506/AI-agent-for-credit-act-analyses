import os
import json
from google import genai
from google.genai import types

def initialize_gemini_client():
    """Initializes and returns the Gemini client."""
    try:
        client = genai.Client()
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None

def summarize_act(client: genai.Client, extracted_text: str) -> list:
    """Summarizes the Act into 5-10 bullet points (Task 2)."""
    print("--- Starting Task 2: Summarization ---")
    prompt = f"""
    Analyze the following legal text from the Universal Credit Act 2025:
    --- TEXT START ---
    {extracted_text[:15000]} 
    --- TEXT END ---
    
    Provide a summary of the entire Act in 5-10 concise bullet points. Focus specifically on the following elements:
    - Purpose
    - Key definitions
    - Eligibility
    - Obligations
    - Enforcement elements
    
    Return the result as a single JSON list of strings, where each string is a bullet point summary.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro', 
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "array",
                    "items": {"type": "string"}
                }
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in summarization task: {e}")
        return ["Error: Could not generate summary."]

def extract_key_sections(client: genai.Client, extracted_text: str) -> dict:
    """Extracts key sections into a structured JSON report (Task 3)."""
    print("--- Starting Task 3: Key Section Extraction ---")
    prompt = f"""
    Analyze the following legal text from the Universal Credit Act 2025:
    --- TEXT START ---
    {extracted_text[:15000]} 
    --- TEXT END ---
    
    Your task is to extract the key legislative text (verbatim sections or highly accurate summaries) related to the following categories.
    Return the response as a single JSON object matching the required structure. If a section is not found, state "Not Found in Text".

    Categories to extract:
    1. Definitions
    2. Obligations (of the recipient/claimant)
    3. Responsibilities (of the administering authority)
    4. Eligibility (criteria for claimants)
    5. Payments / Entitlements (calculation or structure)
    6. Penalties / Enforcement (provisions and sanctions)
    7. Record-keeping / Reporting (requirements for the authority or claimant)
    """
    
    json_schema = {
        "type": "object",
        "properties": {
            "definitions": {"type": "string", "description": "Key definitions from the Act."},
            "obligations": {"type": "string", "description": "Obligations of the claimant/recipient."},
            "responsibilities": {"type": "string", "description": "Responsibilities of the administering authority."},
            "eligibility": {"type": "string", "description": "Eligibility criteria for claimants."},
            "payments": {"type": "string", "description": "Payment/entitlement calculation structure."},
            "penalties": {"type": "string", "description": "Penalties and enforcement provisions."},
            "record_keeping": {"type": "string", "description": "Record-keeping and reporting requirements."}
        },
        "required": ["definitions", "obligations", "responsibilities", "eligibility", "payments", "penalties", "record_keeping"]
    }
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in key section extraction task: {e}")
        return {"error": "Could not extract key sections."}

def apply_rule_checks(client: genai.Client, extracted_text: str) -> list:
    """Applies the 6 rule checks and returns a structured list (Task 4)."""
    print("--- Starting Task 4: Rule Checks ---")
    rules_to_check = [
        "Act must define key terms",
        "Act must specify eligibility criteria",
        "Act must specify responsibilities of the administering authority",
        "Act must include enforcement or penalties",
        "Act must include payment calculation or entitlement structure",
        "Act must include record-keeping or reporting requirements"
    ]
    
    rule_schema = {
        "type": "object",
        "properties": {
            "rule": {"type": "string", "description": "The specific rule being checked."},
            "status": {"type": "string", "description": "Result: 'pass' or 'fail'."},
            "evidence": {"type": "string", "description": "A short quote or section reference proving the status."},
            "confidence": {"type": "integer", "description": "Confidence level in the status (0-100)."}
        },
        "required": ["rule", "status", "evidence", "confidence"]
    }

    prompt = f"""
    Analyze the following legal text from the Universal Credit Act 2025:
    --- TEXT START ---
    {extracted_text[:15000]} 
    --- TEXT END ---
    
    Check the Act against the following 6 rules. For each rule, determine if it 'pass' or 'fail'.
    Provide a specific, concise piece of text from the Act as 'evidence' to justify your status.
    Assign a 'confidence' score (0-100) for your determination.

    Rules to check: {rules_to_check}
    
    Return the result as a single JSON list of objects, where each object strictly follows the provided schema for a single rule check result.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "array",
                    "items": rule_schema
                }
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in rule checks task: {e}")
        return [{"rule": r, "status": "fail", "evidence": "Error during processing.", "confidence": 0} for r in rules_to_check]

def run_agent(extracted_text: str) -> dict:
    """
    Runs all AI agent tasks and compiles the final structured report.
    """
    client = initialize_gemini_client()
    if not client:
        return {"error": "Failed to initialize Gemini API client. Check your GEMINI_API_KEY."}

    summary_data = summarize_act(client, extracted_text)        # Task 2
    key_sections_data = extract_key_sections(client, extracted_text) # Task 3
    rule_checks_data = apply_rule_checks(client, extracted_text)    # Task 4

    final_report = {
        "report_title": "Universal Credit Act 2025 - AI Agent Analysis Report",
        "task_1_summary": {
            "description": "5-10 point summary focusing on Purpose, Definitions, Eligibility, Obligations, and Enforcement.",
            "summary_points": summary_data
        },
        "task_2_key_sections": key_sections_data,
        "task_3_rule_checks": {
            "description": "Analysis against 6 mandatory legislative requirements.",
            "rule_checks": rule_checks_data
        },
        "disclaimer": "This report is generated by an AI Agent and should be verified against the original Act."
    }

    return final_report