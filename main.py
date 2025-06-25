import pdfplumber
from transformers import pipeline
import openai
import json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import os
import spacy
import re
from typing import Optional, Dict, Any
from spacy.matcher import PhraseMatcher
from spacy.cli import download
from spacy.util import is_package
import sys
import time
import logging
from openai import RateLimitError
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Rate limiting constants
MAX_RETRIES = 3
INITIAL_DELAY = 1  # seconds
MAX_DELAY = 30  # seconds

def exponential_backoff(retry_count: int) -> float:
    """Calculate delay time using exponential backoff."""
    return min(INITIAL_DELAY * (2 ** retry_count), MAX_DELAY)

def make_openai_request(prompt: str, model: str = "gpt-4") -> Any:
    """
    Make an OpenAI API request with rate limiting and retry logic.
    
    Args:
        prompt: The prompt to send to the API
        model: The OpenAI model to use
        
    Returns:
        The API response
        
    Raises:
        Exception: If all retries fail
    """
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response
        except RateLimitError as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries")
                raise
            
            delay = exponential_backoff(retry_count)
            logger.warning(f"Rate limit hit. Waiting {delay:.1f}s before retry {retry_count}/{MAX_RETRIES}")
            time.sleep(delay)
            
    raise Exception("Failed after all retries")

# === PDF TEXT EXTRACTION ===
def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# === CHUNKING ===
def chunk_text(text, max_words=800):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# === JSON EXTRACTION FROM RESPONSE TEXT ===
def generate_output_filename(contract_name, method):
    """
    Generate a filename for output files with timestamp.
    
    Args:
        contract_name: Name of the contract (without extension)
        method: Extraction method (llm, ner, phrases)
        
    Returns:
        Filename string in format: contract_name_YYYYMMDD_HHMMSS_method.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{contract_name}_{timestamp}_{method}.json"

def save_to_output(data, contract_name, method):
    """
    Save extraction results to output directory.
    
    Args:
        data: Dictionary containing the extraction results
        contract_name: Name of the contract (without extension)
        method: Extraction method (llm, ner, phrases)
    """
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    filename = generate_output_filename(contract_name, method)
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {method} results to {output_path}")

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from LLM response, handling code fences and multiple objects.
    Also supports arrays of objects and provides better error feedback.
    
    Returns the first valid JSON object found, even if it's partial.
    """
    # Remove code fences if present
    response = response.replace("```json", "").replace("```", "")
    
    # Find all JSON objects in the response
    # Use a more lenient regex to capture JSON-like structures
    json_objects = re.findall(r'{[^}]*}', response, re.DOTALL)
    
    for json_str in json_objects:
        try:
            # Try to parse the JSON
            data = json.loads(json_str)
            logging.debug(f"Found JSON object: {json.dumps(data, indent=2)}")
            
            # If we get an array, try to merge them into a single object
            if isinstance(data, list):
                merged_data = {}
                for item in data:
                    if isinstance(item, dict):
                        merged_data.update(item)
                data = merged_data
                logging.debug(f"Merged array into single object: {json.dumps(data, indent=2)}")
            
            # If we have a dictionary, return it regardless of keys
            if isinstance(data, dict):
                # Log what keys we found
                logging.debug(f"Found dictionary with keys: {data.keys()}")
                return data
            
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                # Try to fix missing commas
                fixed_json = re.sub(r'(?<=[}\]"])(?=[\[\{\"])', ',', json_str)
                data = json.loads(fixed_json)
                logging.debug(f"Fixed JSON with missing commas: {json.dumps(data, indent=2)}")
                return data
            except json.JSONDecodeError:
                # Try to fix trailing commas
                try:
                    fixed_json = re.sub(r',\s*}', '}', json_str)
                    fixed_json = re.sub(r',\s*\]', ']', fixed_json)
                    data = json.loads(fixed_json)
                    logging.debug(f"Fixed JSON with trailing commas: {json.dumps(data, indent=2)}")
                    return data
                except json.JSONDecodeError:
                    logging.debug(f"JSON parse error: {e} in response: {json_str[:100]}...")
    
    logging.warning(f"No valid JSON object found in response: {response[:200]}...")
    return None

# === LLM METADATA EXTRACTION WITH OUTPUT AGGREGATION AND FUNCTION CALLING ===
# Define suggested generic terms
SUGGESTED_GENERIC_TERMS = [
    "ID",
    "TITLE",
    "TYPE",
    "PAYER NAME",
    "PLAN TYPE",
    "EXTERNAL ID",
    "PROVIDER ENTITY",
    "EFFECTIVE DATE",
    "EXPIRATION DATE",
    "STATUS",
    "PROVIDER IDS",
    "SIZE",
    "CREATED AT",
    "ADDITIONAL INFO SUBMISSION",
    "APPEALS TIMELINE",
    "CLAIM SUBMISSION",
    "INITIAL TERM",
    "INTEREST",
    "MATERIAL BREACH CURE",
    "PROMPT PAY",
    "REFUND SUBMISSION",
    "RENEWAL NOTICE TIMEFRAME",
    "RENEWAL TERM",
    "TERMINATION WITHOUT CAUSE",
    "PAYMENT MODEL",
    "NEGOTIATION STAGE",
    "STATES",
    "PROVIDERS COUNT",
    "CONTRACT TYPE",
    "ESTIMATED ANNUAL VALUE",
    "FINANCIAL TERMS",
    "HAS AMENDMENTS",
    "AMENDMENTS COUNT",
    "LAST AMENDMENT DATE",
    "HAS QUALITY METRICS",
    "RISK LEVEL",
    "DELEGATE STATUS",
    "METADATA CONTEXT",
    "TERM HISTORY",
    "RATE SHEETS",
    "ESCALATORS",
    "PAYMENT TERMS",
    "LATE PAYMENT PENALTIES",
    "VOLUME THRESHOLDS",
    "LESSER OF LANGUAGE",
    "NEW SERVICES",
    "ANTI DOWNCODING",
    "PAYMENT POLICIES",
    "RETROACTIVE DENIALS",
    "CLINICAL GUIDELINES",
    "CLAIMS DATA",
    "VALUE BASED REPORTING",
    "STEERAGE DATA",
    "ELIGIBILITY FILES",
    "ANTI STEERAGE",
    "PRIOR AUTH",
    "CLAIMS ACCURACY",
    "PERFORMANCE GUARANTEES",
    "REMEDIES",
    "JOC",
    "ESCALATION",
    "DISPUTE RESOLUTION",
    "ANNUAL REVIEW",
    "SHARED SAVINGS"
]

# Define ideal terms structure
IDEAL_TERMS = {
    "FINANCIAL": {
        "rate_sheets": "Attach full rate sheets as contract exhibits",
        "escalators": "Multi-year contracts include pre-defined escalators (e.g. 3% annual increases)",
        "payment_terms": "Strict timely payment terms (e.g. clean claims paid within 30 days)",
        "late_payment_penalties": "Include late payment penalties",
        "volume_thresholds": "Volume/utilization thresholds for mid-contract renegotiation",
        "lesser_of_language": "No 'lesser of' language — full contracted rate applies",
        "new_services": "Payment terms include new services automatically unless excluded"
    },
    "POLICY_COMPLIANCE": {
        "anti_downcoding": "Anti-downcoding clause: payer cannot alter CPT codes without clinical justification",
        "payment_policies": "Payment policies frozen as of contract signing",
        "retroactive_denials": "Limit retroactive denials (e.g. no takebacks after 6-12 months)",
        "clinical_guidelines": "Payers must follow standard clinical guidelines (MCG, InterQual, CMS)"
    },
    "DATA_SHARING": {
        "claims_data": "Monthly or quarterly detailed claims data feeds",
        "value_based_reporting": "Value-based program reporting: quarterly performance against metrics",
        "steerage_data": "Share in-network vs out-of-network utilization reports",
        "eligibility_files": "Monthly eligibility files and care coordination data"
    },
    "PAYER_ACCOUNTABILITY": {
        "anti_steerage": "Anti-steerage clause: no incentives that redirect patients away",
        "prior_auth": "Prior authorization turnaround standards (e.g. 5 days routine, 1 day urgent)",
        "claims_accuracy": "Claims accuracy audits permitted; payer must correct underpayments",
        "performance_guarantees": "Performance guarantees (e.g. 99% clean claims paid correctly)",
        "remedies": "Remedies for ongoing underperformance"
    },
    "GOVERNANCE": {
        "joc": "Joint Operating Committee (JOC): quarterly meetings",
        "escalation": "Escalation path: defined chain from operational leads up to executive leadership",
        "dispute_resolution": "Formal dispute resolution: mediation, then arbitration pathway",
        "annual_review": "Annual contract review process through JOC",
        "shared_savings": "Governance for any shared savings or performance programs"
    }
}

def calculate_openai_cost(response, model):
    """
    Calculate the cost of an OpenAI API call based on token usage.
    Uses current GPT-4 pricing as of June 2025.
    """
    # GPT-4 pricing (as of June 2025)
    if model == "gpt-4":
        input_cost_per_token = 0.00003  # $0.03 per 1K tokens
        output_cost_per_token = 0.00006  # $0.06 per 1K tokens
    else:
        raise ValueError(f"Unsupported model: {model}")

    usage = response.usage
    input_cost = usage.prompt_tokens * input_cost_per_token
    output_cost = usage.completion_tokens * output_cost_per_token
    total_cost = input_cost + output_cost
    return total_cost

def validate_json(response: str, section: str) -> Optional[dict]:
    """Basic JSON validation and error handling."""
    try:
        # Remove any non-JSON text
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]  # Remove ```json
        if response.endswith('```'):
            response = response[:-3]  # Remove ```
        
        # Parse JSON
        data = json.loads(response)
        if not isinstance(data, dict):
            logger.warning(f"Invalid {section} response: expected dict, got {type(data)}")
            return None
        return data
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error in {section}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error processing {section}: {e}")
        return None

def load_prompt(prompt_file, **kwargs):
    """
    Load and format a prompt from file with optional keyword arguments.
    """
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt.format(**kwargs)


def extract_metadata_llm_full(text, contract_name):
    """
    Extract metadata from text using LLM prompts.
    """
    chunks = chunk_text(text)
    aggregated_metadata = {
        "generic_metadata": {},
        "suggested_terms": {},
        "ideal_terms": {}
    }
    total_cost = 0.0  # Track total cost of API calls

    for i, chunk in enumerate(chunks):
        try:
            # Load and format prompts
            generic_prompt = load_prompt('prompts/generic_metadata_prompt.txt', TEXT=chunk)
            suggested_prompt = load_prompt('prompts/suggested_terms_prompt.txt', 
                                        TEXT=chunk,
                                        TERMS=', '.join(SUGGESTED_GENERIC_TERMS))
            ideal_prompt = load_prompt('prompts/ideal_terms_prompt.txt', 
                                     TEXT=chunk,
                                     FINANCIAL_TERMS=', '.join(IDEAL_TERMS['FINANCIAL']),
                                     POLICY_COMPLIANCE_TERMS=', '.join(IDEAL_TERMS['POLICY_COMPLIANCE']),
                                     DATA_SHARING_TERMS=', '.join(IDEAL_TERMS['DATA_SHARING']),
                                     PAYER_ACCOUNTABILITY_TERMS=', '.join(IDEAL_TERMS['PAYER_ACCOUNTABILITY']),
                                     GOVERNANCE_TERMS=', '.join(IDEAL_TERMS['GOVERNANCE']))

            # Get generic metadata
            gen_resp = make_openai_request(prompt=generic_prompt, model="gpt-4")
            if gen_resp:
                gen_cost = calculate_openai_cost(gen_resp, "gpt-4")
                total_cost += gen_cost
                logger.info(f"Generic metadata cost: ${gen_cost:.4f}")
                generic_data = validate_json(gen_resp.choices[0].message.content, "generic_metadata")
                if generic_data:
                    aggregated_metadata["generic_metadata"] = generic_data
            else:
                logger.warning(f"No response received for generic metadata in chunk {i+1}")

            # Get suggested terms
            suggested_resp = make_openai_request(prompt=suggested_prompt, model="gpt-4")
            if suggested_resp:
                suggested_cost = calculate_openai_cost(suggested_resp, "gpt-4")
                total_cost += suggested_cost
                logger.info(f"Suggested terms cost: ${suggested_cost:.4f}")
                suggested_data = validate_json(suggested_resp.choices[0].message.content, "suggested_terms")
                if suggested_data:
                    aggregated_metadata["suggested_terms"] = suggested_data
            else:
                logger.warning(f"No response received for suggested terms in chunk {i+1}")

            # Get ideal terms
            ideal_resp = make_openai_request(prompt=ideal_prompt, model="gpt-4")
            if ideal_resp:
                ideal_cost = calculate_openai_cost(ideal_resp, "gpt-4")
                total_cost += ideal_cost
                logger.info(f"Ideal terms cost: ${ideal_cost:.4f}")
                ideal_data = validate_json(ideal_resp.choices[0].message.content, "ideal_terms")
                if ideal_data:
                    aggregated_metadata["ideal_terms"] = ideal_data
            else:
                logger.warning(f"No response received for ideal terms in chunk {i+1}")

        except RateLimitError as e:
            logger.error(f"Rate limit error in chunk {i+1}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")
            continue

        time.sleep(1)

    logger.info(f"Total OpenAI API cost: ${total_cost:.4f}")
    
    # Save results to output file
    save_to_output(aggregated_metadata, contract_name, "llm")
    return aggregated_metadata

# === NER METADATA EXTRACTION ===
def extract_metadata_ner(text):
    try:
        # ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
        ner_pipeline = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", grouped_entities=True)
    except Exception as e:
        logger.error(f"Error extracting NER metadata: {e}")
        return {}

    ner_results = ner_pipeline(text)
    combined_results = {}
    for entity in ner_results:
        if entity['entity_group'] not in combined_results:
            combined_results[entity['entity_group']] = []
        combined_results[entity['entity_group']].append({
            'text': entity['word'],
            'start': entity['start'],
            'end': entity['end'],
            'score': entity['score']
        })

    # Save results to output file
    save_to_output(combined_results, contract_name, "ner")
    return combined_results

# === LOAD SPACY MODEL ===
def load_spacy_model(model_name="en_core_web_sm"):
    if not is_package(model_name):
        print(f"Downloading SpaCy model: {model_name} ...")
        download(model_name)
    return spacy.load(model_name)

# === PHRASE MATCHING USING SPACY ===
def extract_metadata_phrases(text, key_phrases=None):
    if key_phrases is None:
        key_phrases = [
            "effective date",
            "termination date",
            "contract term",
            "payer name",
            "network rate",
            "reimbursement",
            "covered services",
            "scope of services"
        ]

    nlp = load_spacy_model("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(phrase) for phrase in key_phrases]
    matcher.add("METADATA", patterns)

    doc = nlp(text)
    matches = matcher(doc)

    matched_phrases = {}
    for match_id, start, end in matches:
        phrase = doc[start:end].text.lower()
        context = doc[start:end + 5].text  # Extract phrase with some surrounding text
        matched_phrases.setdefault(phrase, []).append(context)

    return matched_phrases

# === MAIN USAGE ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <pdf_file_path>")
        sys.exit(1)

    # Load environment variables
    load_dotenv()

    # Extract text from PDF
    pdf_path = sys.argv[1]
    contract_name = Path(pdf_path).stem  # Get filename without extension
    pdf_text = extract_pdf_text(pdf_path)

    print("== LLM-Based Metadata Extraction ==")
    llm_metadata = extract_metadata_llm_full(pdf_text, contract_name)
    print(json.dumps(llm_metadata, indent=2))
    save_to_output(llm_metadata, contract_name, "llm")

    # print("\n== NER-Based Metadata Extraction ==")
    # ner_metadata = extract_metadata_ner(pdf_text)
    # print(json.dumps(ner_metadata, indent=2))
    # save_to_output(ner_metadata, contract_name, "ner")

    # print("\n== Phrase Matching Metadata Extraction ==")
    # phrases_metadata = extract_metadata_phrases(pdf_text)
    # print(json.dumps(phrases_metadata, indent=2))
    # save_to_output(phrases_metadata, contract_name, "phrases")