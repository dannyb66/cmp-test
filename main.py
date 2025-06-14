import pdfplumber
from transformers import pipeline
import openai
import json
from dotenv import load_dotenv
import os
import spacy
import re
from spacy.matcher import PhraseMatcher
from spacy.cli import download
from spacy.util import is_package

# Load environment variables from .env
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# === PDF TEXT EXTRACTION ===
def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# === CHUNKING ===
def chunk_text(text, max_words=800):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# === JSON EXTRACTION FROM RESPONSE TEXT ===
def extract_json_from_response(response_text):
    try:
        # Extract the first JSON object found in the text
        json_str_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group(0)
            return json.loads(json_str)
        else:
            print("No JSON object found in response.")
            return {}
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return {}

# === LLM METADATA EXTRACTION WITH OUTPUT AGGREGATION AND FUNCTION CALLING ===
def extract_metadata_llm_full(text):
    chunks = chunk_text(text)
    aggregated_metadata = {}

    for i, chunk in enumerate(chunks[:10]):
        prompt = f"""
        You are analyzing a payer negotiation contract. Based on the text below:
        1. Identify and list the most important metadata fields in this contract.
        2. Extract values for those fields.
        3. Return a well-structured JSON object representing the metadata.

        Example format:
        {{
            "contract_type": "Payer Agreement",
            "effective_date": "2022-01-01",
            "termination_clause": "90 days notice by either party"
        }}        

        Text chunk {i+1}:
        {chunk}
        """
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            # response_format="json",  # Enforce structured JSON output
        )
        response_text = response.choices[0].message.content
        result = extract_json_from_response(response_text)

        # Merge metadata with previous chunks
        for key, value in result.items():
            if key in aggregated_metadata:
                if isinstance(aggregated_metadata[key], list):
                    aggregated_metadata[key].extend(value if isinstance(value, list) else [value])
                else:
                    aggregated_metadata[key] = [aggregated_metadata[key], value]
            else:
                aggregated_metadata[key] = value

    return aggregated_metadata

# === NER METADATA EXTRACTION ===
def extract_metadata_ner(text):
    try:
        # ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
        ner_pipeline = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", grouped_entities=True)
    except Exception as e:
        print("Error loading NER pipeline:", e)
        return {}

    entities = ner_pipeline(text)
    extracted_entities = {}

    for ent in entities:
        label = ent.get('entity_group', 'UNKNOWN')
        value = ent.get('word', '')
        if label not in extracted_entities:
            extracted_entities[label] = []
        extracted_entities[label].append(value)

    return extracted_entities

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
    pdf_text = extract_pdf_text("contract.pdf")

    print("== LLM-Based Metadata Extraction ==")
    llm_metadata = extract_metadata_llm_full(pdf_text)
    print(json.dumps(llm_metadata, indent=2))

    # print("\n== NER-Based Metadata Extraction ==")
    # print(json.dumps(extract_metadata_ner(pdf_text), indent=2))

    # print("\n== Phrase Matching Metadata Extraction ==")
    # print(json.dumps(extract_metadata_phrases(pdf_text), indent=2))