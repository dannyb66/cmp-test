#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
from dotenv import load_dotenv


class EventHandler(AssistantEventHandler):
    """Event handler to stream assistant responses and accumulate text."""

    def __init__(self) -> None:
        super().__init__()
        self.content = ""

    @override
    def on_text_delta(self, delta, snapshot) -> None:
        print(delta.value, end="", flush=True)
        self.content += delta.value or ""

    @override
    def on_end(self) -> None:
        print()  # final newline


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
    if not api_key or not assistant_id:
        print("Error: OPENAI_API_KEY and OPENAI_ASSISTANT_ID must be set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Hardcoded test inputs: list of metadata questions to run sequentially
    file_id = "file-9nUm8RJwgKtj97o78iCz5E"
    questions = [
        ("payer_name", "What is the name of the payer or health plan entity? (as listed under the PAYER section)?"),
        ("payer_address", "What is the address for notice of the payer? (as listed under the PAYER section)? List them if there are multiple."),
        ("payer_email", "What is the email address for notice of the payer? (as listed under the PAYER section)? List them if there are multiple."),
        ("payer_contact_number", "What is the contact number for notice of the payer? (as listed under the PAYER section)? List them if there are multiple."),
        ("provider_entity", "Who is the provider or healthcare organization in this contract? (as listed under the PROVIDER section)? List them if there are multiple."),
        ("provider_address", "What is the address for notice of the provider (as listed under the PROVIDER section)? List them if there are multiple."),
        ("provider_email", "What is the email address for notice of the provider (as listed under the PROVIDER section)? List them if there are multiple."),
        ("provider_contact_number", "What is the contact number for notice of the provider (as listed under the PROVIDER section)? List them if there are multiple."),
        ("provider_npi", "What is the National Provider ID (NPI) of the provider entity? List them if there are multiple."),
        ("provider_tin", "What is the Tax Identification Number (TIN) of the provider entity? List them if there are multiple."),
        ("external_id", "What is the external or third-party reference ID for the contract? List them if there are multiple."),
        ("effective_date", "What is the contract effective date? Return the answer in MM-DD-YYYY format."),
    ]
    answers: dict[str, str] = {}
    for field, qtext in questions:
        user_message = f"{file_id}\n{field},{qtext}"
        # start a fresh thread for each question to isolate context
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)

        handler = EventHandler()
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant_id,
            event_handler=handler,
        ) as stream:
            stream.until_done()

        # If the assistant returned a JSON object with multiple keys, extract only the asked field
        raw = handler.content.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and field in parsed:
                answers[field] = parsed[field]
            else:
                answers[field] = raw
        except json.JSONDecodeError:
            answers[field] = raw

    # Save all responses to a JSON file
    out = {"file_id": file_id, "answers": answers}
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{file_id}_{timestamp}_assistant.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # Print combined JSON result
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()