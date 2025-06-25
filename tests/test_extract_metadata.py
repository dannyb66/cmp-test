import unittest
import json
from main import extract_metadata_llm_full, chunk_text, extract_json_from_response
from unittest.mock import patch, MagicMock
import time
from unittest.mock import call

class RateLimitException(Exception):
    """Custom exception to simulate rate limiting."""
    pass

def create_mock_openai_response(content: str):
    """Create a mock OpenAI response object with the given content."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response

class TestExtractMetadata(unittest.TestCase):
    @patch('main.openai.chat.completions.create')
    def test_extract_suggested_terms(self, mock_create):
        # Mock response for generic prompt
        generic_response = create_mock_openai_response(json.dumps({
            "generic_metadata": {
                "contract_type": "Payer Agreement",
                "effective_date": "2024-01-01"
            },
            "suggested_terms": {
                "ID": "CONTRACT123",
                "PAYER NAME": "Blue Cross Blue Shield",
                "EFFECTIVE DATE": "2024-01-01"
            }
        }))

        # Mock response for ideal terms prompt
        ideal_terms_response = create_mock_openai_response(json.dumps({
            "FINANCIAL": {
                "rate_sheets": "Full rate sheets attached as Exhibit A"
            }
        }))

        # Setup mock to return the responses in order
        mock_create.side_effect = [generic_response, ideal_terms_response]

        # Test input text
        test_text = "This is a test contract with ID CONTRACT123 for Blue Cross Blue Shield."

        # Call the function
        result = extract_metadata_llm_full(test_text)

        # Verify the structure and content of the result
        self.assertIn("contract_type", result)
        self.assertIn("effective_date", result)
        self.assertIn("suggested_terms", result)
        self.assertIn("ID", result["suggested_terms"])
        self.assertIn("PAYER NAME", result["suggested_terms"])
        self.assertIn("ideal_terms", result)
        self.assertIn("FINANCIAL", result["ideal_terms"])

    @patch('main.make_openai_request')
    @patch('main.time.sleep')
    def test_rate_limit_handling(self, mock_sleep, mock_make_request):
        """Test that rate limiting is handled correctly with retries."""
        # First two attempts will fail with rate limit, third will succeed
        mock_make_request.side_effect = [
            RateLimitException("Rate limit exceeded"),
            RateLimitException("Rate limit exceeded"),
            create_mock_openai_response(json.dumps({"generic_metadata": {"test": "success"}}))
        ]

        # Test input text
        test_text = "Test contract"

        # Call the function
        result = extract_metadata_llm_full(test_text)

        # Verify that we retried the correct number of times
        self.assertEqual(mock_create.call_count, 3)
        
        # Verify that we slept between retries
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(mock_sleep.call_args_list, [call(2), call(4)])  # 2^1 and 2^2 delays

        # Verify that we got the successful response
        self.assertIn("test", result["generic_metadata"])
        self.assertEqual(result["generic_metadata"]["test"], "success")

    @patch('main.openai.chat.completions.create')
    def test_aggregation_across_chunks(self, mock_create):
        # Create mock responses for generic prompt
        generic_responses = [
            create_mock_openai_response(json.dumps({
                "generic_metadata": {
                    "contract_type": "Payer Agreement",
                    "effective_date": "2024-01-01"
                },
                "suggested_terms": {
                    "ID": "CONTRACT123",
                    "PAYER NAME": "Blue Cross Blue Shield"
                }
            })),
            create_mock_openai_response(json.dumps({
                "generic_metadata": {
                    "termination_clause": "90 days notice"
                },
                "suggested_terms": {
                    "EFFECTIVE DATE": "2024-01-01"
                }
            }))
        ]

        # Create mock responses for ideal terms
        ideal_responses = [
            create_mock_openai_response(json.dumps({
                "FINANCIAL": {
                    "rate_sheets": "Full rate sheets attached as Exhibit A"
                }
            })),
            create_mock_openai_response(json.dumps({
                "FINANCIAL": {
                    "payment_terms": "Claims paid within 30 days"
                }
            }))
        ]

        # Setup mock to return responses in order
        mock_create.side_effect = [
            generic_responses[0],  # First generic response
            ideal_responses[0],    # First ideal terms response
            generic_responses[1],  # Second generic response
            ideal_responses[1]     # Second ideal terms response
        ]

        # Test input text that will be chunked
        test_text = "This is a long test contract with ID CONTRACT123 for Blue Cross Blue Shield. " * 100

        # Call the function
        result = extract_metadata_llm_full(test_text)

        # Verify aggregation across chunks
        self.assertIn("contract_type", result)
        self.assertIn("effective_date", result)
        self.assertIn("termination_clause", result)
        self.assertIn("suggested_terms", result)
        self.assertIn("ID", result["suggested_terms"])
        self.assertIn("PAYER NAME", result["suggested_terms"])
        self.assertIn("EFFECTIVE DATE", result["suggested_terms"])
        self.assertIn("ideal_terms", result)
        self.assertIn("FINANCIAL", result["ideal_terms"])
        self.assertIn("rate_sheets", result["ideal_terms"]["FINANCIAL"])
        self.assertIn("payment_terms", result["ideal_terms"]["FINANCIAL"])

if __name__ == '__main__':
    unittest.main()
