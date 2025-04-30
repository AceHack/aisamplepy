import os
import base64
import httpx
import magic
import json # Import json for structured output
from openai import AzureOpenAI
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel
# Import the new structured input schema
from models import StructuredImageAnalysisInput, Finding

# Load environment variables early
load_dotenv()

class ImageAnalysisTool(BaseTool):
    name: str = "Structured Image Analyzer"
    description: str = (
        "Analyzes a list of images based on associated descriptions within findings, "
        "provided in a structured JSON input containing 'formid' and 'findings'. "
        # Updated description to reflect single image URL per finding
        "Each finding has a 'description' (used as the prompt) and a single 'image_url'."
    )
    # Use the new schema for input validation
    args_schema: type[BaseModel] = StructuredImageAnalysisInput
    _azure_client: AzureOpenAI = None
    _deployment_name: str = None

    def __init__(self):
        super().__init__()
        # Initialize Azure Client once
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([api_key, endpoint, self._deployment_name, api_version]):
            raise ValueError("Azure OpenAI environment variables missing (KEY, ENDPOINT, DEPLOYMENT_NAME, API_VERSION)")

        try:
            self._azure_client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI client: {e}")

    def _encode_image_bytes(self, image_bytes):
        """Encodes image bytes to a base64 string."""
        return base64.b64encode(image_bytes).decode('utf-8')

    def _analyze_single_image(self, image_url: str, prompt: str) -> str:
        """Helper function to analyze a single image URL."""
        if not self._azure_client or not self._deployment_name:
             return "Error: Azure client not initialized properly."

        image_bytes = None
        mime_type = "application/octet-stream"

        # Download and process the image
        try:
            print(f"Tool: Downloading image from: {image_url}")
            # Add a User-Agent header to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            # Use timeout and follow redirects
            with httpx.Client(follow_redirects=True) as http_client:
                response = http_client.get(image_url, headers=headers, timeout=30.0) # Added headers and timeout
                response.raise_for_status()
                image_bytes = response.content
                # Use python-magic to determine MIME type
                try:
                    mime_type = magic.from_buffer(image_bytes, mime=True)
                    print(f"Tool: Image downloaded. MIME type: {mime_type}")
                except ImportError:
                    print("Warning: python-magic not installed. Falling back to default MIME type.")
                except Exception as magic_e:
                    print(f"Warning: Error determining MIME type with magic: {magic_e}. Falling back.")

        except httpx.RequestError as e:
            # More specific error logging for request errors
            print(f"Tool: HTTP Request Error downloading image ({image_url}): {e}")
            return f"Error downloading image ({image_url}): {e}"
        except httpx.HTTPStatusError as e:
            # More specific error logging for status errors (like 404)
            print(f"Tool: HTTP Status Error downloading image ({image_url}): {e.response.status_code} - {e.response.reason_phrase}")
            return f"Error downloading image ({image_url}): {e.response.status_code} - {e.response.reason_phrase}"
        except Exception as e:
            # General error logging
            print(f"Tool: General Error processing image ({image_url}): {e}")
            return f"Error processing image ({image_url}): {e}"

        if not image_bytes:
            return f"Error: Could not retrieve image bytes from {image_url}"

        # Encode and format for API
        base64_image = self._encode_image_bytes(image_bytes)
        image_data = f"data:{mime_type};base64,{base64_image}"

        # Define the messages for the chat completion
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data, "detail": "auto"}
                    },
                ],
            }
        ]

        # Call the chat completions API
        try:
            print(f"Tool: Sending request to Azure OpenAI for image: {image_url}...")
            response = self._azure_client.chat.completions.create(
                model=self._deployment_name,
                messages=messages,
                max_tokens=1024,
            )
            if response.choices:
                result = response.choices[0].message.content
                print(f"Tool: Received response from Azure OpenAI for image: {image_url}.")
                return result
            else:
                return f"Error: No response content received from Azure OpenAI for image: {image_url}."
        except Exception as ex:
            return f"Error during Azure OpenAI API call for image {image_url}: {ex}"

    def _run(self, formId: str, findings: list) -> str: # Renamed formid to formId
        """Processes structured input with formId and findings.""" # Updated docstring
        print(f"Tool: Received request for formId: {formId}") # Updated print statement
        results = {
            "formId": formId, # Updated key
            "analysis_results": []
        }

        # Validate findings structure (basic check, Pydantic handles deeper validation)
        if not isinstance(findings, list):
            return "Error: 'findings' must be a list."

        for i, finding_data in enumerate(findings):
            # Pydantic should have already validated this, but double-check
            # Check for 'image_url' instead of 'image_urls'
            if not isinstance(finding_data, dict) or 'description' not in finding_data or 'image_url' not in finding_data:
                 print(f"Warning: Skipping invalid finding structure at index {i}: {finding_data}")
                 continue

            description = finding_data['description']
            # Get single image_url
            image_url = finding_data['image_url']

            # Validate image_url is a string
            if not isinstance(image_url, str):
                print(f"Warning: Skipping finding '{description}' due to invalid image_url (not a string): {image_url}")
                continue

            print(f"Tool: Processing finding {i+1}/{len(findings)}: '{description}' with image: {image_url}.")

            # Analyze the single image
            analysis = self._analyze_single_image(image_url, description)

            # Store result for this finding
            finding_result = {
                "finding_description": description,
                "image_url": image_url,
                "analysis": analysis
            }

            results["analysis_results"].append(finding_result)

        # Return results as a JSON string for the agent
        try:
            return json.dumps(results, indent=2)
        except TypeError as e:
            return f"Error serializing results to JSON: {e}"


# --- Test Block --- (Updated)
if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    import json # Import json for test input

    # Load environment variables for Azure credentials
    load_dotenv()

    # Ensure Azure credentials are set (example check)
    if not (
        os.getenv("AZURE_OPENAI_ENDPOINT") and
        os.getenv("AZURE_OPENAI_API_KEY") and
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") and
        os.getenv("AZURE_OPENAI_API_VERSION")
    ):
        print("Error: Azure OpenAI environment variables are not set.")
        print("Please ensure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME, and AZURE_OPENAI_API_VERSION are set.")
    else:
        print("Azure credentials found. Proceeding with test...")

        # Sample structured data for testing (updated key to formId and content)
        test_input_data = {
            "formId": "FORM12345", # Changed key from formid to formId
            "findings": [
                {
                    "description": "Finding 1", # Updated description
                    "image_url": "https://miro.medium.com/v2/resize:fit:4800/format:webp/1*vdvf-ds54uMEZQO7ZpY9iA.png" # Updated URL
                },
                { # Added second finding from main.py
                    "description": "Finding 2",
                    "image_url": "https://www.networkbachelor.com/wp-content/uploads/2021/01/Azure.png"
                }
            ]
        }

        # Instantiate the tool
        image_tool = ImageAnalysisTool()

        # Run the tool's logic using the structured input
        try:
            print(f"Testing StructuredImageAnalysisTool with input:\\n{json.dumps(test_input_data, indent=2)}")
            # Pass the structured data directly to the run method
            # CrewAI/Pydantic handles the parsing based on args_schema
            # Use **test_input_data to unpack the dictionary into arguments for _run
            result_str = image_tool._run(**test_input_data)
            print("\n--- Tool Output (JSON String) ---")
            print(result_str)

            # Optionally parse the JSON string back to a dictionary for inspection
            try:
                result_dict = json.loads(result_str)
                print("\n--- Parsed Tool Output (Python Dict) ---")
                import pprint
                pprint.pprint(result_dict)
            except json.JSONDecodeError:
                print("\nError: Tool output was not valid JSON.")

            print("\n--- Test Complete ---")
        except Exception as e:
            print(f"\nAn error occurred during the tool test: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
# --- End Test Block ---
