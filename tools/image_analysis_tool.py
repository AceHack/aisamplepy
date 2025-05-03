import os
import base64
import httpx
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel
from models import StructuredImageAnalysisInput, Finding
from .download_utils import download_file_from_url

# Load environment variables early
load_dotenv()

class ImageAnalysisTool(BaseTool):
    name: str = "Structured Image Analyzer"
    description: str = (
        "Analyzes a list of images based on associated descriptions within findings, "
        "provided in a structured JSON input containing 'formid' and 'findings'. "
        "Each finding has a 'description' (used as the prompt) and a single 'image_url'."
    )
    args_schema: type[BaseModel] = StructuredImageAnalysisInput
    _azure_client: AzureOpenAI = None
    _deployment_name: str = None

    def __init__(self):
        super().__init__()
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
        """Downloads image to memory, analyzes it, and returns the result."""
        if not self._azure_client or not self._deployment_name:
             return "Error: Azure client not initialized properly."

        try:
            # 1. Download image bytes using the shared utility
            print(f"Tool: Attempting download via utility for: {image_url}")
            image_bytes = download_file_from_url(image_url)
            print(f"Tool: Image downloaded to memory ({len(image_bytes)} bytes).")

            # 2. Encode image bytes to base64
            base64_image = self._encode_image_bytes(image_bytes)
            # Use a generic but common MIME type; adjust if needed. GPT-4V is flexible.
            image_data = f"data:image/jpeg;base64,{base64_image}"

            # 3. Define the messages for the chat completion
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

            # 4. Call the chat completions API
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

        # Handle specific download errors from the utility
        except httpx.RequestError as e:
            print(f"Tool: Download failed (RequestError): {e}")
            return f"Error downloading image ({image_url}): Network request failed."
        except httpx.HTTPStatusError as e:
            print(f"Tool: Download failed (HTTPStatusError {e.response.status_code}): {e}")
            return f"Error downloading image ({image_url}): Server returned status {e.response.status_code}."
        except Exception as ex:
            # Catch general errors during download, encoding, or API call
            print(f"Tool: Error during analysis for image {image_url}: {ex}")
            return f"Error during analysis for image {image_url}: {ex}"

    def _run(self, formId: str, findings: list) -> str:
        """Processes structured input with formId and findings."""
        print(f"Tool: Received request for formId: {formId}")
        results = {
            "formId": formId,
            "analysis_results": []
        }

        if not isinstance(findings, list):
            return "Error: 'findings' must be a list."

        for i, finding_data in enumerate(findings):
            if not isinstance(finding_data, dict) or 'description' not in finding_data or 'image_url' not in finding_data:
                 print(f"Warning: Skipping invalid finding structure at index {i}: {finding_data}")
                 continue

            description = finding_data['description']
            image_url = finding_data['image_url']

            if not isinstance(image_url, str):
                print(f"Warning: Skipping finding '{description}' due to invalid image_url (not a string): {image_url}")
                continue

            print(f"Tool: Processing finding {i+1}/{len(findings)}: '{description}' with image: {image_url}.")

            analysis = self._analyze_single_image(image_url, description)

            finding_result = {
                "finding_description": description,
                "image_url": image_url,
                "analysis": analysis
            }

            results["analysis_results"].append(finding_result)

        try:
            return json.dumps(results, indent=2)
        except TypeError as e:
            return f"Error serializing results to JSON: {e}"


# --- Test Block ---
if __name__ == '__main__':
    # Imports needed specifically for the test block
    from dotenv import load_dotenv
    import json
    import os # Keep os import for getenv

    load_dotenv()

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

        test_input_data = {
            "formId": "FORM12345",
            "findings": [
                {
                    "description": "Finding 1",
                    "image_url": "https://miro.medium.com/v2/resize:fit:4800/format:webp/1*vdvf-ds54uMEZQO7ZpY9iA.png"
                },
                {
                    "description": "Finding 2",
                    "image_url": "https://www.networkbachelor.com/wp-content/uploads/2021/01/Azure.png"
                }
            ]
        }

        image_tool = ImageAnalysisTool()

        try:
            print(f"Testing StructuredImageAnalysisTool with input:\n{json.dumps(test_input_data, indent=2)}")
            result_str = image_tool._run(**test_input_data)
            print("\n--- Tool Output (JSON String) ---")
            print(result_str)

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
            traceback.print_exc()
# --- End Test Block ---
