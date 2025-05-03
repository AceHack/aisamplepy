import os
import httpx
import time
import json
from openai import AzureOpenAI  # Import AzureOpenAI client
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# --- Environment Variable Check ---
AZURE_OPENAI_TRANSCRIPTION_API_KEY = os.getenv("AZURE_OPENAI_TRANSCRIPTION_API_KEY")  # Renamed variable
AZURE_OPENAI_TRANSCRIPTION_ENDPOINT = os.getenv("AZURE_OPENAI_TRANSCRIPTION_ENDPOINT")
AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT = os.getenv("AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT", "gpt-4o-transcribe")  # Get deployment name or default
API_VERSION = "2025-03-01-preview"  # As specified in the URL

if not AZURE_OPENAI_TRANSCRIPTION_API_KEY:  # Renamed variable
    print("Warning: AZURE_OPENAI_TRANSCRIPTION_API_KEY environment variable not set.")  # Renamed variable
if not AZURE_OPENAI_TRANSCRIPTION_ENDPOINT:
    print("Warning: AZURE_OPENAI_TRANSCRIPTION_ENDPOINT environment variable not set.")


class AudioAnalysisInput(BaseModel):
    """Input schema for AudioAnalysisTool."""
    audio_url: str = Field(..., description="The URL of the audio file to analyze.")


class AudioAnalysisTool(BaseTool):
    name: str = "Azure OpenAI Audio Transcriber"  # Updated name
    description: str = (  # Updated description
        "Downloads an audio file from a URL and transcribes it using a specified Azure OpenAI deployment (e.g., gpt-4o-transcribe). "
        "Input must be a JSON object with the key 'audio_url'."
    )
    args_schema: type[BaseModel] = AudioAnalysisInput
    _openai_client: AzureOpenAI = None  # Store the OpenAI client

    def __init__(self):
        super().__init__()
        # Initialize Azure OpenAI Client
        if not AZURE_OPENAI_TRANSCRIPTION_API_KEY or not AZURE_OPENAI_TRANSCRIPTION_ENDPOINT:  # Renamed variable
            raise ValueError(
                "Azure OpenAI environment variables missing (AZURE_OPENAI_TRANSCRIPTION_API_KEY, AZURE_OPENAI_TRANSCRIPTION_ENDPOINT)"  # Renamed variable
            )

        try:
            self._openai_client = AzureOpenAI(
                api_key=AZURE_OPENAI_TRANSCRIPTION_API_KEY,  # Renamed variable
                azure_endpoint=AZURE_OPENAI_TRANSCRIPTION_ENDPOINT,
                api_version=API_VERSION,
            )
            print(f"Tool: Azure OpenAI client initialized (Endpoint: {AZURE_OPENAI_TRANSCRIPTION_ENDPOINT}, Deployment: {AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT}).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI client: {e}")

    def _run(self, audio_url: str) -> str:
        """Downloads audio, sends it to Azure OpenAI for transcription."""
        try:
            from .download_utils import download_file_from_url
        except ImportError:
            from download_utils import download_file_from_url  # Fallback for direct run

        if not self._openai_client:
            return json.dumps({"error": "Azure OpenAI client not initialized properly."})

        print(f"Tool: Received request to analyze audio URL: {audio_url}")
        try:
            # 1. Download audio bytes using the shared utility
            print(f"Tool: Attempting download via utility for: {audio_url}")
            audio_bytes = download_file_from_url(audio_url)
            audio_file_tuple = ("audio.wav", audio_bytes, "audio/wav")  # Assuming WAV, adjust if needed
            print(f"Tool: Audio downloaded to memory ({len(audio_bytes)} bytes).")

            # 2. Transcribe audio using Azure OpenAI client
            print(f"Tool: Sending audio to Azure OpenAI endpoint for transcription (Deployment: {AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT})...")
            start_time = time.time()
            response = self._openai_client.audio.transcriptions.create(
                model=AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT,  # Use the deployment name here
                file=audio_file_tuple
            )
            end_time = time.time()
            print(f"Tool: Transcription received from Azure OpenAI in {end_time - start_time:.2f} seconds.")

            transcription = response.text  # Extract transcription text

            return json.dumps({"transcription": transcription})

        except httpx.RequestError as e:
            print(f"Tool: Download failed (RequestError): {e}")
            return json.dumps({"error": f"Error downloading audio ({audio_url}): Network request failed."})
        except httpx.HTTPStatusError as e:
            print(f"Tool: Download failed (HTTPStatusError {e.response.status_code}): {e}")
            return json.dumps({"error": f"Error downloading audio ({audio_url}): Server returned status {e.response.status_code}."})
        except Exception as e:
            print(f"Tool: Unexpected error during audio analysis for {audio_url}: {e}")
            import traceback
            traceback.print_exc()
            error_message = f"An unexpected error occurred: {e}"
            if hasattr(e, 'status_code'):
                error_message = f"Azure OpenAI API error (Status {e.status_code}): {e.message}"
            elif hasattr(e, 'message'):
                error_message = f"Azure OpenAI API error: {e.message}"

            return json.dumps({"error": error_message})


# --- Test Block ---
if __name__ == '__main__':
    from dotenv import load_dotenv
    import json
    import os
    try:
        from .download_utils import download_file_from_url
    except ImportError:
        from download_utils import download_file_from_url  # Fallback for direct run

    load_dotenv()

    if not AZURE_OPENAI_TRANSCRIPTION_API_KEY or not AZURE_OPENAI_TRANSCRIPTION_ENDPOINT:  # Renamed variable
        print("Error: Required Azure OpenAI environment variables are not set.")
        print("Please ensure AZURE_OPENAI_TRANSCRIPTION_API_KEY and AZURE_OPENAI_TRANSCRIPTION_ENDPOINT are set.")  # Renamed variable
        if not AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT:
            print("Info: AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT not set, using default 'gpt-4o-transcribe'.")
        else:
            print(f"Using deployment: {AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT}")

    else:
        print(f"Azure OpenAI credentials found (Endpoint: {AZURE_OPENAI_TRANSCRIPTION_ENDPOINT}, Deployment: {AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT}). Proceeding with test...")

        test_audio_urls = [
            "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
            "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0061_8k.wav"
        ]

        try:
            audio_tool = AudioAnalysisTool()

            for i, test_audio_url in enumerate(test_audio_urls):
                print(f"\n--- Testing URL {i+1}/{len(test_audio_urls)}: {test_audio_url} ---")
                test_input_data = {"audio_url": test_audio_url}

                try:
                    print(f"Testing AudioAnalysisTool with input:\n{json.dumps(test_input_data, indent=2)}")
                    result_str = audio_tool._run(**test_input_data)
                    print("\n--- Tool Output (JSON String) ---")
                    print(result_str)

                    try:
                        result_dict = json.loads(result_str)
                        print("\n--- Parsed Tool Output (Python Dict) ---")
                        import pprint
                        pprint.pprint(result_dict)
                    except json.JSONDecodeError:
                        print("\nError: Tool output was not valid JSON.")

                except Exception as e:
                    print(f"\nAn error occurred during the tool test for {test_audio_url}: {e}")
                    import traceback
                    traceback.print_exc()

            print("\n--- All Tests Complete ---")
        except ValueError as e:
            print(f"Error initializing tool: {e}")
        except RuntimeError as e:
            print(f"Error initializing tool: {e}")

# --- End Test Block ---
