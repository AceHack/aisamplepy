import os
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import tempfile  # Added for temporary file handling

# Assuming VideoIndexerClient is in the parent directory relative to tools/
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VideoIndexerClient.Consts import Consts
from VideoIndexerClient.VideoIndexerClient import VideoIndexerClient
from tools.download_utils import download_file_from_url  # Import the download utility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

def _get_video_indexer_config() -> Optional[Consts]:
    """Loads Video Indexer configuration from environment variables."""
    try:
        consts = Consts(
            ApiVersion=os.environ["VI_API_VERSION"],
            ApiEndpoint=os.environ["VI_API_ENDPOINT"],
            AzureResourceManager=os.environ["VI_AZURE_RESOURCE_MANAGER"],
            AccountName=os.environ["VI_ACCOUNT_NAME"],
            ResourceGroup=os.environ["VI_RESOURCE_GROUP"],
            SubscriptionId=os.environ["VI_SUBSCRIPTION_ID"],
        )
        return consts
    except KeyError as e:
        logging.error(f"Missing environment variable for Video Indexer configuration: {e}")
        return None
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return None

def analyze_video_with_indexer(video_path_or_url: str, video_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Analyzes a video using Azure AI Video Indexer.

    Uploads the video (local file or URL), waits for indexing to complete,
    and returns the extracted insights including transcription, keywords, and topics.

    Args:
        video_path_or_url: The local file path or public URL of the video.
        video_name: Optional name for the video in Video Indexer. If None, derived from the path/URL.

    Returns:
        A dictionary containing the analysis results (insights) or None if an error occurs.
    """
    consts = _get_video_indexer_config()
    if not consts:
        return None

    client = VideoIndexerClient()
    temp_file_path = None  # Initialize temporary file path variable
    try:
        logging.info("Authenticating with Video Indexer...")
        client.authenticate_async(consts)
        logging.info("Authentication successful.")

        video_id = None
        parsed_url = urlparse(video_path_or_url)

        # Determine if it's a URL or local path
        is_url = bool(parsed_url.scheme and parsed_url.netloc)

        if is_url:
            logging.info(f"Downloading video from URL: {video_path_or_url}")
            try:
                video_bytes = download_file_from_url(video_path_or_url)
                if not video_bytes:
                    logging.error(f"Failed to download video from URL: {video_path_or_url}")
                    return None

                # Create a temporary file to store the downloaded video
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:  # Use a common video suffix
                    temp_file.write(video_bytes)
                    temp_file_path = temp_file.name
                logging.info(f"Video downloaded and saved to temporary file: {temp_file_path}")

                if video_name is None:
                    # Basic name derivation from URL path
                    video_name = os.path.splitext(os.path.basename(parsed_url.path))[0] or "video_from_url"

                logging.info(f"Uploading temporary video file: {temp_file_path}")
                video_id = client.file_upload_async(
                    media_path=temp_file_path,
                    video_name=video_name,
                    # No wait_for_index param in file_upload_async, so we wait separately
                )
                # Wait for indexing explicitly after file upload starts
                if video_id:
                    logging.info(f"Waiting for indexing to complete for video ID: {video_id}...")
                    client.wait_for_index_async(video_id)

            except Exception as download_err:
                logging.error(f"Error during URL download or processing: {download_err}", exc_info=True)
                return None

        else:
            # Assume it's a local file path
            if not os.path.exists(video_path_or_url):
                logging.error(f"Local video file not found: {video_path_or_url}")
                return None
            logging.info(f"Uploading local video file: {video_path_or_url}")
            video_id = client.file_upload_async(
                media_path=video_path_or_url,
                video_name=video_name,
                # No wait_for_index param in file_upload_async, so we wait separately
            )
            # Wait for indexing explicitly after file upload starts
            if video_id:
                logging.info(f"Waiting for indexing to complete for video ID: {video_id}...")
                client.wait_for_index_async(video_id)

        if not video_id:
            logging.error("Failed to upload video or retrieve video ID.")
            return None

        logging.info(f"Video indexing complete for ID: {video_id}. Retrieving insights...")
        insights = client.get_video_async(video_id)

        if not insights:
            logging.error(f"Failed to retrieve insights for video ID: {video_id}")
            return None

        logging.info(f"Successfully retrieved insights for video ID: {video_id}")

        # Extract key insights (transcription, keywords, topics)
        # You can customize this based on the specific insights you need
        extracted_data = {
            "videoId": video_id,
            "transcription": insights.get("videos", [{}])[0].get("insights", {}).get("transcript"),
            "keywords": insights.get("videos", [{}])[0].get("insights", {}).get("keywords"),
            "topics": insights.get("videos", [{}])[0].get("insights", {}).get("topics"),
            "full_insights": insights  # Include full insights for flexibility
        }
        return extracted_data

    except Exception as e:
        logging.error(f"An error occurred during video analysis: {e}", exc_info=True)
        return None
    finally:
        # Clean up the temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logging.info(f"Temporary file deleted: {temp_file_path}")
            except OSError as delete_err:
                logging.error(f"Error deleting temporary file {temp_file_path}: {delete_err}")

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Replace with a valid video URL or local file path for testing
    test_video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4"
    # test_video_path = "/path/to/your/local/video.mp4" # CHANGE THIS

    # Use the URL for testing
    analysis_results = analyze_video_with_indexer(test_video_url)
    if analysis_results:
        print("\n--- Video Analysis Results ---")
        print(f"Video ID: {analysis_results.get('videoId')}")
        # Print snippets of results for brevity
        print("\nTranscription (first 5 lines):")
        transcript = analysis_results.get('transcription')
        if transcript:
            for i, line in enumerate(transcript[:5]):
                print(f"  {line.get('instances')[0].get('start')} - {line.get('instances')[0].get('end')}: {line.get('text')}")
        else:
            print("  No transcription found.")

        print("\nKeywords (first 5):")
        keywords = analysis_results.get('keywords')
        if keywords:
            for i, keyword in enumerate(keywords[:5]):
                print(f"  - {keyword.get('text')}")
        else:
            print("  No keywords found.")

        print("\nTopics (first 5):")
        topics = analysis_results.get('topics')
        if topics:
            for i, topic in enumerate(topics[:5]):
                print(f"  - {topic.get('name')} (Confidence: {topic.get('confidence')})")
        else:
            print("  No topics found.")
        print("\n-----------------------------")
    else:
        print("Video analysis failed.")

