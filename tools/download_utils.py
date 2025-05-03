import os
import httpx
from typing import Optional

def download_file_from_url(url: str) -> bytes:
    """Downloads a file from a URL and returns its content as bytes.

    Args:
        url: The URL of the file to download.

    Returns:
        The content of the file as bytes.

    Raises:
        httpx.RequestError: If there's an issue with the network request.
        httpx.HTTPStatusError: If the server returns an error status code.
        Exception: For other potential errors during download.
    """
    try:
        print(f"Utils: Downloading file from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            file_bytes = response.content
            print(f"Utils: File downloaded successfully ({len(file_bytes)} bytes).")
            return file_bytes

    except httpx.RequestError as e:
        print(f"Utils: HTTP Request Error downloading file ({url}): {e}")
        raise httpx.RequestError(f"Error during network request for {url}: {e}", request=e.request) from e
    except httpx.HTTPStatusError as e:
        print(f"Utils: HTTP Status Error downloading file ({url}): {e.response.status_code} - {e.response.reason_phrase}")
        raise httpx.HTTPStatusError(f"Server returned error for {url}: {e.response.status_code} {e.response.reason_phrase}", request=e.request, response=e.response) from e
    except Exception as e:
        print(f"Utils: General Error processing file ({url}): {e}")
        raise Exception(f"An unexpected error occurred while downloading {url}: {e}") from e
