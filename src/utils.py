import io
import os
import zipfile

import requests
from pydantic import BaseModel
from tqdm import tqdm

# Used to manually verify title quality
WRITE_TITLES = True

TITLE_SYSTEM_PROMPT = (
    """
# Instructions

The User will provide to you a very long contract. Your job is to create a reasonable title of it.
Keep the title succinct and clear.
You MUST mention the relevant companies involved in the contract, and what the contract purpose and topic is.
{EXTRA_INSTRUCTIONS}
# Format

Your output format should be JSON, matching the following JSON schema.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "thoughts": {
      "type": "array",
      "description": "An array of thoughts or considerations related to the parties involved in the contract and what the contract's purpose is.",
      "items": {
        "type": "string"
      }
    },
    "title": {
      "type": ["string"],
      "description": "A reasonable and succinct title for the contract, including all parties and companies involved, and what the contract's purpose is."
    }
  },
  "required": ["thoughts", "title"],
  "additionalProperties": false
}


Here's an example,
{
  "thoughts": [
    "I read the Licensing Agreement and the parties involved appear to be Company A and Company B."
  ],
  "title": "Licensing Agreement between Company A and Company B",
}

Do NOT output ```json OR ```.
""".strip()
    + "\n"
)

TITLE_USER_PROMPT = (
    """
FILENAME: {FILENAME}
CONTENT:
{CONTENT}
""".strip()
    + "\n"
)


class TitleResponse(BaseModel):
    thoughts: list[str]
    title: str


async def ai_call_openai(model: str, messages: list[dict]) -> str:
    """Simplified ai_call using OpenAI directly."""
    try:
        from openai import AsyncOpenAI
        import os
        
        # Try to load from .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # dotenv not installed, skip
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it as environment variable or in .env file"
            )
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Convert messages format
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        response = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            response_format={"type": "json_object"} if "gpt-4" in model else None,
        )
        
        return response.choices[0].message.content.strip()
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    except Exception as e:
        raise RuntimeError(f"Error calling OpenAI API: {e}")


async def create_title(
    filename: str, text: str, *, extra_instructions: str = ""
) -> str:
    if len(text) > 30000:
        text = text[:10000] + text[-10000:]
    
    # Use OpenAI directly
    messages = [
        {
            "role": "system",
            "content": TITLE_SYSTEM_PROMPT.replace(
                "{EXTRA_INSTRUCTIONS}",
                extra_instructions,
            ),
        },
        {
            "role": "user",
            "content": TITLE_USER_PROMPT.format(
                FILENAME=filename,
                CONTENT=text,
            ),
        },
    ]
    response = await ai_call_openai("gpt-4o-mini", messages)
    
    response = response.strip()
    if response.endswith(",\n}"):
        response = response[:-3] + "\n}"
    title_response = TitleResponse.model_validate_json(response)
    return title_response.title


def download_zip(name: str, url: str, save_path: str, check_path: str) -> None:
    from pathlib import Path
    save_path_obj = Path(save_path)
    check_path_obj = save_path_obj / check_path
    if check_path_obj.exists():
        print(f"{name} dataset already exists. Skipping download.")
        return

    # Streaming download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 4096

    tqdm_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    zip_file = io.BytesIO()

    for data in response.iter_content(block_size):
        tqdm_bar.update(len(data))
        zip_file.write(data)
    tqdm_bar.close()
    zip_file.seek(0)

    if total_size != 0 and tqdm_bar.n != total_size:
        print("ERROR, something went wrong")

    # Extract the contents of the zip file
    save_path_obj.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(save_path_obj)

    if not check_path_obj.exists():
        raise RuntimeError(f"{name} Download Failure! Folder not found.")

    print(f"{name} Download and extraction completed successfully.")

