import json
import os
import re
import sys
from datetime import datetime

from halo import Halo
from openai import OpenAI

system_description = """You are a translator of the game Where Winds Meets. You master Chinese and Vietnamese languages.
Translate the following Chinese text to Vietnamese accurately, not missing any Chinese word, maintaining the game's tone and context.
Just response as json, do not add any extra explanation like ```
"""

auth_api_key = os.getenv("OR_API_KEY", "sk-or-...")

# Maximum size for each chunk (4MB to be safe, API limit is 8MB)
MAX_CHUNK_SIZE_MB = 4


def replace_filename_pattern(filename, out_prefix):
    pattern = r"^(.+?)_(\d+)\.json$"
    match = re.match(pattern, filename)

    if match:
        number = match.group(2)
        return f"p{out_prefix}_{number}.json"

    return filename


def estimate_text_size_mb(text):
    """Estimate text size in MB"""
    return len(text.encode('utf-8')) / (1024 * 1024)


def split_json_into_chunks(data, max_size_mb=MAX_CHUNK_SIZE_MB):
    """Split JSON dict into smaller chunks based on size"""
    chunks = []
    current_chunk = {}
    current_size = 0
    
    for key, value in data.items():
        item_json = json.dumps({key: value}, ensure_ascii=False)
        item_size = estimate_text_size_mb(item_json)
        
        # If single item is too large, skip it with warning
        if item_size > max_size_mb:
            print(f"  ⚠️  Warning: Item '{key[:50]}...' is too large ({item_size:.2f}MB), skipping...")
            continue
        
        # If adding this item exceeds limit, start new chunk
        if current_size + item_size > max_size_mb and current_chunk:
            chunks.append(current_chunk)
            current_chunk = {}
            current_size = 0
        
        current_chunk[key] = value
        current_size += item_size
    
    # Add last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def translate_chunk(client, spinner, chunk_data, chunk_num, total_chunks):
    """Translate a single chunk of JSON data"""
    chunk_json = json.dumps(chunk_data, ensure_ascii=False, indent=2)
    chunk_size = estimate_text_size_mb(chunk_json)
    
    spinner.text = f"Chunk {chunk_num}/{total_chunks} ({chunk_size:.2f}MB, {len(chunk_data)} entries)..."
    
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "X-Title": "WWM Locale Tool",
            },
            model="google/gemini-2.5-flash-lite-preview-09-2025",
            messages=[
                {
                    "role": "system",
                    "content": system_description,
                },
                {"role": "user", "content": chunk_json},
            ],
            stream=True,
        )
    except Exception as e:
        spinner.fail(f"Network error on chunk {chunk_num}: {e}")
        return None
    
    translated_text = ""
    try:
        for chunk in completion:
            if chunk.choices[0].delta.content:
                resp_content = chunk.choices[0].delta.content
                translated_text += resp_content
                spinner.text = f"Chunk {chunk_num}/{total_chunks} > {resp_content.replace(chr(10), '').strip()[:30]}..."
    except Exception as e:
        spinner.fail(f"Streaming error on chunk {chunk_num}: {e}")
        return None
    
    # Extract JSON from response
    translated_text = translated_text.strip()
    i1 = translated_text.find("{")
    i2 = translated_text.rfind("}")
    if i1 != -1 and i2 != -1:
        translated_text = translated_text[i1 : i2 + 1]
    
    try:
        return json.loads(translated_text)
    except json.JSONDecodeError as e:
        spinner.fail(f"Failed to parse JSON response for chunk {chunk_num}: {e}")
        return None


def translate_text(spinner, input_file, output_file):
    started_at = os.times()

    output_file = output_file or os.path.join(
        os.path.dirname(input_file), "translated_" + os.path.basename(input_file)
    )

    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return -1

    # Read and parse JSON
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            print("Input file is empty.")
            return -1
        
        # Check if it's JSON format
        data = json.loads(content)
        if not isinstance(data, dict):
            print("Input file must be a JSON object (dict)")
            return -1
        
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        return -1

    # Check file size
    file_size_mb = estimate_text_size_mb(content)
    spinner.info(f"File size: {file_size_mb:.2f}MB, {len(data)} entries")
    
    # Split into chunks if too large
    if file_size_mb > MAX_CHUNK_SIZE_MB:
        spinner.info(f"File too large, splitting into chunks (max {MAX_CHUNK_SIZE_MB}MB each)...")
        chunks = split_json_into_chunks(data, MAX_CHUNK_SIZE_MB)
        spinner.info(f"Split into {len(chunks)} chunks")
    else:
        chunks = [data]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=auth_api_key,
    )

    # Translate each chunk
    translated_data = {}
    for i, chunk in enumerate(chunks, 1):
        spinner.start(f"Translating chunk {i}/{len(chunks)}...")
        result = translate_chunk(client, spinner, chunk, i, len(chunks))
        
        if result is None:
            spinner.fail(f"Failed to translate chunk {i}/{len(chunks)}")
            return -1
        
        translated_data.update(result)
        spinner.succeed(f"Chunk {i}/{len(chunks)} completed ({len(result)} entries)")

    # Save translated result
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    processed_at = os.times()
    elapsed = processed_at[4] - started_at[4]
    
    return elapsed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python trans-vi.py <source folder> <output folder>")
        sys.exit(1)

    missing_folder = sys.argv[1]
    output_folder = sys.argv[2]

    spinner = Halo(text="Processing", spinner="dots")
    spinner.start()

    files = [f for f in os.listdir(missing_folder) if f.endswith(".json")]
    
    if not files:
        spinner.fail("No JSON files found in input folder!")
        sys.exit(1)

    now = datetime.now()
    run_at = (
        f"{now.strftime('%y')}"
        f"{now.strftime('%V')}"
        f"{now.strftime('%u')}"
        f"{now.strftime('%H')}"
        f"{now.strftime('%M')}"
    )

    success_count = 0
    failed_count = 0

    for idx, filename in enumerate(files):
        new_filename = replace_filename_pattern(filename, run_at)

        input_file = os.path.join(missing_folder, filename)

        if new_filename == filename:
            output_file = os.path.join(output_folder, f"t{run_at}_{filename}")
        else:
            output_file = os.path.join(output_folder, new_filename)

        spinner.info(f"[{idx + 1}/{len(files)}] Translating {filename}")
        spinner.start("Preparing...")
        
        processed_time = translate_text(spinner, input_file, output_file)
        
        if processed_time < 0:
            spinner.fail(f"[{idx + 1}/{len(files)}] Translation FAILED for {filename}")
            failed_count += 1
            # Continue with next file instead of exiting
        else:
            msg = f"[{idx + 1}/{len(files)}] Translation completed in {processed_time:.2f} seconds."
            spinner.succeed(msg)
            success_count += 1

    # Summary
    print("\n" + "="*60)
    print(f"Translation Summary:")
    print(f"  ✅ Success: {success_count}/{len(files)}")
    print(f"  ❌ Failed:  {failed_count}/{len(files)}")
    print("="*60)
    
    # Exit with error if any file failed
    if failed_count > 0:
        sys.exit(1)
