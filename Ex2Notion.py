import whisper
import sys
import os
from datetime import datetime
from notion_client import Client
from openai import OpenAI
import warnings
import yt_dlp
import tempfile
import re

# Configuration - Replace with your actual values
NOTION_TOKEN = os.getenv("NOTION_KEY")
NOTION_DATABASE_ID = "23b5ec9c41fa80de8ee3c48ae199e957"
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# Initialize clients
notion = Client(auth=NOTION_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def is_youtube_url(url):
    """Check if the input is a YouTube URL."""
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube\.com|youtu\.be)',
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?(www\.)?youtu\.be/'
    ]
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')

    # Limit length and strip whitespace
    filename = filename.strip()[:100]  # Limit to 100 characters
    return filename


def download_youtube_audio(url, output_dir="downloads"):
    """Download audio from YouTube URL using yt-dlp."""
    print(f"Downloading audio from: {url}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,  # Reduce yt-dlp output noise
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first to extract title
            print("🔍 Extracting video information...")
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown Title')
            video_id = info.get('id', 'unknown')

            print(f"📹 Video Title: {video_title}")
            print(f"🆔 Video ID: {video_id}")

            # Download the audio
            print("⬇️ Downloading and extracting audio...")
            ydl.download([url])

            # Find the downloaded MP3 file
            import glob
            pattern = os.path.join(output_dir, "*.mp3")
            downloaded_files = glob.glob(pattern)

            # Get the most recently downloaded file
            if downloaded_files:
                audio_file = max(downloaded_files, key=os.path.getctime)
                print(f"✅ Downloaded: {os.path.basename(audio_file)}")
                return audio_file, video_title, video_id
            else:
                # Fallback: construct expected filename
                safe_title = sanitize_filename(video_title)
                audio_file = os.path.join(output_dir, f"{safe_title}.mp3")
                if os.path.exists(audio_file):
                    return audio_file, video_title, video_id
                else:
                    raise Exception("Could not find downloaded audio file")

    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Check if the YouTube URL is valid and accessible")
        print("   - Verify FFmpeg is properly installed")
        print("   - Try a different video (some may be restricted)")
        raise


def save_original_transcription(title, content, video_url=None):
    """Save the original transcription to a local file."""
    # Create transcriptions directory if it doesn't exist
    transcriptions_dir = "transcriptions"
    os.makedirs(transcriptions_dir, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = sanitize_filename(title)
    filename = f"{safe_title}_{timestamp}.txt"
    filepath = os.path.join(transcriptions_dir, filename)

    # Save the full transcription
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Title: {title}\n")
        if video_url:
            f.write(f"Source URL: {video_url}\n")
        f.write(f"Transcribed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(content)

    print(f"Original transcription saved to: {filepath}")
    return filepath


def summarize_with_openai(content, title, max_words=300):
    """Use OpenAI to create a concise summary of the transcription."""
    try:
        prompt = f"""Please create a concise summary of the following transcription titled "{title}". 

The summary should:
- Be approximately {max_words} words or less
- Capture the main points and key insights
- Be well-structured and easy to read
- Focus on actionable information and important concepts

Transcription:
{content}

Summary:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" for faster/cheaper option
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that creates concise, well-structured summaries of transcribed content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        summary = response.choices[0].message.content.strip()
        print(f"Summary created ({len(summary)} characters)")
        return summary

    except Exception as e:
        print(f"Error creating summary with OpenAI: {e}")
        # Fallback to truncated original if OpenAI fails
        return content[:1500] + "..." if len(content) > 1500 else content


def split_text_into_chunks(text, max_length=1900):
    """Split text into chunks that fit Notion's character limit with buffer."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    sentences = text.split('. ')
    current_chunk = ""

    for sentence in sentences:
        # Check if adding this sentence would exceed the limit
        test_chunk = current_chunk + sentence + ". "
        if len(test_chunk) <= max_length:
            current_chunk = test_chunk
        else:
            # Save current chunk and start a new one
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def get_database_properties():
    """Get the properties of the Notion database to understand its structure."""
    try:
        database = notion.databases.retrieve(database_id=NOTION_DATABASE_ID)
        properties = database["properties"]

        print("Available database properties:")
        for prop_name, prop_info in properties.items():
            print(f"  - {prop_name}: {prop_info['type']}")

        return properties
    except Exception as e:
        print(f"Error retrieving database properties: {e}")
        return {}


def save_to_notion(title, subject, summary, original_file_path, video_url=None):
    """Save the summarized content to Notion with reference to original file."""
    try:
        # Get database properties to understand structure
        db_properties = get_database_properties()

        # Add reference to original file and video URL at the end
        full_content = f"{summary}\n\n📄 Original transcription saved locally: {os.path.basename(original_file_path)}"
        if video_url:
            full_content += f"\n🔗 Source: {video_url}"

        # Split content into chunks if needed
        text_chunks = split_text_into_chunks(full_content)

        # Create children blocks for each chunk
        children = []
        for chunk in text_chunks:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": chunk
                            }
                        }
                    ]
                }
            })

        # Build properties dynamically based on available properties
        properties = {}

        # Find title property (usually the first one or one with 'title' type)
        title_property = None
        subject_property = None
        subject_property_type = None

        for prop_name, prop_info in db_properties.items():
            if prop_info["type"] == "title":
                title_property = prop_name
            elif prop_name.lower() == "subject" or (
                    prop_info["type"] in ["rich_text", "select"] and not subject_property):
                subject_property = prop_name
                subject_property_type = prop_info["type"]

        date_property = "Date"  # or "Deadline" depending on which you want to use
        # Debug: Show what we found
        print(f"Title property: '{title_property}' (type: title)")
        print(f"Subject property: '{subject_property}' (type: {subject_property_type})")
        print(f"Date property: '{date_property}' (type: date)")

        # Get current date in ISO format for Notion
        current_date = datetime.now().date().isoformat()

        # Set title property
        if title_property:
            properties[title_property] = {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            }
            print(f"✓ Set {title_property} to: '{title}'")

        # Set date property to today's date
        if date_property:
            properties[date_property] = {
                "date": {
                    "start": current_date
                }
            }
            print(f"✓ Set {date_property} to: {current_date}")

        # Set subject property based on its type
        if subject_property and subject_property_type:
            if subject_property_type == "select":
                # For select properties, use the correct format
                properties[subject_property] = {
                    "select": {
                        "name": subject
                    }
                }
                print(f"✓ Set {subject_property} (select) to: '{subject}'")
            elif subject_property_type == "rich_text":
                properties[subject_property] = {
                    "rich_text": [
                        {
                            "text": {
                                "content": subject
                            }
                        }
                    ]
                }
                print(f"✓ Set {subject_property} (rich_text) to: '{subject}'")
            else:
                print(f"⚠ Unsupported property type for {subject_property}: {subject_property_type}")
        else:
            print("⚠ No suitable subject property found")

        print(f"Final properties being sent: {properties}")

        # Create the page
        response = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties=properties,
            children=children
        )

        print(f"Successfully saved to Notion: {response['url']}")
        return response

    except Exception as e:
        print(f"Error saving to Notion: {e}")
        raise


def transcribe_audio(audio_file_path):
    """Transcribe audio file using Whisper."""
    print(f"Transcribing {audio_file_path}...")

    # Load Whisper model
    model = whisper.load_model("base")  # You can use "small", "medium", "large" for better accuracy

    # Transcribe
    result = model.transcribe(audio_file_path)

    print("Transcription completed!")
    return result["text"]


def cleanup_downloaded_file(file_path, keep_audio=False):
    """Clean up downloaded audio file unless user wants to keep it."""
    if not keep_audio and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Cleaned up downloaded file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not remove downloaded file {file_path}: {e}")


def show_help():
    """Display comprehensive help information."""
    help_text = """
🎵 EX2NOTION - AUDIO EXTRACTION TO NOTION SCRIPT 🎵

DESCRIPTION:
    This script downloads audio from YouTube URLs or processes local audio files,
    transcribes them using OpenAI Whisper, creates summaries with OpenAI GPT,
    and saves everything to a Notion database.

SETUP REQUIREMENTS:
    1. Install Python packages:
       pip install whisper openai notion-client yt-dlp

    2. Update configuration in the script:
       - NOTION_TOKEN: Your Notion integration token
       - NOTION_DATABASE_ID: Your Notion database ID
       - OPENAI_API_KEY: Your OpenAI API key
       Note: You must set your OpenAI API key and Notion Integration Tokens 
       as environmental variables before running this script.
       Here's a guide from OpenAI explaining how to do so: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

    3. Ensure FFmpeg is installed (required by yt-dlp and whisper):
       - Windows: Download from https://ffmpeg.org/download.html
       - macOS: brew install ffmpeg
       - Linux: sudo apt install ffmpeg

USAGE:
    python Ex2Notion.py <input> <subject> [title] [options]

ARGUMENTS:
    input       YouTube URL or local audio file path
    subject     Subject/category for the Notion database
    title       (Optional) Custom title for the transcription

OPTIONS:
    --keep-audio    Keep downloaded audio files (for YouTube URLs only)
    --help, -h      Show this help message

EXAMPLES:
    # YouTube video with auto-detected title
    python Ex2Notion.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "Music"

    # YouTube video with custom title
    python Ex2Notion.py "https://youtu.be/dQw4w9WgXcQ" "Music" "Rick Roll Classic"

    # Local audio file
    python Ex2Notion.py "recording.mp3" "Meeting" "Team Standup"

    # Keep downloaded audio file
    python Ex2Notion.py "https://youtube.com/..." "Education" --keep-audio

SUPPORTED FORMATS:
    - YouTube URLs: youtube.com, youtu.be, youtube.com/watch, etc.
    - Local files: .mp3, .mp4, .wav, .m4a, .flac, and most audio formats

OUTPUT:
    - Original transcription saved locally in 'transcriptions/' folder
    - AI-generated summary saved to your Notion database
    - Downloaded audio files saved in 'downloads/' folder (deleted unless --keep-audio)

CONFIGURATION HELP:
    To get your Notion tokens and database ID:
    1. Go to https://www.notion.so/my-integrations
    2. Create a new integration
    3. Copy the "Internal Integration Token"
    4. Share your database with the integration
    5. Get database ID from the database URL

    To get your OpenAI API key:
    1. Go to https://platform.openai.com/api-keys
    2. Create a new API key
    3. Copy the key (starts with 'sk-')

TROUBLESHOOTING:
    - "Permission denied" errors: Check file paths and permissions
    - "Module not found": Run pip install commands above
    - "FFmpeg not found": Install FFmpeg for your operating system
    - Notion errors: Verify your tokens and database permissions
    - YouTube download fails: Check if URL is valid and publicly accessible

VERSION: Ex2Notion v1.0 - Enhanced with yt-dlp integration
AUTHOR: Audio extraction and transcription automation
"""
    print(help_text)


def main(input_source=None, subject=None, custom_title=None, keep_audio=False):
    # If no parameters provided, use command line arguments
    if input_source is None:
        # Check for help flags first
        if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
            show_help()
            sys.exit(0)

        if len(sys.argv) < 3:
            print("❌ Error: Missing required arguments")
            print("\nQuick Usage:")
            print("  python Ex2Notion.py <input> <subject> [title]")
            print("\nFor detailed help:")
            print("  python Ex2Notion.py --help")
            sys.exit(1)

        # Parse arguments
        input_source = sys.argv[1]
        subject = sys.argv[2]
        custom_title = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None

        # Check for flags
        keep_audio = '--keep-audio' in sys.argv

    try:
        # Determine if input is YouTube URL or local file
        if is_youtube_url(input_source):
            print("📺 YouTube URL detected - downloading audio...")
            try:
                audio_file, video_title, video_id = download_youtube_audio(input_source)
                title = custom_title or video_title
                cleanup_needed = True
                source_url = input_source
                print(f"✅ Audio downloaded successfully: {audio_file}")
            except Exception as e:
                print(f"❌ Error downloading from YouTube: {e}")
                sys.exit(1)
        else:
            # Local file
            if not os.path.exists(input_source):
                print(f"❌ Error: File not found: {input_source}")
                sys.exit(1)

            audio_file = input_source
            title = custom_title or os.path.splitext(os.path.basename(input_source))[0]
            cleanup_needed = False
            source_url = None

        print(f"🎵 Processing: {title}")
        print(f"🎤 Audio file to transcribe: {audio_file}")

        # Verify the audio file exists before transcribing
        if not os.path.exists(audio_file):
            print(f"❌ Error: Audio file not found: {audio_file}")
            sys.exit(1)

        # Step 1: Transcribe the audio
        original_transcription = transcribe_audio(audio_file)

        # Step 2: Save original transcription locally
        original_file_path = save_original_transcription(title, original_transcription, source_url)

        # Step 3: Create summary using OpenAI
        print("🤖 Creating summary with OpenAI...")
        summary = summarize_with_openai(original_transcription, title)

        # Step 4: Save summary to Notion
        print("📝 Saving to Notion...")
        save_to_notion(title, subject, summary, original_file_path, source_url)

        # Step 5: Cleanup if needed
        if cleanup_needed and not keep_audio:
            cleanup_downloaded_file(audio_file)

        print("\n✅ Process completed successfully!")
        print(f"- Original transcription: {original_file_path}")
        print(f"- Summary sent to Notion")
        if cleanup_needed and keep_audio:
            print(f"- Downloaded audio kept at: {audio_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()