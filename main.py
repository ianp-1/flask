from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from typing_extensions import override
from openai import AssistantEventHandler
import requests
import json
import os
from paddleocr import PaddleOCR

# Load environment variables
load_dotenv()

# Configuration
OPEN_LIBRARY_SEARCH_URL = 'https://openlibrary.org/search.json'
GRAPHQL_ENDPOINT = 'https://api.hardcover.app/v1/graphql'
HARDCOVER_BEARER_TOKEN = os.getenv('HARDCOVER_BEARER_TOKEN')
ASSISTANT_ID = "asst_Eexxahpbuh67V4i0Rob9m16Z"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate environment variables
if not HARDCOVER_BEARER_TOKEN:
    raise ValueError("HARDCOVER_BEARER_TOKEN is not set in the environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization='org-FC57og8jDQ11tBkrgIgFX6hB',
    project='proj_0j12yDVX33RfyR5xcnspR60p',
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize PaddleOCR
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')  # Adjust language as needed

# Function to perform OCR using PaddleOCR
def ocr_image(image_path):
    try:
        # Perform OCR on the image
        result = ocr_model.ocr(image_path, cls=True)
        
        # Extract detected text from PaddleOCR results
        text = ''
        for line in result:
            for word_info in line:
                text += word_info[1][0] + ' '
                
        return text.strip()
    except Exception as e:
        return str(e)

# Define the custom EventHandler class for streaming
class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

# Function to clean up the OCR text using OpenAI and parse JSON response
def clean_text_with_openai(text):
    # Create a new thread
    thread = client.beta.threads.create()

    # Send the initial message to the assistant
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=text,
    )

    # Stream the response using the EventHandler class
    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

    # Retrieve the messages in the thread
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # Find the assistant's response
    assistant_message = None
    for message in messages:
        if message.role == 'assistant':
            assistant_message = message
            break

    if not assistant_message:
        return {"error": "Assistant response not found"}

    # Extract the JSON content from the assistant's response
    response_json = None
    for content_part in assistant_message.content:
        if content_part.type == 'text':
            try:
                response_json = json.loads(content_part.text.value)
                break
            except json.JSONDecodeError:
                return {"error": "Failed to decode JSON from assistant response"}

    # Validate the parsed JSON structure
    if response_json and "books" in response_json:
        return response_json["books"]
    else:
        return {"error": "Invalid response schema"}

# Function to perform GraphQL query to Hardcover API
def query_hardcover_graphql(similar_title):
    query = """
    query MyQuery($similarTitle: String!) {
      books(where: {title: {_similar: $similarTitle}} limit:1 order_by: {ratings_count: desc}) {
        title
        contributions {
            author {
                name
            }
        }
        description
        pages
        rating
        ratings_count
        image {
          url
        }
        
      }
    }
    """
    variables = {
        "similarTitle": similar_title
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": HARDCOVER_BEARER_TOKEN  # Correctly formatted header
    }
    payload = {
        "query": query,
        "variables": variables
    }

    try:
        response = requests.post(GRAPHQL_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if 'errors' in data:
            return {"error": data['errors']}
        return data.get('data', {})
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Flask route to process an image and return book info
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image temporarily using tempfile for security
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        file.save(temp.name)

        # Perform OCR using PaddleOCR
        extracted_text = ocr_image(temp.name)
        if not extracted_text:
            return jsonify({"error": "Could not extract text from image"}), 500

    # Clean up the extracted text using OpenAI and parse the JSON response
    cleaned_books = clean_text_with_openai(extracted_text)
    if "error" in cleaned_books:
        return jsonify({"error": cleaned_books["error"]}), 500

    # Initialize list to hold all book information
    all_books_info = []

    for book in cleaned_books:
        title = book.get('title', '')
        author = book.get('author', '')

        if not title:
            continue  # Skip if title is missing

        # Perform GraphQL query to Hardcover API
        graphql_response = query_hardcover_graphql(title)

        if "error" in graphql_response:
            # Optionally handle errors or continue
            return jsonify({"error": graphql_response["error"]}), 500

        # Extract book data from GraphQL response
        hardcover_books = graphql_response.get('books', [])

        all_books_info.append({
            "extracted_title": title,
            "extracted_author": author,
            "hardcover_books": hardcover_books
        })

    return jsonify({
        "extracted_text": extracted_text,
        "books_info": all_books_info
    })

if __name__ == "__main__":
    app.run(port=5000)
