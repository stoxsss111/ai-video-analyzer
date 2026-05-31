import os
import time
import tempfile
import streamlit as st

from google import genai
from google.genai import types

from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Google GenAI client for Vertex AI
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "fleet-impact-493909-j6")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

client = genai.Client(
    vertexai=True,
    project=project_id,
    location=location
)

st.set_page_config(
    page_title="Multimodal AI Agent - Chat & Video Analysis",
    page_icon="🎥",
    layout="wide",
)

# Custom CSS for premium UI
st.markdown("""
    <style>
        .stChatFloatingInputContainer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 1rem;
            z-index: 100;
            max-width: 100% !important;
            width: 100% !important;
        }
        .main {
            margin-bottom: 100px;
        }
        .stChatMessage {
            max-width: 100% !important;
            width: 100% !important;
            margin: 1rem 0;
        }
        .stChatInputContainer {
            max-width: 100% !important;
            padding: 0 1rem;
        }
        .stChatInput {
            max-width: 100% !important;
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

# Application Title and Header
st.title("AI Video Analyzer & Chat Agent 🤖🎥")
st.header("Powered by Gemini 2.5 Flash on Vertex AI & DuckDuckGo")

def initialize_session_state():
    """Initialize all session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_video_path" not in st.session_state:
        st.session_state.processed_video_path = None
    if "uploaded_video_name" not in st.session_state:
        st.session_state.uploaded_video_name = None
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = time.time()

initialize_session_state()

def auto_scroll():
    """Auto-scroll to the bottom of the chat"""
    if st.session_state.chat_history:
        js = """
        <script>
            window.scrollTo(0, document.body.scrollHeight);
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)

def check_session_timeout():
    """Check if session has timed out (1 hour of inactivity)"""
    if time.time() - st.session_state.last_activity > 3600:
        st.session_state.clear()
        st.experimental_rerun()
    st.session_state.last_activity = time.time()

def process_video(file):
    """Process uploaded video file and save it locally for analysis"""
    try:
        # Check if we've already processed this video
        if (st.session_state.uploaded_video_name == file.name and 
            st.session_state.processed_video_path is not None):
            return True

        # Save the uploaded file to a persistent folder in the workspace
        temp_dir = os.path.join(os.getcwd(), "temp_videos")
        os.makedirs(temp_dir, exist_ok=True)
        
        video_path = os.path.join(temp_dir, f"video_{int(time.time())}.mp4")
        with open(video_path, "wb") as f:
            f.write(file.read())

        st.session_state.processed_video_path = video_path
        st.session_state.uploaded_video_name = file.name
        st.success("Video processed successfully for analysis! 🎉")
        return True
    except Exception as e:
        st.error(f"Video processing error: {e}")
        return False

def generate_response(query):
    """Generate AI response using video content and external knowledge via Vertex AI"""
    try:
        if not st.session_state.processed_video_path:
            return "Please upload a video first."

        with open(st.session_state.processed_video_path, "rb") as f:
            video_bytes = f.read()

        # Create inline video part for Vertex AI Gemini 2.5 Flash
        video_part = types.Part.from_bytes(
            data=video_bytes,
            mime_type="video/mp4"
        )

        # Build list of contents starting with the video part
        contents = [video_part]

        # Add chat history to retain context
        for msg in st.session_state.chat_history:
            if msg["user"].startswith("🔍 Web search for:"):
                continue
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["user"])]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg["ai"])]))

        # Add the current prompt
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

        system_instruction = (
            "You are a professional AI video analyzer. Answer the user's questions about the uploaded video "
            "based on its content and your analysis. If the question is not about the video, use your general knowledge. "
            "Be concise, structured, and informative. Always reply in English."
        )

        with st.spinner("Analyzing video..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.4,
                )
            )

        ai_response = response.text
        if not ai_response or len(ai_response.strip()) < 10:
            return perform_web_search(query)
        return ai_response
    except Exception as e:
        return f"An error occurred: {e}"

def perform_web_search(query):
    """Perform web search using DuckDuckGo"""
    try:
        with st.spinner("Searching the web..."):
            search = DuckDuckGoSearchRun()
            search_prompt = f"Search for information about: {query}"
            response = search.invoke(search_prompt)
            if response:
                return response
            else:
                return f"No relevant results found for '{query}'."
    except Exception as e:
        return f"Search error: {e}"

# Check session timeout
check_session_timeout()

# Main UI Components
video_file = st.file_uploader(
    "Upload a video file to Analyse",
    type=["mp4", "mov", "avi", "mkv"],
)

if video_file:
    # Only process if it's a new video or not processed yet
    if (st.session_state.uploaded_video_name != video_file.name or 
        st.session_state.processed_video_path is None):
        process_video(video_file)
    
    # Display video and chat interface
    if st.session_state.processed_video_path:
        st.video(st.session_state.processed_video_path, format="video/mp4", start_time=0)
    
    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(message["user"])
            with st.chat_message("assistant"):
                st.write(message["ai"])
    
    # Input interface
    col1, col2 = st.columns([5, 1])
    with col1:
        prompt = st.chat_input("Ask anything about the video...")
    with col2:
        search_button = st.button("Web Search 🔍")

    # Handle user input
    if prompt:
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                response = generate_response(prompt)
                message_placeholder.markdown(response)
            st.session_state.chat_history.append({"user": prompt, "ai": response})
            auto_scroll()

    # Handle web search
    if search_button and st.session_state.chat_history:
        last_query = st.session_state.chat_history[-1]["user"]
        with chat_container:
            with st.chat_message("assistant"):
                search_results = perform_web_search(last_query)
                st.markdown(search_results)
                st.session_state.chat_history.append(
                    {"user": f"🔍 Web search for: {last_query}", 
                     "ai": search_results}
                )
                auto_scroll()

else:
    st.info("Upload a video file to begin analysis.")