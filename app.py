import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Controller
from pydantic import BaseModel, SecretStr
from typing import List
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in .env file")
    st.stop()

# Define Pydantic models
class SearchResult(BaseModel):
    title: str
    url: str

class SearchResults(BaseModel):
    results: List[SearchResult]

# Streamlit UI configuration
st.set_page_config(page_title="LIC Policy Explorer", page_icon="🔍")
st.title("LIC Policy Research Agent 🔍")

# Initialize components (cached for performance)
@st.cache_resource
def initialize_components():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Adjusted to a valid model name
        api_key=SecretStr(api_key)
    )
    controller = Controller(output_model=SearchResults)
    return llm, controller

llm, controller = initialize_components()

# Async function to run the agent
async def run_agent(task_query):
    """Async function to run the agent with progress indicators"""
    agent = Agent(
        task=task_query,
        llm=llm,
        controller=controller
    )

    with st.status("Researching policies...", expanded=True) as status:
        st.write("🔍 Starting web research...")
        history = await agent.run()
        
        st.write("📊 Processing results...")
        result = history.final_result()
        
        if result:
            try:
                parsed = SearchResults.model_validate_json(result)
                status.update(label="Research complete!", state="complete")
                return parsed
            except Exception as e:
                st.error(f"Error parsing results: {e}")
                return None
        else:
            st.error("No results found")
            return None

# Function to run async tasks inside Streamlit safely
async def async_run(query):
    loop = asyncio.get_running_loop()
    return await loop.create_task(run_agent(query))

# Streamlit UI elements
query = st.text_input(
    "Enter your research query:",
    value="List of LIC endowment plans with details about Amritbaal policy key features from licindia.in"
)

if st.button("Start Research"):
    if query:
        with st.spinner("Initializing research agent..."):
            results = asyncio.run(async_run(query))  # Uses the correct async handling
        
        if results:
            st.success(f"Found {len(results.results)} results:")
            for idx, res in enumerate(results.results, 1):
                st.markdown(f"""
                ### Result {idx}: {res.title}
                **URL:** [{res.url}]({res.url})
                """)
                st.divider()
        else:
            st.warning("No structured results returned.")
    else:
        st.warning("Please enter a research query first")

st.caption("Note: This may take 1-2 minutes to complete web research and processing")
