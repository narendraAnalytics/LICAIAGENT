from pathlib import Path
from agno.agent import Agent
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.google import GeminiEmbedder
from agno.models.google import Gemini
import os
import requests
from agno.vectordb.search import SearchType
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance
import io
import pytesseract
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Function to extract content from PDFs with improved table/image handling
def extract_pdf_content(pdf_url):
    try:
        response = requests.get(pdf_url)
        pdf_content = response.content
        text_content = ""

        # Text and table extraction with pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract text
                    text = page.extract_text()
                    if text:
                        text_content += f"\nPage {page_num + 1} Text:\n{text}\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        text_content += f"\nPage {page_num + 1} Table {table_num + 1}:\n"
                        for row in table:
                            text_content += "|".join(str(cell).replace('\n', ' ') if cell else "N/A" for cell in row) + "\n"
                        text_content += "\n"
                    
                    # Fallback to OCR if no text found
                    if not text or not text.strip():
                        img = page.to_image(resolution=300)
                        image = ImageEnhance.Contrast(img.original).enhance(2.0)
                        ocr_text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
                        if ocr_text.strip():
                            text_content += f"\nPage {page_num + 1} OCR Text:\n{ocr_text}\n"
                        else:
                            text_content += f"\nPage {page_num + 1}: No text found in image\n"

                except Exception as page_error:
                    logger.error(f"Error processing page {page_num + 1}: {str(page_error)}")
                    text_content += f"\nError processing page {page_num + 1}\n"

        return text_content
    except Exception as e:
        logger.error(f"PDF processing error for {pdf_url}: {str(e)}")
        return f"Error processing PDF: {str(e)}"

# Enhanced WebsiteKnowledgeBase with improved product and PDF link following
class EnhancedWebsiteKnowledgeBase(WebsiteKnowledgeBase):
    def process_url(self, url):
        # Check if URL contains ".pdf" anywhere
        if ".pdf" in url.lower():
            logger.info(f"Processing PDF: {url}")
            try:
                content = extract_pdf_content(url)
                if content and content.strip():
                    self.add_document(url, content)
                    logger.info(f"Successfully processed PDF: {url}")
                else:
                    logger.warning(f"Empty content from PDF: {url}")
            except Exception as e:
                logger.error(f"Failed to process PDF {url}: {str(e)}")
        else:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                if soup:
                    # Log all links for debugging
                    links = soup.find_all("a", href=True)
                    logger.info(f"Found {len(links)} links on {url}")
                    for link in links:
                        href = link.get("href")
                        absolute_url = self._make_absolute_url(url, href)
                        logger.info(f"Considering link: {absolute_url}")
                        # Follow product name links and PDF links
                        if self._is_product_link(soup, link) or ".pdf" in href.lower():
                            # Additional check for PDF links labeled "Sales Brochure"
                            if "sales brochure" in link.get_text(strip=True).lower():
                                logger.info(f"Prioritizing Sales Brochure PDF link: {absolute_url}")
                            logger.info(f"Following link: {absolute_url}")
                            self.process_url(absolute_url)
                    
                    # Process normal page content
                    self._extract_and_store_content(url, soup)
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")

    def _make_absolute_url(self, base_url, link):
        return urljoin(base_url, link)

    def _is_product_link(self, soup, link):
        # Improved logic to identify product name links within a table
        parent_table = link.find_parent("table")
        if parent_table:
            # Check if the link is in the "Product Name" column (first column by index)
            parent_row = link.find_parent("tr")
            if parent_row:
                cells = parent_row.find_all("td")
                if cells and cells[0] == link.parent:  # First column is Product Name
                    return True
            # Alternative: Check for specific text in the link
            link_text = link.get_text(strip=True).lower()
            if "lic" in link_text and any(keyword in link_text for keyword in ["plan", "endowment", "premium"]):
                return True
        return False

    def add_document(self, url, content):
        self.vector_db.add_texts(
            texts=[content],
            metadatas=[{"source": url, "content_type": "pdf" if ".pdf" in url.lower() else "html"}]
        )

# Initialize the embedder
embedder = GeminiEmbedder(id="models/text-embedding-004", dimensions=768, api_key=api_key)

# Create Enhanced Website knowledge base with the LIC URL
website_kb = EnhancedWebsiteKnowledgeBase(
    urls=[
        "https://licindia.in/web/guest/home",
        "https://licindia.in/web/guest/endowment-plans"
    ],
    max_links=50,
    max_depth=10,  # Increased to ensure navigation to product pages and PDFs
    vector_db=LanceDb(
        table_name="website_documents",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=embedder,
    ),
    name="LIC India Website",
    instructions=[
        "Prioritize PDF processing for policy documents",
        "Extract tabular data from premium charts and benefit illustrations",
        "Capture surrender value calculations and bonus declarations",
        "Process all discovered PDFs recursively",
        "Maintain source URLs for reference",
        "Follow links to individual product pages and their associated PDFs for detailed data"
    ]
)

# Initialize the Agent with the enhanced website knowledge base
agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
    knowledge=website_kb,
    search_knowledge=True,
    description="Agent with enhanced knowledge from LIC website and PDFs",
    instructions=[
        "Search across website data and PDFs to provide detailed answers.",
        "Cite the source (PDF URL or website) when possible.",
        "If data is missing, indicate it and suggest where to look.",
        "Handle tabular data from PDFs and web pages carefully and include in responses."
    ]
)

# Load the knowledge base using the default load method
try:
    logger.info("Loading knowledge base...")
    website_kb.load(recreate=True)  # Use True to rebuild with new data
    logger.info("Knowledge base loaded successfully.")
except Exception as e:
    logger.error(f"Error loading knowledge base: {str(e)}")

# Use the agent with a relevant query
agent.print_response("What are the key features of the LIC's Single Premium Endowment Plan?", markdown=True)