from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.google import GeminiEmbedder
from agno.models.google import Gemini
from agno.agent import Agent
from agno.vectordb.search import SearchType
from dotenv import load_dotenv
import os
import requests
import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
import io
import pytesseract
import logging
from urllib.parse import urljoin
import lancedb
import re  # Import the regular expression module
import time  # Import the time module

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Function to extract content from PDFs with improved table/image handling
def extract_pdf_content(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # added
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
                            text_content += "|".join(str(cell).replace('\n', ' ') for cell in row) + "\n"
                        text_content += "\n"

                    # Fallback to OCR if no text found
                    if not text.strip():
                        img = page.to_image(resolution=300)
                        ocr_text = pytesseract.image_to_string(img.original)
                        if ocr_text.strip():
                            text_content += f"\nPage {page_num + 1} OCR Text:\n{ocr_text}\n"
                        else:
                            text_content += f"\nPage {page_num + 1}: No text found in image\n"

                except Exception as page_error:
                    logger.error(f"Error processing page {page_num + 1}: {str(page_error)}")
                    text_content += f"\nError processing page {page_num + 1}\n"

        return text_content
    except requests.exceptions.RequestException as e:
        logger.error(f"PDF download error for {pdf_url}: {str(e)}")
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        logger.error(f"PDF processing error for {pdf_url}: {str(e)}")
        return f"Error processing PDF: {str(e)}"

# Enhanced WebsiteKnowledgeBase with PDF link discovery
class EnhancedWebsiteKnowledgeBase(WebsiteKnowledgeBase):
    def process_url(self, url):
        # Corrected PDF link matching
        if self.is_pdf_url(url):
            logger.info(f"Processing PDF: {url}")
            try:
                content = extract_pdf_content(url)
                if content:
                    self.add_document(url, content, content_type="pdf")
                    logger.info(f"Successfully processed PDF: {url}")
                else:
                    logger.warning(f"Empty content from PDF: {url}")
            except Exception as e:
                logger.error(f"Failed to process PDF {url}: {str(e)}")
        else:
            try:
                soup = self._get_page_content(url)
                if soup:
                    # Find and process PDF links first
                    pdf_links = [a["href"] for a in soup.find_all("a", href=True)
                                 if self.is_pdf_url(a["href"])]  # modified
                    for pdf_link in pdf_links:
                        absolute_url = self._make_absolute_url(url, pdf_link)
                        if absolute_url not in self.visited_urls:
                            self.process_url(absolute_url)

                    # Then process normal page content
                    self._extract_and_store_content(url, soup)
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")

    def _make_absolute_url(self, base_url, link):
        return urljoin(base_url, link)

    def add_document(self, url, content, content_type="page"):
        self.vector_db.add_texts(
            texts=[content],
            metadatas=[{"source": url, "content_type": content_type}]
        )

    def _extract_and_store_content(self, url, soup):
        """
        Extracts and stores content from a web page.
        """
        text = self._extract_text(soup)
        if text:
            self.add_document(url, text, content_type="page")
            logger.info(f"Successfully processed: {url}")
            self.visited_urls.add(url)
        else:
            logger.warning(f"No content found on: {url}")

    def is_pdf_url(self, url):
        """
        Checks if a URL is likely a PDF URL, even if it doesn't end with ".pdf".
        """
        pdf_pattern = re.compile(r"\.pdf($|\?)", re.IGNORECASE)
        return bool(pdf_pattern.search(url))

# Initialize components
embedder = GeminiEmbedder(id="models/text-embedding-004", dimensions=768, api_key=api_key)
vector_db = LanceDb(
    table_name="lic_documents",
    uri="tmp/lancedb",
    search_type=SearchType.vector,
    embedder=embedder,
)

knowledge_base = EnhancedWebsiteKnowledgeBase(
    urls=[
        "https://licindia.in/web/guest/home",
        "https://licindia.in/documents/20121/1256234/102259+LIC_Linked+Accident+Benefit+Rider+Sales+Brochure_OCT+24.pdf/30b31c1c-fc8d-5f78-45a7-9b6b07e9c7f6?t=1729168072699"
        ],
    vector_db=vector_db,
    max_links=100,  # Increased
    max_depth=4, #increased
    name="LIC India Website",
    instructions=[
        "Prioritize PDF processing for policy documents",
        "Extract tabular data from premium charts and benefit illustrations",
        "Capture surrender value calculations and bonus declarations",
        "Process all discovered PDFs recursively",
        "Maintain source URLs for reference",
        "Handle URLs that do not end with .pdf but still point to a pdf file",
        "Extract data from the LIC's Single Premium Endowment Plan Sales Brochure", #added
        "Extract data from the LIC's Single Premium Endowment Plan", #added
        "Give detail information about the Key features of LIC’s SINGLE PREMIUM ENDOWMENT PLAN",
        "Extract all policy details from PDFs" #added
    ]
)

# Initialize the Agent with enhanced instructions
agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
    knowledge=knowledge_base,
    search_knowledge=True,
    description="📄 LIC India Policy Expert with PDF Analysis",
    instructions=[
        "Reference specific PDF documents when citing policy details",
        "Highlight tabular data from premium charts and benefit illustrations",
        "Mention OCR confidence for image-extracted content",
        "Compare information across multiple PDF sources when available",
        "Include document source URLs in responses",
        "Handle calculation-heavy content with care (e.g., premium tables)",
        "Acknowledge limitations of OCR-extracted data when applicable",
        "find the Key features of LIC’s SINGLE PREMIUM ENDOWMENT PLAN, you should only reply with information from PDF", #added
        "If there are multiple documents available, compare the information across multiple PDF sources and reply with the aggregated content",
        "When ask about the Key Features, provide the correct details and also tell the document name from where the data is taken",
        "If you know the question will be answered with the PDF then look into only PDF not for the other sources"
         #added
    ]
)

# Rebuild the knowledge base with PDF discovery
def initialize_knowledge():
    logger.info("Starting knowledge base initialization...")
    knowledge_base.load(recreate=False)  # Changed to True
    logger.info("Knowledge base initialization complete.")

if __name__ == "__main__":
    initialize_knowledge()
    time.sleep(1) # added
    # Example query
    agent.print_response("Details of the LIC’s Linked Accidental Death Benefit Rider")
