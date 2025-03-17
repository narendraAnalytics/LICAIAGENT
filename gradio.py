import os
import gradio as gr
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Controller, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from pydantic import SecretStr, BaseModel
from typing import List, Optional

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Browser configuration
config = BrowserConfig(headless=False, disable_security=True)

# Browser and Context Configuration
context_config = BrowserContextConfig(
    cookies_file="cookies.json",
    wait_for_network_idle_page_load_time=3.0,
    browser_window_size={'width': 1280, 'height': 1100},
    locale='en-US',
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    viewport_expansion=500
)

# Initialize browser and context globally
browser = Browser(config=config)
context = BrowserContext(browser=browser, config=context_config)

# Define data models
class QuarterlyFinancials(BaseModel):
    month: str
    revenue_inr: float
    net_income_inr: float
    net_profit_margin: float
    year_over_year: float

class StockDetails(BaseModel):
    company: str
    current_price: str
    daily_change: str
    open_price: str
    high_price: str
    pe_ratio: str
    high_52_week: str
    low_52_week: str
    related_stocks: List[str]
    quarterly_financials: List[QuarterlyFinancials]
    about: Optional[str] = None
    ceo: Optional[str] = None
    founder: Optional[str] = None
    parent_organization: Optional[str] = None 

# Initialize LLM Model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))
controller = Controller(output_model=StockDetails)

# Function to create an agent for stock analysis
def create_agent(company_name):
    return Agent(
        task=f"""First, open google.com. 
                 Then, search for '{company_name} share price' and extract the current price, P/E ratio, 
                 52-week high/low, RSI, and analyst rating.
                 Scroll down to find the 'Quarterly Financials' section and extract revenue, net income, net profit margin, and Y/Y percentage.
                 Extract the company's 'About', 'CEO', 'Founder', and 'Parent Organization' details.
                 Also, identify related stocks from the same sector.
                 After serach completed go back to the google.com page""",
        llm=llm,
        controller=controller,
        browser_context=context
    )

# Function to run the agent with error handling and retries
async def run_agent(company_name, max_retries=3):
    if company_name.lower() == "exit":
        await browser.close()
        return "🔴 Browser closed. Type 'exit' to confirm closure."

    agent = create_agent(company_name)

    for attempt in range(1, max_retries + 1):
        try:
            history = await agent.run()
            result = history.final_result()

            if result:
                stock_data = StockDetails.model_validate_json(result)

                output = f"### Stock Analysis for {stock_data.company}\n"
                output += f"**Current Price:** {stock_data.current_price}\n"
                output += f"**P/E Ratio:** {stock_data.pe_ratio}\n"
                output += f"**52-Week High:** {stock_data.high_52_week}\n"
                output += f"**52-Week Low:** {stock_data.low_52_week}\n"
                output += f"**Daily Change:** {stock_data.daily_change}\n"
                output += f"**Open Price:** {stock_data.open_price}\n"
                output += f"**High Price:** {stock_data.high_price}\n"
                output += f"**Related Stocks:** {', '.join(stock_data.related_stocks) if stock_data.related_stocks else 'N/A'}\n"
                output += f"**About:** {stock_data.about if stock_data.about else 'N/A'}\n"
                output += f"**CEO:** {stock_data.ceo if stock_data.ceo else 'N/A'}\n"
                output += f"**Founder:** {stock_data.founder if stock_data.founder else 'N/A'}\n"
                output += f"**Parent Organization:** {stock_data.parent_organization if stock_data.parent_organization else 'N/A'}\n\n"

                if stock_data.quarterly_financials:
                    output += "### Quarterly Financials\n"
                    for quarter in stock_data.quarterly_financials:
                        output += f"- **{quarter.month}:** Revenue - ₹{quarter.revenue_inr} Cr, Net Income - ₹{quarter.net_income_inr} Cr, "
                        output += f"Net Profit Margin - {quarter.net_profit_margin}%, YoY Change - {quarter.year_over_year}%\n"
                else:
                    output += "No quarterly financials available.\n"

                return output

            else:
                raise ValueError("No stock data found.")

        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")

            if attempt == max_retries:
                return f"❌ Error: Unable to fetch stock details after {max_retries} attempts."

            await asyncio.sleep(2)  # Short delay before retrying

    return "❌ Unknown error occurred."

# Gradio Interface
async def stock_analysis(company):
    return await run_agent(company)

iface = gr.Interface(
    fn=stock_analysis,
    inputs=gr.Textbox(label="Enter Company Name (or type 'exit' to close)"),
    outputs=gr.Markdown(label="Stock Analysis Result"),
    title="Stock Analysis AI Assistant",
    description="Enter a company name to get real-time stock analysis, financials, and insights.\nType 'exit' to close the browser.",
)

iface.launch()
