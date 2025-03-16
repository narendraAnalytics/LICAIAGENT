import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Controller, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from pydantic import SecretStr, BaseModel

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")



# Browser configuration
config = BrowserConfig(
    headless=True,        # Run in headless mode to prevent UI issues
    disable_security=True, # Bypass security warnings
)

# Browser and Context Configuration
context_config = BrowserContextConfig(
    cookies_file="cookies.json",  # Store and reuse session cookies
    wait_for_network_idle_page_load_time=3.0,  # Ensure pages fully load
    browser_window_size={'width': 1280, 'height': 1100},  # Set window size
    locale='en-US',  # Set language preference
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',  # Custom user-agent
    highlight_elements=True,  # Useful for debugging (highlights extracted elements)
    viewport_expansion=500,  # Expand viewable area
    #allowed_domains=['about:blank','google.com', 'wikipedia.org'],  # Restrict navigation to these sites
)

# Initialize Browser and Context
browser = Browser(config=config)
context = BrowserContext(browser=browser, config=context_config)

# Define structured output model
class StockDetails(BaseModel):
    company: str
    current_price: str
    pe_ratio: str
    high_52_week: str
    low_52_week: str
    rsi: str
    analyst_rating: str
    recommendation: str 

# Initialize LLM Model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))
controller = Controller(output_model=StockDetails)

# Function to determine Buy/Hold/Sell recommendation
def make_stock_decision(price, pe_ratio, rsi):
    try:
        pe = float(pe_ratio) if pe_ratio.replace('.', '', 1).isdigit() else None
        rsi_value = float(rsi) if rsi.replace('.', '', 1).isdigit() else None

        if pe and rsi_value:
            if pe < 15 and rsi_value < 30:
                return "Buy"
            elif 15 <= pe <= 25 and 30 <= rsi_value <= 70:
                return "Hold"
            else:
                return "Sell"
        return "Hold"  # Default
    except:
        return "Hold"

# Function to create an agent for stock analysis
def create_agent(company_name):
    return Agent(
        task=f"""First, open https://www.google.com/. 
                 Then, search for '{company_name} share price' and extract the current price, P/E ratio, 
                 52-week high/low, RSI, and analyst rating.""",
        llm=llm,
        controller=controller,
        browser_context=context,  # Use persistent browser context
    )

# Function to run the agent
async def run_agent(company_name):
    agent = create_agent(company_name)

    try:
        history = await agent.run()
        result = history.final_result()  # Get structured result

        if result:
            stock_data = StockDetails.model_validate_json(result)
            stock_data.recommendation = make_stock_decision(
                stock_data.current_price,
                stock_data.pe_ratio,
                stock_data.rsi
            )

            print("\n--- Stock Analysis ---")
            print(f"Company: {stock_data.company}")
            print(f"Current Price: {stock_data.current_price}")
            print(f"P/E Ratio: {stock_data.pe_ratio}")
            print(f"52-Week High: {stock_data.high_52_week}")
            print(f"52-Week Low: {stock_data.low_52_week}")
            print(f"RSI: {stock_data.rsi}")
            print(f"Analyst Rating: {stock_data.analyst_rating}")
            print(f"Recommendation: {stock_data.recommendation}")
        else:
            print("No stock data found")
    except Exception as e:
        print(f"Error: {e}")

# Main function: keeps browser open & waits for user input
async def main():
    try:
        while True:
            company = input("\nEnter company name (or type 'exit' to quit): ")
            if company.lower() == "exit":
                break
            await run_agent(company)  # Run agent for each input
    finally:
        print("\nClosing browser...")
        await browser.close()  # Manually close the browser when exiting

# Run the event loop
if __name__ == "__main__":
    asyncio.run(main())
