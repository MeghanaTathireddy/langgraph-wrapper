import os
from dotenv import load_dotenv
import sentry_sdk
from wrapper import LangGraphPDFWrapper

load_dotenv()

#Sentry initialization
sentry_dsn = os.getenv("SENTRY_DSN")
sentry_sdk.init(
    dsn=sentry_dsn,
    traces_sample_rate=1.0  
)


agent = LangGraphPDFWrapper()

pdf_path = input("Enter path to your PDF file: ").strip()

if not os.path.exists(pdf_path):
    sentry_sdk.capture_message(f"PDF file not found: {pdf_path}")
    print("PDF file not found. Exiting.")
    exit()

user_prompt = input("Ask your question: ").strip()

try:
    response = agent.run(user_prompt, pdf_path)
    print("\nAnswer:")
    print(response)
except Exception as e:
    sentry_sdk.capture_exception(e)
    print("An error occurred while processing your request. Check Sentry dashboard for details.")