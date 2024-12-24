from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import io
import os
import PyPDF2
import json
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Agent class for Generative AI interaction
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append(SystemMessage(content=self.system))
    
    def __call__(self, message):
        self.messages.append(HumanMessage(content=message))
        result = self.execute()
        self.messages.append(AIMessage(content=result))
        return result
    
    def execute(self):
        # Use Gemini API without specifying the version
        chat = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,convert_system_message_to_human=True)
        result = chat.invoke(self.messages)
        return result.content

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

# Function to extract structured data from CV text
def extract_structured_data(cv_text):
    structured_prompt_2 = """ 
Read the following CV text deeply and convert it into a JSON data structure with the following keys and specifications: 
"Skills": The value should be a dictionary like technical skills, tools, programming languages, communication, leadership, etc., from the resume.
"Title": This should describe their expertise like software developer, web designer, chartered accountant, etc. This should be their main profession mentioned in the resume.
"Education": The value should be a list of dictionaries, each containing details like institution, degree, field of study, start/end dates, description, etc.
"Work experience": The value should be a list of dictionaries, each containing details like company name, position, start/end dates, description, etc.
"Languages": A list of dictionaries with language and proficiency (basic, fluent, expert, or null if not mentioned).
"Bio": A paragraph or bullet points summarizing the individual.
"Hourly rate": The rate mentioned in the CV, or null if not available.
"Phone number": Phone numbers mentioned in the CV.
"Date of birth": Date of birth if mentioned.
"Address": Address if mentioned.
"Country": Country of residence if mentioned.
"State": State of residence if mentioned.
"City/Province": City or province of residence if mentioned.
"Zip/Postal code": Postal code if mentioned.
The output should be in JSON format.
""".strip() 

    bot2 = Agent(structured_prompt_2)
    res = bot2(cv_text)
    text_to = res.strip().strip('```').strip('```json')
    return json.loads(text_to)

# FastAPI route for processing PDF and extracting structured data
@app.post("/process_resume")
async def process_resume(file: UploadFile = File(...)):
    try:
        # Step 1: Read the PDF file
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        # Step 2: Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_file)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        
        # Step 3: Extract structured data from the text
        structured_data = extract_structured_data(extracted_text)
        
        return {
            "text": extracted_text,
            "structured_data": structured_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Resume Processing API!",
        "instructions": "Use /process_resume to upload a PDF and extract structured data."
    }
