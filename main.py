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
            self.messages.append(SystemMessage(content=system))
    
    def __call__(self, message):
        self.messages.append(HumanMessage(content=message))
        result = self.execute()
        self.messages.append(AIMessage(content=result))
        return result
    
    def execute(self):
        chat = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
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
Read the following CV text 100 times, deeply understand it and than Convert the following CV text into a JSON data structure with the following keys and specifications: 
"Skills": The value should be a dictionary like technical skills, tools, programming languages,communication, leadership etc from the resume
"Title":This should describe their expertise like software develop,web designer,chartered accountant,loco pilot etc.This should be their main profession which they have mentioned or done before mentioned in the resume
"Education": The value should be a list of dictionaries, each containing information about the below ones[The values are examples given for understanding but the keys should not be changed]
1-School/institution:XYZ college
2-Degree(optional):BTECH
3-Field of study(optional):Chemical engineering
4-Percentage(optional):95%
5-CGPA(optional):9.8
6-Start date(optional):12-12-12
7-End date(optional):12-12-14
8-Description- of the particular education like studied btech computer science in xyz university
All the things mentioned from 1 to 6 should be keys for their own without changing the keys anytime you give the answer.If there are multiple educations each should be provided in the same format as above with a numbering for every education qualification starting from 1 like 1,2,3,4..
"Work experience":The value should be a list of dictionaries, each containing information about the below ones[The values are examples given for understanding but the keys should not be changed]
1- Company: Company XYZ
2-Position :Software developer
3-Start date: 12-12-16
4-End date: 12-12-18
5-Description:Worked in fixing bugs
If there are multiple work/intern/job experience each should be provided in the same format as above with a numbering for every work/job/intern starting from 1 like 1,2,3,4..
"Languages":The value should be a list of dictionaries, each dictionary ahould be like the below ones 
{language known mentioned in the cv like english,tamil,telugu etc:proficiency of the specified language(one of basic,fluent,expert and null if nothing is mentioned)
If there are multiple languages each should be provided in the same format as above with a numbering for every work/job/intern starting from 1 like 1,2,3,4..
"Bio":From the cv analyse and give a paragraph or bullet points help people get to know a glance at the person(whose cv we are analysing) what work does the person(whose cv we are analysing) do the best
"Hourly rate":This is the rate in which clients hire.If it is mentioned in the cv mention it or else null
"Phone number":Phone number if it is mentioned in the cv mention it here.If there are multiple phone numbers then mention everything here
"Date of birth":date of birth if it is mentioned in the cv mention it here
"Address":Address if it is mentioned in cv mention it here
"Country":Mention the country of living it is mentioned in the cv
"State":Mention the state of living it is mentioned in the cv
"City/Province":Mention the city/province of living it is mentioned in the cv
"Zip/Postal code":Mention the zip/postal code of living it is mentioned in the cv


The Final Output Should start with '
json' and trailing with '
'. 
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
