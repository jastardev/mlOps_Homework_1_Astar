from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.models.similarity_model import EmailClassifierModel
from app.dataclasses import Email

router = APIRouter()

def email_data_file():
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')

class EmailRequest(BaseModel):
    subject: str
    body: str
    method: str = "topic"  # "topic" or "nearest_neighbor"

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email, method=request.method)
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StoreEmailItem(BaseModel):
    subject: str
    body: str
    topic: Optional[str] = None

class StoreEmailsRequest(BaseModel):
    emails: List[StoreEmailItem]

@router.put("/emails")
async def store_emails(request: StoreEmailsRequest):
    try:    
        with open(email_data_file(), "r") as f:
            emails = json.load(f)

        emails.extend([email.model_dump() for email in request.emails])

        with open(email_data_file(), "w") as f:
            json.dump(emails, f, indent=2)

        return {
            "message": "Emails Stored Successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}


class TopicInput(BaseModel):
    name: str
    description: str

class CreateTopicsRequest(BaseModel):
    topics: List[TopicInput]

class CreateTopicsResponse(BaseModel):
    message: str
    created: List[TopicInput]

@router.post("/topics", status_code=201)
async def create_topics(request: CreateTopicsRequest):
    """Create one or more new topics and store them in the topics config file"""
    try:
        model = EmailClassifierModel()
        model.create_topics([t.model_dump() for t in request.topics])

        inference_service = EmailTopicInferenceService()
        info = inference_service.get_pipeline_info()

        return {
            "message": f"{len(request.topics)} topic(s) created successfully",
            "Available Topics": info["available_topics"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

