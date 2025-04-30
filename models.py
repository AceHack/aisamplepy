from pydantic import BaseModel, Field
from typing import List

# Existing schema (can be kept or removed if no longer used elsewhere)
class ImageAnalysisToolSchema(BaseModel):
    image_url: str = Field(..., description="URL of the image to analyze.")
    prompt: str = Field(..., description="Prompt for the analysis.")


# New schema for structured input
class Finding(BaseModel):
    description: str = Field(..., description="Text description of the finding.")
    image_url: str = Field(..., description="Image URL associated with the finding.")

class StructuredImageAnalysisInput(BaseModel):
    formId: str = Field(..., description="Identifier for the form.")
    findings: List[Finding] = Field(..., description="List of findings, each with a description and a single image URL.")

# Define the Pydantic models for structured output from main.py
class BusinessOpportunity(BaseModel):
    title: str = Field(..., description="Concise title for the business opportunity.")
    description: str = Field(..., description="Detailed description of the opportunity.")
    potential_impact: str = Field(..., description="Potential positive impact if pursued.")
    source_image_url: str = Field(..., description="The URL of the image finding that inspired this opportunity.")
    source_description: str = Field(..., description="The description of the finding that inspired this opportunity.")

class OpportunitiesReport(BaseModel):
    opportunities: List[BusinessOpportunity] = Field(description="A list of 3-5 identified business opportunities.")
