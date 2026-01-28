from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
llm = ChatGroq(
    temperature=0.5,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)
def get_recommendations(disease, probability):
    """
    Generates clinical decision-support recommendations
    for retinal diseases.
    Fully compatible with OPEN-SET (UNKNOWN) predictions.
    """

    prompt_template = """
You are an expert ophthalmologist AI assistant.

The retinal disease prediction is generated using a deep learning–based
OPEN-SET classification system with CLASS-WISE THRESHOLDING.
This means the system explicitly decides whether an image belongs to a
KNOWN disease class or should be labeled as UNKNOWN based on feature-distance thresholds.

Predicted Disease Label: {disease}
Model Confidence (supporting score): {probability}

IMPORTANT:
- The final KNOWN / UNKNOWN decision has ALREADY been made by the model.
- Your role is ONLY to provide clinical decision-support guidance.

STRICTLY return the response in the following JSON format ONLY:

{{
  "predicted_disease": "{disease}",
  "confidence_score": {probability},
  "risk_level": "Low / Moderate / High / Uncertain",
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2",
    "Recommendation 3"
  ],
  "suggested_tests": [
    "Test 1",
    "Test 2"
  ],
  "common_medications": [
    "Medication class or commonly used therapy (no dosage)",
    "Medication class or therapy option"
  ],
  "diet_and_lifestyle": [
    "Dietary or lifestyle recommendation 1",
    "Dietary or lifestyle recommendation 2"
  ],
  "referral_advice": "Primary care / Ophthalmologist / Retina specialist / Emergency referral",
  "clinical_note": "Short clinically relevant note for the physician",
  "disclaimer": "AI-based decision support only. Final diagnosis and treatment must be made by a qualified medical professional."
}}

RULES (MANDATORY):
- If predicted disease is "UNKNOWN":
    - Set risk_level to "Uncertain"
    - Do NOT assume or guess a disease
    - Emphasize need for further clinical evaluation
    - Recommend ophthalmologist or retina specialist consultation
    - Medications must be general supportive categories only (no disease-specific drugs)
- If predicted disease is KNOWN:
    - Risk level should reflect disease severity in general clinical practice
- Do NOT provide medication dosages
- Do NOT provide brand names
- Do NOT provide invasive treatment instructions
- Diet suggestions must be general, safe, and supportive
- Do NOT add any text outside the JSON
"""


    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "disease": disease,
        "probability": round(float(probability), 3)
    })

    return json.loads(response)
