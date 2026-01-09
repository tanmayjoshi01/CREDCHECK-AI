from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from duckduckgo_search import DDGS

# Load models
tokenizer_style = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model_style = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
nlp = spacy.load("en_core_web_sm")

def verify_headline(headline):
    trusted_domains = ["bbc", "reuters", "apnews", "npr", "gov", "edu"]
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(headline, max_results=3))
        for result in results:
            url = result.get('href', '').lower()
            for domain in trusted_domains:
                if domain in url:
                    source_name = "Government" if domain == "gov" else "Educational" if domain == "edu" else domain.capitalize()
                    return f"Verified by {source_name}"
        return "Unverified claim"
    except Exception:
        return "Unverified claim"

def analyze_text(text):
    # Base Prediction
    inputs = tokenizer_style(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model_style(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence_scores = probs[0].tolist()
    label = "Fake" if torch.argmax(probs).item() == 1 else "Real"
    score = round(confidence_scores[torch.argmax(probs).item()], 2)

    # --- EXPLAINABLE AI (XAI) LOGIC ---
    words = text.split()
    explanations = []
    
    # Calculate impact of each word
    # If removing a word makes the 'Fake' score drop significantly, that word is a 'suspicious signal'
    for i in range(len(words)):
        temp_words = words.copy()
        temp_words.pop(i)
        temp_text = " ".join(temp_words)
        
        t_inputs = tokenizer_style(temp_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            t_outputs = model_style(**t_inputs)
            t_probs = torch.nn.functional.softmax(t_outputs.logits, dim=-1)
            # Difference in 'Fake' probability (index 1)
            impact = confidence_scores[1] - t_probs[0][1].item()
            explanations.append({"word": words[i], "score": round(impact, 4)})

    # Extract entities
    doc = nlp(text)
    entities = list(set([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]))
    
    return {
        "label": label,
        "score": score,
        "entities": entities,
        "verification_note": verify_headline(text),
        "explanation": explanations # This sends the highlight data to the frontend
    }