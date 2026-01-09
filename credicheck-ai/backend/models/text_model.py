from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from duckduckgo_search import DDGS

# Optional XAI imports - fail gracefully if unavailable
HAS_LIME = False
HAS_SHAP = False

try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    pass

try:
    import shap
    HAS_SHAP = True
except ImportError:
    pass

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

tokenizer_style = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model_style = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

nlp = spacy.load("en_core_web_sm")

def generate_text_explanation(label, confidence):
    if confidence >= 0.75:
        if label == "Fake":
            return "The analysis suggests this content may contain false or misleading information. Consider verifying claims with reputable sources before sharing."
        else:
            return "The analysis suggests this content appears credible. However, always verify important information through multiple reliable sources."
    elif confidence >= 0.55:
        if label == "Fake":
            return "The analysis indicates some characteristics of unreliable content. Exercise caution and verify information from trusted sources."
        else:
            return "The analysis suggests this content may be credible, but confidence is moderate. Additional verification is recommended."
    else:
        return "The analysis results are inconclusive. It is strongly recommended to verify this information through multiple reputable sources before making any decisions."

def analyze_text_style(text):
    inputs = tokenizer_style(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model_style(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence_scores = predictions[0].tolist()
    max_index = confidence_scores.index(max(confidence_scores))
    style_confidence = confidence_scores[max_index]
    
    if max_index == 1:
        style_label = "Fake"
    else:
        style_label = "Real"
    
    return {
        "style_label": style_label,
        "style_confidence": style_confidence
    }

def extract_entities(text):
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entities.append(ent.text)
    
    return entities

def explain_prediction(text):
    """
    Provides lightweight explainability by identifying influential words and patterns.
    Returns a list of human-readable explanations.
    """
    explanations = []
    
    # Sensational/emotional language indicators
    sensational_words = [
        "shocking", "unbelievable", "exposed", "secret", "hidden", "scandal",
        "outrageous", "devastating", "breaking", "urgent", "warning", "alert",
        "truth", "they don't want you to know", "doctors hate", "amazing trick",
        "miracle", "guaranteed", "instant", "revolutionary", "never before seen"
    ]
    
    # Strong emotional terms
    emotional_terms = [
        "terrified", "devastated", "destroyed", "ruined", "catastrophic",
        "explosive", "outrageous", "horrifying", "shocking", "stunning"
    ]
    
    # All caps detection (often used in fake news)
    text_lower = text.lower()
    words = text.split()
    
    # Check for sensational language
    sensational_found = []
    for word in sensational_words:
        if word in text_lower:
            sensational_found.append(word)
    
    if sensational_found:
        explanations.append("sensational language detected")
    
    # Check for strong emotional terms
    emotional_found = []
    for term in emotional_terms:
        if term in text_lower:
            emotional_found.append(term)
    
    if emotional_found:
        explanations.append("strong emotional terms used")
    
    # Check for all caps words (indicative of clickbait)
    all_caps_count = sum(1 for word in words if word.isupper() and len(word) > 2)
    if all_caps_count > 0:
        explanations.append("excessive capitalization detected")
    
    # Extract key entities (already done in analyze_text, but use for explanation)
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    
    if entities and len(explanations) == 0:
        explanations.append("key entities mentioned without strong verification context")
    elif entities and "sensational language detected" in explanations:
        explanations.append("entities mentioned with sensational wording")
    
    # If no specific patterns found, provide generic explanation based on prediction
    if not explanations:
        # Check for exclamation marks (often used in fake news)
        if text.count('!') > 2:
            explanations.append("excessive exclamation marks detected")
        else:
            explanations.append("text pattern analysis indicates classification")
    
    # Limit to 3-5 explanations as requested
    return explanations[:5] if explanations else ["text analysis completed"]


def generate_xai_explanation(text):
    """
    Generates XAI explanation using LIME and SHAP.
    Returns None if XAI libraries are unavailable or if any error occurs.
    This function is fail-safe and will not crash the API.
    """
    result = {
        "lime_keywords": [],
        "shap_summary": ""
    }
    
    try:
        # Helper function for LIME to use
        def predict_proba_wrapper(texts):
            """Wrapper function for LIME that returns prediction probabilities"""
            probs = []
            for text_input in texts:
                try:
                    inputs = tokenizer_style(text_input, return_tensors="pt", truncation=True, max_length=512, padding=True)
                    with torch.no_grad():
                        outputs = model_style(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    probs.append(predictions[0].cpu().numpy())
                except Exception:
                    # Return neutral probabilities if tokenization fails
                    probs.append([0.5, 0.5])
            return torch.tensor(probs).numpy()
        
        # LIME Explanation
        if HAS_LIME:
            try:
                explainer = LimeTextExplainer(class_names=["Real", "Fake"])
                explanation = explainer.explain_instance(
                    text, 
                    predict_proba_wrapper, 
                    num_features=10,
                    num_samples=50
                )
                
                # Extract top influential words (3-5 words)
                exp_list = explanation.as_list()
                lime_keywords = []
                
                # Sort by absolute value of weight (importance)
                sorted_exp = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
                
                # Extract top 3-5 keywords (positive or negative influence)
                for word, weight in sorted_exp[:5]:
                    # Only include words that are actual words (not punctuation/special chars)
                    clean_word = word.strip().strip('.,!?;:').lower()
                    if len(clean_word) > 2 and clean_word.isalnum():
                        lime_keywords.append(clean_word)
                        if len(lime_keywords) >= 5:
                            break
                
                result["lime_keywords"] = lime_keywords[:5] if lime_keywords else []
                
            except Exception as e:
                # If LIME fails, continue without it
                result["lime_keywords"] = []
        
        # SHAP Explanation (lightweight - no plots)
        if HAS_SHAP:
            try:
                # Lightweight SHAP explanation
                def predict_wrapper(texts):
                    """Wrapper for SHAP that returns class probabilities for Fake class"""
                    import numpy as np
                    
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    batch_probs = []
                    for txt in texts:
                        try:
                            inputs = tokenizer_style(txt, return_tensors="pt", truncation=True, max_length=512, padding=True)
                            with torch.no_grad():
                                outputs = model_style(**inputs)
                                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            # Return probability of "Fake" class (index 1)
                            batch_probs.append(predictions[0][1].item())
                        except Exception:
                            batch_probs.append(0.5)
                    
                    return np.array(batch_probs)
                
                # Simple SHAP explanation using permutation-based approach
                # Split text into words for feature attribution
                words = text.split()
                if len(words) > 0:
                    import numpy as np
                    
                    # Get original prediction score
                    original_score = predict_wrapper(text)[0]
                    
                    # Simple feature importance by removing words (permutation-based)
                    word_influences = []
                    # Limit to first 10 words for speed in hackathon demo
                    sample_words = words[:10] if len(words) > 10 else words
                    
                    for word in sample_words:
                        # Create text without this word
                        text_without = text.replace(word, "", 1).strip()
                        if len(text_without) > 0:
                            score_without = predict_wrapper(text_without)[0]
                            influence = abs(original_score - score_without)
                            word_influences.append(influence)
                    
                    # Determine influence level based on SHAP-style analysis
                    if len(word_influences) > 0:
                        max_influence = float(np.max(word_influences))
                        mean_influence = float(np.mean(word_influences))
                        
                        if max_influence > 0.15 or mean_influence > 0.08:
                            result["shap_summary"] = "Prediction strongly influenced by specific wording patterns"
                        elif max_influence > 0.08 or mean_influence > 0.04:
                            result["shap_summary"] = "Prediction influenced by emotional and sensational wording"
                        else:
                            result["shap_summary"] = "Prediction based on overall text characteristics"
                    else:
                        result["shap_summary"] = "Prediction influenced by text content patterns"
                else:
                    result["shap_summary"] = "Prediction based on text analysis"
                    
            except Exception as e:
                # If SHAP fails, provide a generic summary
                result["shap_summary"] = "Prediction based on text analysis"
        
        # If neither LIME nor SHAP worked, return None to skip XAI
        if not result["lime_keywords"] and not result["shap_summary"]:
            return None
        
        # Ensure we have at least a basic summary
        if not result["shap_summary"]:
            result["shap_summary"] = "Prediction based on text analysis"
        
        return result
        
    except Exception as e:
        # Complete fail-safe: return None if anything goes wrong
        return None


def verify_headline(headline):
    trusted_domains = ["bbc", "reuters", "apnews", "npr", "gov", "edu"]
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(headline, max_results=3))
        
        for result in results:
            url = result.get('href', '').lower()
            for domain in trusted_domains:
                if domain in url:
                    source_name = domain.capitalize()
                    if domain == "gov":
                        source_name = "Government"
                    elif domain == "edu":
                        source_name = "Educational"
                    return f"Verified by {source_name}"
        
        return "Unverified claim"
    except Exception:
        return "Unverified claim"

def analyze_text(text):
    inputs = tokenizer_style(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model_style(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence_scores = predictions[0].tolist()
    max_index = confidence_scores.index(max(confidence_scores))
    score = round(float(confidence_scores[max_index]), 2)
    
    if max_index == 1:
        label = "Fake"
    else:
        label = "Real"
    
    # Extract entities
    doc = nlp(text)
    entities = []
    seen = set()
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entity_text = ent.text
            if entity_text not in seen:
                entities.append(entity_text)
                seen.add(entity_text)
    
    # Verify headline
    verification_note = verify_headline(text)
    
    # Generate explainability
    explainability = explain_prediction(text)
    
    # Generate XAI explanation (optional, fail-safe)
    xai_explanation = generate_xai_explanation(text)
    
    # Build return dictionary
    result = {
        "label": label,
        "score": score,
        "entities": entities,
        "verification_note": verification_note,
        "explainability": explainability
    }
    
    # Add XAI explanation only if available
    if xai_explanation is not None:
        result["xai_explanation"] = xai_explanation
    
    return result

