from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import spacy
import torch
import hashlib
import time
import base64
import io
import os
from duckduckgo_search import DDGS
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

cache = {}
RESULTS_CACHE = {}

# Source tier definitions
TIER_1_DOMAINS = ["bbc", "reuters", "apnews", "npr"]
TIER_2_DOMAINS = ["cnn", "foxnews", "theguardian", "nytimes", "aljazeera", "washingtonpost", "bloomberg"]
TIER_1_NAMES = {"bbc": "BBC News", "reuters": "Reuters", "apnews": "AP News", "npr": "NPR"}
TIER_2_NAMES = {
    "cnn": "CNN", "foxnews": "Fox News", "theguardian": "The Guardian",
    "nytimes": "The New York Times", "aljazeera": "Al Jazeera",
    "washingtonpost": "The Washington Post", "bloomberg": "Bloomberg"
}

# Load all models at startup
print("Loading models...")
text_tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
text_model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
spacy_nlp = spacy.load("en_core_web_sm")
clip_model = SentenceTransformer('clip-ViT-B-32')
print("Models loaded successfully!")

# Helper function for headline verification using DuckDuckGo Search with tier-based classification
def verify_headline(headline):
    articles = []
    tier_found = None
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(headline, max_results=10))
        
        for result in results:
            url = result.get('href', '')
            url_lower = url.lower()
            title = result.get('title', '')
            
            # Determine tier based on URL domain
            source_tier = None
            source_name = None
            
            # Check Tier 1 (Global Neutral)
            for domain in TIER_1_DOMAINS:
                if domain in url_lower:
                    source_tier = 1
                    source_name = TIER_1_NAMES.get(domain, domain.capitalize())
                    break
            
            # Check Tier 2 (Major Media) if not Tier 1
            if source_tier is None:
                for domain in TIER_2_DOMAINS:
                    if domain in url_lower:
                        source_tier = 2
                        source_name = TIER_2_NAMES.get(domain, domain.capitalize())
                        break
            
            # Check Tier 3 (Regional/Others) - any valid news domain
            if source_tier is None:
                # Common news domain patterns
                news_patterns = ['.com/', '.org/', '.net/', '.edu/']
                if any(pattern in url_lower for pattern in news_patterns) and url_lower.startswith('http'):
                    # Extract domain name
                    try:
                        parsed = urlparse(url)
                        domain_parts = parsed.netloc.replace('www.', '').split('.')
                        if len(domain_parts) >= 2:
                            source_tier = 3
                            source_name = domain_parts[0].capitalize()
                    except:
                        pass
            
            # Add article if it's a valid news source and not duplicate
            if source_tier and source_name and url:
                if not any(a['url'] == url for a in articles):
                    articles.append({
                        "title": title,
                        "source": source_name,
                        "url": url,
                        "tier": source_tier
                    })
                    
                    # Track the highest tier found
                    if tier_found is None or source_tier < tier_found:
                        tier_found = source_tier
        
        # Determine source_coverage_level based on highest tier found
        if tier_found == 1:
            source_coverage_level = "Tier 1 - Global Neutral"
        elif tier_found == 2:
            source_coverage_level = "Tier 2 - Major Media"
        elif tier_found == 3:
            source_coverage_level = "Tier 3 - Regional/Others"
        else:
            source_coverage_level = "No Coverage Found"
        
        # Return structured response with tier-based classification
        return {
            "source_coverage_level": source_coverage_level,
            "articles": articles
        }
    except Exception:
        # Graceful fallback on any error
        return {
            "source_coverage_level": "No Coverage Found",
            "articles": []
        }

# Analysis functions using global models
def analyze_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = text_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence_scores = predictions[0].tolist()
    max_index = confidence_scores.index(max(confidence_scores))
    score = round(float(confidence_scores[max_index]), 2)
    
    if max_index == 1:
        label = "Fake"
    else:
        label = "Real"
    
    # Extract entities
    doc = spacy_nlp(text)
    entities = []
    seen = set()
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entity_text = ent.text
            if entity_text not in seen:
                entities.append(entity_text)
                seen.add(entity_text)
    
    # Verify headline
    verification = verify_headline(text)
    
    return {
        "label": label,
        "score": score,
        "entities": entities,
        "verification": verification
    }

def verify_image_context(image_input, headline_text):
    """
    Verify image context. Accepts either:
    - image_path (str): Path to image file
    - image_data (str): Base64 encoded image data (data URL format)
    """
    # Handle base64 image data
    if isinstance(image_input, str) and image_input.startswith('data:image'):
        # Extract base64 data from data URL
        header, encoded = image_input.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_input, str) and os.path.exists(image_input):
        # Handle file path
        image = Image.open(image_input)
    else:
        # Try to decode as base64 directly
        try:
            image_bytes = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_bytes))
        except:
            raise ValueError("Invalid image input format")
    
    image_embedding = clip_model.encode(image)
    text_embedding = clip_model.encode(headline_text)
    
    image_embedding = image_embedding.reshape(1, -1)
    text_embedding = text_embedding.reshape(1, -1)
    
    similarity = cosine_similarity(image_embedding, text_embedding)[0][0]
    
    if similarity < 0.20:
        image_context = "Mismatch"
    elif similarity >= 0.25:
        image_context = "Consistent"
    else:
        image_context = "Mismatch"
    
    return {
        "image_context": image_context,
        "similarity_score": float(similarity)
    }

@app.route('/analyze-text', methods=['POST'])
def analyze_text_endpoint():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    text = data['text']
    
    # Generate MD5 hash of input text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache
    if text_hash in RESULTS_CACHE:
        cached_result = RESULTS_CACHE[text_hash].copy()
        cached_result['cached'] = True
        return jsonify(cached_result)
    
    try:
        result = analyze_text(text)
        
        # Prepare response with same structure
        response = {
            'label': result['label'],
            'score': round(result['score'], 2),
            'entities': result['entities'],
            'verification': result['verification'],
            'cached': False
        }
        
        # Store in cache (without cached flag)
        RESULTS_CACHE[text_hash] = {
            'label': result['label'],
            'score': round(result['score'], 2),
            'entities': result['entities'],
            'verification': result['verification']
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image_endpoint():
    data = request.json
    
    image_input = None
    if 'image_data' in data:
        image_input = data['image_data']
    elif 'image_path' in data:
        image_input = data['image_path']
    else:
        return jsonify({'error': 'Missing image_data or image_path field'}), 400
    
    if 'headline' not in data:
        return jsonify({'error': 'Missing headline field'}), 400
    
    headline = data['headline']
    
    try:
        result = verify_image_context(image_input, headline)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-full', methods=['POST'])
def analyze_full_endpoint():
    start_time = time.time()
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing required field: text'}), 400
    
    text = data['text']
    
    if 'headline' not in data:
        headline = text
    else:
        headline = data['headline']
    
    # Check for image data (base64) or image_path
    has_image = 'image_data' in data or 'image_path' in data
    image_input = data.get('image_data') or data.get('image_path', '')
    
    # Generate hash from inputs (use first 100 chars of image data for cache key)
    image_hash_input = image_input[:100] if image_input else ''
    hash_input = f"{text}{image_hash_input}{headline}"
    cache_key = hashlib.md5(hash_input.encode()).hexdigest()
    
    # Check cache
    if cache_key in cache:
        response_time_ms = (time.time() - start_time) * 1000
        cached_result = cache[cache_key].copy()
        result = {
            'final_label': cached_result.get('final_label', 'Unverified'),
            'final_trust_score': cached_result.get('final_trust_score', 50),
            'reason': cached_result.get('reason', 'Analysis result retrieved from cache.'),
            'cache_hit': True,
            'response_time_ms': round(response_time_ms, 2),
            'image_context': cached_result.get('image_context'),
            'similarity_score': cached_result.get('similarity_score'),
            'verification': cached_result.get('verification', {'source_coverage_level': 'No Coverage Found', 'articles': []})
        }
        return jsonify(result)
    
    try:
        text_result = analyze_text(text)
        text_label = text_result['label']
        text_confidence = text_result['score']
        verification = text_result.get('verification', {'source_coverage_level': 'No Coverage Found', 'articles': []})
        
        # Image analysis (skip if image not provided)
        image_context = None
        image_similarity = None
        if has_image and image_input:
            try:
                image_result = verify_image_context(image_input, headline)
                image_context = image_result['image_context']
                image_similarity = image_result['similarity_score']
            except Exception as e:
                print(f"Image analysis error: {e}")
                has_image = False
        
        # Aggregation logic
        text_is_fake = text_label == "Fake"
        text_high_confidence = text_confidence >= 0.75
        image_mismatch = image_context == "Mismatch" if has_image else False
        
        # Determine final_label
        if has_image:
            if text_is_fake and text_high_confidence and image_mismatch:
                final_label = "High Risk Fake"
                base_score = 20
                reason = "The text analysis indicates highly unreliable content with strong confidence, and the image does not match the headline, suggesting potential manipulation or misinformation."
            elif (text_is_fake and text_high_confidence) or (text_is_fake and image_mismatch) or (image_mismatch and text_confidence >= 0.55):
                final_label = "Possibly Misleading"
                base_score = 45
                reason = "Multiple signals suggest this content may be unreliable - either the text shows characteristics of misinformation or the image does not align with the headline, but confidence levels vary."
            elif (text_is_fake and not text_high_confidence) or (not text_is_fake and image_mismatch):
                final_label = "Unverified"
                base_score = 60
                reason = "The analysis shows conflicting or uncertain signals. The text and image context do not fully align, and verification through additional sources is recommended before sharing or relying on this content."
            else:
                final_label = "Likely Credible"
                base_score = 80
                reason = "Both the text analysis and image-headline alignment suggest this content appears credible. However, always verify important claims through multiple reliable sources as automated analysis is not infallible."
        else:
            # Text-only analysis
            if text_is_fake and text_high_confidence:
                final_label = "Possibly Misleading"
                base_score = 45
                reason = "The text analysis suggests this content may be unreliable. Image context was not available for verification. Additional verification through reliable sources is recommended."
            elif text_is_fake:
                final_label = "Unverified"
                base_score = 60
                reason = "The text analysis shows some characteristics of unreliable content, but confidence is moderate. Image context was not available. Verification through additional sources is recommended."
            else:
                final_label = "Likely Credible"
                base_score = 75
                reason = "The text analysis suggests this content appears credible. However, image context was not available for verification. Always verify important claims through multiple reliable sources."
        
        # Calculate final_trust_score
        if has_image and image_mismatch:
            similarity_penalty = max(0, (0.20 - image_similarity) * 100)
            final_trust_score = max(0, base_score - similarity_penalty)
        elif has_image:
            confidence_bonus = (text_confidence - 0.5) * 40
            final_trust_score = min(100, base_score + confidence_bonus)
        else:
            confidence_bonus = (text_confidence - 0.5) * 30
            final_trust_score = min(100, base_score + confidence_bonus)
        
        final_trust_score = round(final_trust_score)
        
        response_time_ms = (time.time() - start_time) * 1000
        
        result = {
            'final_label': final_label,
            'final_trust_score': final_trust_score,
            'reason': reason,
            'verification': verification
        }
        
        # Add image context to result if available
        if has_image and image_context:
            result['image_context'] = image_context
            result['similarity_score'] = image_similarity
        
        # Store in cache (include image context in cache)
        cache_entry = result.copy()
        cache[cache_key] = cache_entry
        
        result['cache_hit'] = False
        result['response_time_ms'] = round(response_time_ms, 2)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "API Active",
        "models_loaded": True
    })

if __name__ == "__main__":
    app.run()
