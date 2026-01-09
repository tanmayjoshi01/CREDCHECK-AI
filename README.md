CrediCheck AI – Backend
Problem Statement
The proliferation of fake news and misinformation presents a critical challenge in the digital age. Deepfakes, AI-generated content, and deliberately misleading information can spread rapidly across platforms, undermining public trust and decision-making. Traditional fact-checking methods are too slow to keep pace with viral content, while automated systems face challenges in distinguishing between authentic and manipulated media. This system addresses these challenges through a multi-signal analysis approach that combines linguistic pattern detection, visual-text consistency verification, and external source cross-checking to provide transparent credibility assessments.

Backend Architecture Overview
CrediCheck AI employs a 3-layer analysis system that provides comprehensive content verification:

1. Linguistic Analysis (Text Model)
Uses a fine-tuned RoBERTa model (hamzab/roberta-fake-news-classification) to analyze text patterns and classify content as "Real" or "Fake". The system also performs Named Entity Recognition (NER) using spaCy to extract key entities (PERSON, ORG, GPE) that may be referenced in the content.

2. Visual Context Verification (Image–Text Matching)
Leverages CLIP (Contrastive Language–Image Pre-training) via sentence-transformers to compute semantic similarity between images and associated headlines. This layer detects when images do not align with the textual claims, identifying potential mismatches that suggest manipulation or misleading content.

3. External Verification (Trusted Source Cross-Check)
Performs lightweight web searches using DuckDuckGo to verify if claims appear on trusted domains (BBC, Reuters, AP News, NPR, government, and educational institutions). This provides external validation signals without requiring full content scraping.

The system emphasizes multi-signal analysis and transparency, providing users with clear explanations of why content receives a particular credibility assessment rather than relying on a single binary classification.

Day-wise Backend Improvements
Day 1: Text Accuracy + Entity Extraction
Implemented RoBERTa-based fake news classification for improved text analysis accuracy
Integrated spaCy NER for extracting named entities (persons, organizations, geographic locations)
Enhanced text analysis output with structured entity information
Day 2: Image–Text Semantic Consistency
Integrated CLIP model for image-text semantic matching
Implemented cosine similarity calculation to detect image-headline misalignments
Added visual context verification as a separate analysis layer
Day 3: Performance Optimization via Global Model Loading
Refactored model loading to global scope for single-instance initialization
Models now load once at server startup, eliminating redundant loading on each request
Significant performance improvement for high-throughput scenarios
Day 4: External Trusted-Source Verification
Integrated DuckDuckGo search API for external verification
Implemented trusted domain checking (BBC, Reuters, AP News, NPR, gov, edu)
Added verification notes to text analysis responses
Day 5: In-Memory Caching for Scalability
Implemented MD5 hash-based caching for text analysis results
Added caching layer to reduce redundant computation for repeated queries
Improved response times for frequently analyzed content
Technologies & Libraries Used
Python (3.9+)
Flask – Web framework for REST API
Flask-CORS – Cross-origin resource sharing for frontend integration
HuggingFace Transformers – Model loading and inference
RoBERTa (hamzab/roberta-fake-news-classification) – Fake news text classification (auto-downloaded from HuggingFace)
spaCy (en_core_web_sm) – Named Entity Recognition (manual download required)
CLIP (sentence-transformers/clip-ViT-B-32) – Image-text semantic matching (auto-downloaded from HuggingFace)
DuckDuckGo Search – External verification queries
PyTorch (Torch) – Deep learning framework
scikit-learn – Cosine similarity calculations
Pillow (PIL) – Image processing
System Requirements
Python 3.9 or higher
Internet connection required for:
Initial model downloads from HuggingFace Hub (~1.1GB total)
External verification searches (DuckDuckGo API)
CPU-based execution (no GPU required)
Minimum 4GB RAM recommended for model loading
~2GB free disk space for model cache storage
Installation Instructions
Navigate to the backend directory:
cd backend
Install Python dependencies:
pip install -r requirements.txt
Install DuckDuckGo Search (required for external verification):
pip install duckduckgo-search
Download the spaCy English model (required for named entity recognition):
python -m spacy download en_core_web_sm
Model Downloads (Automatic): The following models are automatically downloaded from HuggingFace Hub on first run:

RoBERTa Fake News Classifier (hamzab/roberta-fake-news-classification) - ~500MB
CLIP Model (sentence-transformers/clip-ViT-B-32) - ~600MB
Note: First-time server startup will download these models automatically (requires internet connection). Models are cached locally after initial download. Total download size: ~1.1GB. Subsequent startups use cached models and do not require internet.

How to Run the Backend
Navigate to the backend directory:
cd backend
Start the Flask server:
python app.py
The API will be available at http://127.0.0.1:5000

Verify the server is running by checking the health endpoint:

curl http://127.0.0.1:5000/
Expected output:

{"status": "API Active", "models_loaded": true}
API Endpoints Documentation
GET /
Purpose: Health check endpoint to verify API status and model loading

Request: No parameters required

Example:

curl http://127.0.0.1:5000/
Response:

{
  "status": "API Active",
  "models_loaded": true
}
POST /analyze-text
Purpose: Analyze text content for fake news classification, extract entities, and perform external verification

Required JSON payload:

{
  "text": "Your news text here"
}
Example:

curl -X POST http://127.0.0.1:5000/analyze-text -H "Content-Type: application/json" -d "{\"text\":\"NASA confirms presence of water on Mars\"}"
Response:

{
  "label": "Real",
  "score": 0.85,
  "entities": ["NASA", "Mars"],
  "verification_note": "Verified by Government",
  "cached": false
}
POST /analyze-image
Purpose: Verify semantic alignment between an image and headline text

Required JSON payload:

{
  "image_path": "path/to/image.jpg",
  "headline": "Headline text describing the image"
}
Example:

curl -X POST http://127.0.0.1:5000/analyze-image -H "Content-Type: application/json" -d "{\"image_path\":\"forest_fire.jpg\",\"headline\":\"Floods devastate Mumbai\"}"
Response:

{
  "image_context": "Mismatch",
  "similarity_score": 0.11
}
POST /analyze-full
Purpose: Comprehensive analysis combining text classification, image-text verification, and aggregated trust scoring

Required JSON payload:

{
  "text": "News article text",
  "headline": "Headline text (optional, defaults to text)",
  "image_path": "path/to/image.jpg (optional)"
}
Example:

curl -X POST http://127.0.0.1:5000/analyze-full -H "Content-Type: application/json" -d "{\"text\":\"Breaking news article content\",\"headline\":\"Article headline\",\"image_path\":\"article_image.jpg\"}"
Response:

{
  "final_label": "Likely Credible",
  "final_trust_score": 82,
  "reason": "Both the text analysis and image-headline alignment suggest this content appears credible. However, always verify important claims through multiple reliable sources as automated analysis is not infallible.",
  "cache_hit": false,
  "response_time_ms": 1250.45
}
Example Outputs
Text Analysis with Entities and Verification
{
  "label": "Real",
  "score": 0.92,
  "entities": ["Joe Biden", "White House", "Washington"],
  "verification_note": "Verified by Reuters",
  "cached": false
}
Image Context Mismatch
{
  "image_context": "Mismatch",
  "similarity_score": 0.15
}
Cached Response Example
{
  "label": "Fake",
  "score": 0.78,
  "entities": ["Elon Musk"],
  "verification_note": "Unverified claim",
  "cached": true
}
Design Philosophy
CrediCheck AI is built on principles of transparency, multi-signal analysis, and decision support rather than absolute judgment:

Separation of Analysis Layers
Linguistic analysis and external verification operate as independent signals. This design allows the system to identify cases where AI pattern detection suggests one classification while external sources indicate another, providing users with a comprehensive view rather than a single verdict.

Non-Binary Decision Support
The system does not blindly override AI decisions with external verification. Instead, it presents all signals transparently, allowing users to understand both the pattern-based classification and the external validation status. This approach acknowledges that:

Trusted sources may not always be the first to report breaking news
AI pattern detection may identify characteristics that external sources do not yet cover
Multiple signals together provide more reliable assessment than any single indicator
Transparency Over Classification
Rather than providing a simple "real" or "fake" binary output, the system emphasizes explaining why content receives its assessment. The /analyze-full endpoint provides detailed reasoning, confidence scores, and source verification notes, enabling users to make informed decisions rather than blindly trusting an automated system.

Scalability & Performance Notes
Global Model Loading
All machine learning models (RoBERTa, CLIP, spaCy) are loaded once at server startup into global memory. This design eliminates redundant model loading overhead on each request, significantly improving response times for production deployments.

In-Memory Caching
Two caching layers optimize performance:

RESULTS_CACHE: Caches text analysis results using MD5 hash of input text
cache: Caches full analysis results for the /analyze-full endpoint
Repeated requests for identical content are served from cache, reducing computational load and improving response times. Cache hit responses typically return in under 10ms compared to 1-3 seconds for fresh analysis.

Efficient Request Handling
The system is designed for CPU-based execution, making it accessible without specialized hardware. Model inference uses PyTorch's optimized CPU operations, and the global model loading strategy ensures consistent performance under load.

Limitations & Future Scope
Current Limitations
Models are not retrained during hackathon period; system uses pre-trained models only
External verification relies on lightweight search results (first 3 results), not comprehensive fact-checking
Image analysis requires local file paths; URL-based image processing not yet implemented
Caching is in-memory only; server restart clears cache (no persistence)
Future Enhancements
Aggregated Trust Scoring: Develop more sophisticated aggregation algorithms that weight different signals (linguistic patterns, image consistency, external verification) into a unified trust score
URL-Based News Analysis: Extend image analysis to accept URLs, enabling web-based content verification workflows
Persistent Caching: Implement Redis or database-backed caching for cache persistence across server restarts
Batch Processing: Add endpoints for analyzing multiple articles/images in a single request
Confidence Calibration: Implement confidence score calibration to improve reliability of probability estimates
Multi-language Support: Extend NER and text analysis to support multiple languages beyond English
