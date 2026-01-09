from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

clip_model = SentenceTransformer('clip-ViT-B-32')

def analyze_image(image_path, headline_text):
    image = Image.open(image_path)
    
    image_embedding = clip_model.encode(image)
    text_embedding = clip_model.encode(headline_text)
    
    image_embedding = image_embedding.reshape(1, -1)
    text_embedding = text_embedding.reshape(1, -1)
    
    similarity = cosine_similarity(image_embedding, text_embedding)[0][0]
    
    if similarity > 0.25:
        context_status = "Consistent"
    else:
        context_status = "Suspicious Mismatch"
    
    return {
        "context_status": context_status,
        "similarity_score": float(similarity)
    }

def verify_image_context(image_path, headline_text):
    image = Image.open(image_path)
    
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

