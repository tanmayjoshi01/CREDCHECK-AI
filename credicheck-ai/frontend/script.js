// CrediCheck AI - Backend Integration
// Backend integration for Day 2

const API_BASE_URL = 'http://127.0.0.1:5000';

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    const analyzeButton = document.getElementById('analyze-button');
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    
    if (analyzeButton) {
        analyzeButton.addEventListener('click', handleAnalyze);
    }
    
    // Handle image upload feedback
    if (imageInput) {
        imageInput.addEventListener('change', handleImageChange);
    }
});

function handleImageChange(event) {
    const imageInput = event.target;
    const imageStatus = document.getElementById('image-status');
    const imagePreview = document.getElementById('image-preview');
    const uploadArea = imageInput.closest('.flex.flex-col').querySelector('.rounded-lg.border-2');
    
    if (imageInput.files && imageInput.files[0]) {
        const file = imageInput.files[0];
        const fileName = file.name;
        
        // Show filename
        if (imageStatus) {
            imageStatus.innerHTML = `<p class="text-sm text-green-600 dark:text-green-400 font-medium">Image uploaded: ${fileName}</p>`;
        }
        
        // Show small preview
        if (imagePreview) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview" class="max-w-32 max-h-32 mx-auto rounded border border-border-light dark:border-border-dark">`;
            };
            reader.readAsDataURL(file);
        }
        
        // Hide placeholder text
        if (uploadArea) {
            const placeholderText = uploadArea.querySelector('p.text-base');
            const subText = uploadArea.querySelector('p.text-sm');
            if (placeholderText) placeholderText.style.display = 'none';
            if (subText) subText.style.display = 'none';
        }
    } else {
        // Clear status and preview
        if (imageStatus) {
            imageStatus.innerHTML = '';
        }
        if (imagePreview) {
            imagePreview.innerHTML = '';
        }
        
        // Restore placeholder text
        if (uploadArea) {
            const placeholderText = uploadArea.querySelector('p.text-base');
            const subText = uploadArea.querySelector('p.text-sm');
            if (placeholderText) placeholderText.style.display = 'block';
            if (subText) subText.style.display = 'block';
        }
    }
}

function handleAnalyze() {
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    const analyzeButton = document.getElementById('analyze-button');
    
    // Get input values
    const text = textInput ? textInput.value.trim() : '';
    const imageFile = imageInput ? imageInput.files[0] : null;
    
    // Validate input
    if (!text) {
        showError('Please enter text to analyze.');
        return;
    }
    
    // Disable button and show loading
    if (analyzeButton) {
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="loading"></span> Analyzing...';
    }
    
    // Function to send request after image is processed (if present)
    const sendRequest = (imageData) => {
        // Prepare payload
        const payload = {
            text: text,
            headline: text // Use text as headline if not provided separately
        };
        
        // Add image data if file is selected
        if (imageData) {
            payload.image_data = imageData;
            payload.image_filename = imageFile.name;
        }
        
        // Send request to backend
        fetch(`${API_BASE_URL}/analyze-full`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Backend request failed');
                });
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.message || 'Failed to connect to backend. Make sure the server is running.');
        })
        .finally(() => {
            // Re-enable button
            if (analyzeButton) {
                analyzeButton.disabled = false;
                analyzeButton.innerHTML = '<span class="material-symbols-outlined">query_stats</span><span class="truncate">Analyze</span>';
            }
        });
    };
    
    // Process image if present, otherwise send request immediately
    if (imageFile) {
        // Convert image to base64
        const reader = new FileReader();
        reader.onload = function(e) {
            // Remove data URL prefix (e.g., "data:image/jpeg;base64,") if you want just the base64 string
            // Or keep it if backend expects full data URL
            sendRequest(e.target.result);
        };
        reader.onerror = function() {
            showError('Failed to read image file.');
            if (analyzeButton) {
                analyzeButton.disabled = false;
                analyzeButton.innerHTML = '<span class="material-symbols-outlined">query_stats</span><span class="truncate">Analyze</span>';
            }
        };
        reader.readAsDataURL(imageFile);
    } else {
        // No image, send request immediately
        sendRequest(null);
    }
}

function displayResults(data) {
    // Display text analysis (from aggregated result)
    const textAnalysisDiv = document.getElementById('text-analysis-result');
    if (textAnalysisDiv) {
        let html = '';
        if (data.final_label) {
            html += `<p class="font-semibold">Classification: ${data.final_label}</p>`;
        }
        if (data.final_trust_score !== undefined) {
            html += `<p>Trust Score: ${data.final_trust_score}%</p>`;
        }
        if (data.cache_hit !== undefined) {
            html += `<p class="text-sm mt-2">Cache: ${data.cache_hit ? 'Hit' : 'Miss'}</p>`;
        }
        textAnalysisDiv.innerHTML = html || '<p>No text analysis data available.</p>';
    }
    
    // Display image context (if available in response)
    const imageContextDiv = document.getElementById('image-context-result');
    if (imageContextDiv) {
        let html = '';
        
        // Check if image analysis was actually performed by checking for image_context field
        const hasImageContext = data.image_context !== undefined && 
                                data.image_context !== null && 
                                data.image_context !== '';
        
        if (hasImageContext) {
            html += `<p class="text-sm font-medium text-text-light dark:text-text-dark">Image analysis performed</p>`;
            
            // Show specific status based on image_context value
            if (data.image_context === "Mismatch" || data.image_context === "Suspicious Mismatch") {
                html += `<p class="text-sm mt-1 text-orange-600 dark:text-orange-400">⚠ Image–headline mismatch detected</p>`;
            } else if (data.image_context === "Consistent") {
                html += `<p class="text-sm mt-1 text-green-600 dark:text-green-400">✓ Image matches the headline</p>`;
            } else {
                html += `<p class="text-sm mt-1">${data.image_context}</p>`;
            }
            
            // Show similarity score if available
            if (data.similarity_score !== undefined && data.similarity_score !== null) {
                html += `<p class="text-xs mt-1 text-subtext-light dark:text-subtext-dark">Similarity: ${(data.similarity_score * 100).toFixed(1)}%</p>`;
            }
        } else {
            // Check if an image was actually uploaded by checking the image input
            const imageInput = document.getElementById('image-input');
            const imageFile = imageInput ? imageInput.files[0] : null;
            
            if (imageFile) {
                // Image was uploaded but analysis failed or wasn't performed
                html += `<p class="text-sm text-orange-600 dark:text-orange-400">⚠ Image uploaded but analysis unavailable</p>`;
            } else {
                // No image was provided
                html += `<p class="text-sm text-subtext-light dark:text-subtext-dark">No image provided</p>`;
            }
        }
        
        imageContextDiv.innerHTML = html;
    }
    
    // Display source verification with dynamic messaging
    const sourceVerificationDiv = document.getElementById('source-verification-result');
    if (sourceVerificationDiv) {
        let html = '';
        
        // Check if image was analyzed
        const hasImageContext = data.image_context !== undefined && data.image_context !== null && data.image_context !== '';
        const imageMismatch = hasImageContext && (data.image_context === "Mismatch" || data.image_context === "Suspicious Mismatch");
        
        // CASE 1: Verified by trusted source
        if (data.verification_note && data.verification_note.includes("Verified by")) {
            const sourceMatch = data.verification_note.match(/Verified by (.+)/);
            const source = sourceMatch ? sourceMatch[1] : "trusted source";
            html += `<p class="text-sm font-bold text-green-600 dark:text-green-400">✔ Claim verified by trusted source(s): ${source}</p>`;
        }
        // CASE 2: Unverified claim without image or with consistent image
        else if (data.verification_note === "Unverified claim" && !imageMismatch) {
            html += `<p class="text-sm text-text-light dark:text-text-dark">Text style appears neutral, but no confirmation was found from major trusted sources yet.</p>`;
            html += `<p class="text-xs mt-1 text-subtext-light dark:text-subtext-dark italic">Breaking or regional news may take time to appear on global outlets.</p>`;
        }
        // CASE 3: Unverified claim with image mismatch
        else if (data.verification_note === "Unverified claim" && imageMismatch) {
            html += `<p class="text-sm text-text-light dark:text-text-dark">Claim is not confirmed by trusted sources and the image does not align with the text.</p>`;
        }
        // Fallback: show verification_note as-is if it exists
        else if (data.verification_note) {
            html += `<p class="text-sm">${data.verification_note}</p>`;
        }
        // Fallback: show reason if no verification_note
        else if (data.reason) {
            html += `<p class="text-sm">${data.reason}</p>`;
        }
        
        // CASE 4: Append cache note if cached
        if (data.cached === true || data.cache_hit === true) {
            html += `<p class="text-xs mt-2 text-subtext-light dark:text-subtext-dark italic">Previously analyzed result (cache hit).</p>`;
        }
        
        // Show response time if available
        if (data.response_time_ms !== undefined) {
            html += `<p class="text-xs mt-2 text-gray-500 dark:text-gray-400">Response: ${data.response_time_ms}ms</p>`;
        }
        
        sourceVerificationDiv.innerHTML = html || '<p>No verification data available.</p>';
    }
    
    // Display final verdict
    const finalVerdictDiv = document.getElementById('final-verdict-result');
    if (finalVerdictDiv) {
        let html = '';
        if (data.final_label) {
            html += `<p class="text-2xl font-bold mb-2">${data.final_label}</p>`;
        }
        if (data.final_trust_score !== undefined) {
            const scoreColor = data.final_trust_score >= 70 ? 'text-green-600 dark:text-green-400' : 
                             data.final_trust_score >= 50 ? 'text-yellow-600 dark:text-yellow-400' : 
                             'text-red-600 dark:text-red-400';
            html += `<p class="text-xl font-semibold ${scoreColor}">Trust Score: ${data.final_trust_score}%</p>`;
        }
        if (data.reason) {
            html += `<p class="text-base mt-2">${data.reason}</p>`;
        }
        finalVerdictDiv.innerHTML = html || '<p>No verdict data available.</p>';
    }
}

function showError(message) {
    // Remove existing error messages
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Create error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    // Insert before results section
    const resultsSection = document.querySelector('.space-y-8');
    if (resultsSection) {
        resultsSection.parentNode.insertBefore(errorDiv, resultsSection);
    } else {
        // Fallback: insert at top of main content
        const main = document.querySelector('main');
        if (main) {
            main.insertBefore(errorDiv, main.firstChild);
        }
    }
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

