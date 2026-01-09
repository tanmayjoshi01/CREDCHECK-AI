// CrediCheck AI - Backend Integration
// Updated for XAI (Explainable AI) Highlights

const API_BASE_URL = 'http://127.0.0.1:5000';

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    const analyzeButton = document.getElementById('analyze-button');
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    
    if (analyzeButton) {
        analyzeButton.addEventListener('click', handleAnalyze);
    }
    
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
        if (imageStatus) {
            imageStatus.innerHTML = `<p class="text-sm text-green-600 dark:text-green-400 font-medium">Image ready: ${file.name}</p>`;
        }
        if (imagePreview) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.innerHTML = `<img src="${e.target.result}" class="max-w-32 max-h-32 mx-auto rounded border shadow-sm">`;
            };
            reader.readAsDataURL(file);
        }
        if (uploadArea) {
            uploadArea.querySelectorAll('p').forEach(p => p.style.display = 'none');
        }
    }
}

function handleAnalyze() {
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    const analyzeButton = document.getElementById('analyze-button');
    
    const text = textInput ? textInput.value.trim() : '';
    const imageFile = imageInput ? imageInput.files[0] : null;
    
    if (!text) {
        showError('Please enter text to analyze.');
        return;
    }
    
    if (analyzeButton) {
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="loading"></span> Processing...';
    }
    
    // Payload for the /analyze-full endpoint
    const payload = {
        text: text,
        headline: text.split('\n')[0], // Use first line as headline
        image_path: imageFile ? imageFile.name : "" 
    };
    
    fetch(`${API_BASE_URL}/analyze-full`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Could not connect to AI Server. Please ensure app.py is running.');
    })
    .finally(() => {
        if (analyzeButton) {
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = '<span class="material-symbols-outlined">query_stats</span><span class="truncate">Analyze & Verify</span>';
        }
    });
}

function displayResults(data) {
    // 1. XAI Text Highlighting Logic
    const textAnalysisDiv = document.getElementById('text-analysis-result');
    if (textAnalysisDiv && data.explanation) {
        let highlightHTML = `<div class="space-y-2">`;
        data.explanation.forEach(item => {
            let colorClass = "text-text-light dark:text-text-dark"; // Default
            
            // Red highlight if word contributes to "Fake" label (Positive impact)
            if (item.score > 0.01) {
                colorClass = "bg-red-200 dark:bg-red-900/50 text-red-900 dark:text-red-100 px-1 rounded";
            } 
            // Green highlight if word contributes to "Real" label (Negative impact)
            else if (item.score < -0.01) {
                colorClass = "bg-green-200 dark:bg-green-900/50 text-green-900 dark:text-green-100 px-1 rounded";
            }
            
            highlightHTML += `<span class="xai-word ${colorClass} mx-0.5" title="Impact: ${item.score}">${item.word}</span> `;
        });
        highlightHTML += `</div>`;
        textAnalysisDiv.innerHTML = highlightHTML;
    }

    // 2. Image Context Result
    const imageContextDiv = document.getElementById('image-context-result');
    if (imageContextDiv) {
        if (data.image_context) {
            const isMatch = data.image_context === "Consistent";
            imageContextDiv.innerHTML = `
                <div>
                    <p class="font-bold ${isMatch ? 'text-green-500' : 'text-red-500'}">${data.image_context}</p>
                    <p class="text-xs text-subtext-light">Similarity: ${(data.similarity_score * 100).toFixed(1)}%</p>
                </div>`;
        } else {
            imageContextDiv.innerHTML = `<p class="text-sm text-subtext-light">No image provided.</p>`;
        }
    }

    // 3. Source Verification
    const sourceDiv = document.getElementById('source-verification-result');
    if (sourceDiv) {
        const isVerified = data.verification_note && data.verification_note.includes("Verified");
        sourceDiv.innerHTML = `
            <p class="text-sm ${isVerified ? 'text-green-500 font-bold' : 'text-subtext-light'}">
                ${data.verification_note || "No source data."}
            </p>`;
    }

    // 4. Final Verdict Result
    const finalVerdictDiv = document.getElementById('final-verdict-result');
    if (finalVerdictDiv) {
        const score = data.final_trust_score || 0;
        const color = score > 70 ? 'text-green-500' : (score > 40 ? 'text-yellow-500' : 'text-red-500');
        
        finalVerdictDiv.innerHTML = `
            <div class="flex flex-col items-center">
                <div class="text-5xl font-black ${color} mb-2">${score}%</div>
                <p class="text-xl font-bold uppercase tracking-wider">${data.final_label}</p>
                <p class="mt-4 text-sm text-subtext-light max-w-md mx-auto leading-relaxed italic">
                    "${data.reason}"
                </p>
                ${data.cache_hit ? '<span class="mt-2 text-[10px] opacity-50">Retrieved from Cache</span>' : ''}
            </div>`;
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'fixed bottom-4 right-4 bg-red-600 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-bounce';
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);
    setTimeout(() => errorDiv.remove(), 4000);
}