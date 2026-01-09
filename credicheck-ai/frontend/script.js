// CredCheck AI - Frontend Logic
const API_BASE_URL = 'http://127.0.0.1:5000';

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    const analyzeButton = document.getElementById('analyze-button');
    const analyzeText = document.getElementById('analyze-text');
    const analyzeLoading = document.getElementById('analyze-loading');
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    const errorContainer = document.getElementById('error-container');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const removeImageButton = document.getElementById('remove-image');

    // State
    let selectedImageFile = null;

    // --- Event Listeners ---

    // Image Upload
    if (imageInput) {
        imageInput.addEventListener('change', handleImageSelect);
    }

    // Remove Image
    if (removeImageButton) {
        removeImageButton.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            clearImageSelection();
        });
    }

    // Analyze Button
    if (analyzeButton) {
        analyzeButton.addEventListener('click', handleAnalyze);
    }

    // --- Handlers ---

    function handleImageSelect(event) {
        const file = event.target.files[0];
        if (file) {
            if (file.size > 5 * 1024 * 1024) {
                showError('Image size must be less than 5MB');
                return;
            }

            selectedImageFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
                uploadPlaceholder.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }
    }

    function clearImageSelection() {
        selectedImageFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        imagePreviewContainer.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
    }

    async function handleAnalyze() {
        const text = textInput.value.trim();

        // Validation
        if (!text && !selectedImageFile) {
            showError('Please provide text or an image to analyze.');
            return;
        }

        if (text.length > 0 && text.length < 20) {
            showError('Text is too short. Please provide at least 20 characters.');
            return;
        }

        // Reset UI
        hideError();
        resultsSection.classList.add('hidden');
        loadingState.classList.remove('hidden');

        // Button Loading State
        analyzeButton.disabled = true;
        analyzeText.textContent = 'Analyzing...';
        analyzeLoading.classList.remove('hidden');

        try {
            // Prepare Payload
            let imageData = null;
            if (selectedImageFile) {
                imageData = await readFileAsBase64(selectedImageFile);
            }

            const payload = {
                text: text,
                headline: text.split('\n')[0].substring(0, 100), // Simple headline extraction
                image_path: null // Backend expects path or data
            };

            if (imageData) {
                payload.image_data = imageData;
            }

            // Call API
            const response = await fetch(`${API_BASE_URL}/analyze-full`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            // Render Results
            renderResults(data);

        } catch (error) {
            console.error('Analysis failed:', error);
            showError('Analysis failed. Please try again later. (' + error.message + ')');
        } finally {
            loadingState.classList.add('hidden');
            analyzeButton.disabled = false;
            analyzeText.textContent = 'Analyze Credibility';
            analyzeLoading.classList.add('hidden');
        }
    }

    // --- Rendering ---

    function renderResults(data) {
        resultsSection.classList.remove('hidden');

        // 1. Verdict Card
        const verdictBadge = document.getElementById('verdict-badge');
        const verdictSummary = document.getElementById('verdict-summary');
        const verdictColorBar = document.getElementById('verdict-color-bar');
        const confidenceValue = document.getElementById('confidence-value');
        const confidenceBar = document.getElementById('confidence-bar');
        const cacheBadge = document.getElementById('cache-badge');

        // Determine Verdict State
        const label = data.final_label || 'Unknown';
        const score = data.final_trust_score || 0;
        // Map backend 'reason' to UI 'explanation'
        const summary = data.explanation || data.reason || 'No explanation provided.';

        let badgeClass = 'bg-gray-100 text-gray-600';
        let barClass = 'bg-gray-600';

        if (label.toLowerCase().includes('fake') || label.toLowerCase().includes('false') || label.toLowerCase().includes('risk')) {
            badgeClass = 'bg-red-100 text-red-700 border border-red-200 badge-pulse-red';
            barClass = 'bg-danger';
        } else if (label.toLowerCase().includes('credible') || label.toLowerCase().includes('real') || label.toLowerCase().includes('true')) {
            badgeClass = 'bg-green-100 text-green-700 border border-green-200 badge-pulse-green';
            barClass = 'bg-success';
        } else {
            badgeClass = 'bg-amber-100 text-amber-700 border border-amber-200 badge-pulse-amber';
            barClass = 'bg-warning';
        }

        verdictBadge.textContent = label;
        verdictBadge.className = `px-4 py-1 rounded-full text-sm font-black uppercase tracking-wide transition-all duration-500 ${badgeClass}`;
        verdictColorBar.className = `absolute left-0 top-0 bottom-0 w-2 transition-colors duration-500 ${barClass}`;
        verdictSummary.textContent = summary;

        // Confidence
        // Estimate confidence if not provided directly
        const confidence = Math.abs(score - 50) * 2;
        confidenceValue.textContent = `${Math.round(confidence)}%`;

        // Animate Confidence Bar
        setTimeout(() => {
            confidenceBar.style.width = `${confidence}%`;
            confidenceBar.className = `bg-primary h-2.5 rounded-full transition-all duration-1000 ${barClass}`;
        }, 100);

        // Cache Hit
        if (data.cache_hit) {
            cacheBadge.classList.remove('hidden');
            cacheBadge.classList.add('inline-flex');
        } else {
            cacheBadge.classList.add('hidden');
            cacheBadge.classList.remove('inline-flex');
        }

        // 2. Trust Score Animation
        const trustScoreValue = document.getElementById('trust-score-value');
        const trustScoreCircle = document.getElementById('trust-score-circle');
        const trustScoreLabel = document.getElementById('trust-score-label');

        // Reset animation
        trustScoreCircle.style.strokeDashoffset = 376.99;

        // Animate
        setTimeout(() => {
            const offset = 376.99 - (376.99 * score) / 100;
            trustScoreCircle.style.strokeDashoffset = offset;

            // Color based on score
            trustScoreCircle.classList.remove('text-success', 'text-warning', 'text-danger');
            if (score >= 70) trustScoreCircle.classList.add('text-success');
            else if (score >= 40) trustScoreCircle.classList.add('text-warning');
            else trustScoreCircle.classList.add('text-danger');
        }, 100);

        // Counter animation
        animateValue(trustScoreValue, 0, score, 1500);

        trustScoreLabel.textContent = score >= 70 ? 'High Credibility' : (score >= 40 ? 'Questionable' : 'Low Credibility');

        // 3. Explanation Panel
        const explanationContent = document.getElementById('explanation-content');
        // Handle list if it exists (user requirement), otherwise string
        if (Array.isArray(data.explanation) && data.explanation.length > 0) {
            let html = '<div class="flex flex-wrap gap-2">';
            data.explanation.forEach(item => {
                // Assuming item is {word: "...", score: ...} or just string
                const word = typeof item === 'object' ? item.word : item;
                html += `<span class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-medium">${word}</span>`;
            });
            html += '</div>';
            explanationContent.innerHTML = html;
        } else {
            explanationContent.innerHTML = `<p class="text-sm leading-relaxed">${summary}</p>`;
        }

        // 4. Image Context
        const imageContextContent = document.getElementById('image-context-content');
        if (imageContextContent) {
            if (data.image_context && data.image_context !== "No Image") {
                let statusColor = 'text-gray-600';
                let statusIcon = 'image';

                if (data.image_context.includes('Consistent')) {
                    statusColor = 'text-success';
                    statusIcon = 'check_circle';
                } else if (data.image_context.includes('Mismatch')) {
                    statusColor = 'text-danger';
                    statusIcon = 'error';
                }

                imageContextContent.innerHTML = `
                    <div class="flex flex-col items-center justify-center text-center h-full animate-in fade-in">
                        <span class="material-symbols-outlined text-3xl ${statusColor} mb-2">${statusIcon}</span>
                        <p class="font-bold ${statusColor}">${data.image_context}</p>
                        ${data.similarity_score !== undefined ? `
                            <div class="w-full max-w-[120px] mt-2">
                                <div class="flex justify-between text-[10px] text-subtext-light mb-1">
                                    <span>Similarity</span>
                                    <span>${(data.similarity_score * 100).toFixed(0)}%</span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-1.5">
                                    <div class="bg-primary h-1.5 rounded-full" style="width: ${(data.similarity_score * 100)}%"></div>
                                </div>
                            </div>
                        ` : ''}
                    </div>
                `;
            } else {
                imageContextContent.innerHTML = `
                    <div class="flex flex-col items-center justify-center h-40 text-center">
                        <span class="material-symbols-outlined text-4xl text-gray-300 dark:text-gray-600 mb-2">hide_image</span>
                        <p class="text-subtext-light dark:text-subtext-dark text-sm">No image analyzed</p>
                    </div>
                `;
            }
        }

        // 5. Source Verification
        const sourceContent = document.getElementById('source-verification-content');
        if (sourceContent) {
            // Check for verification object or legacy note
            const verification = data.verification;
            let verificationNote = data.verification_note;
            let articles = [];

            if (!verificationNote && verification) {
                if (verification.source_coverage_level) {
                    verificationNote = verification.source_coverage_level;
                }
                if (verification.articles) {
                    articles = verification.articles;
                }
            }

            if (verificationNote) {
                let icon = 'info';
                let color = 'text-blue-600';
                let bg = 'bg-blue-50';

                // Determine status based on text content
                const lowerNote = verificationNote.toLowerCase();
                if (lowerNote.includes('verified') || lowerNote.includes('tier 1') || lowerNote.includes('tier 2')) {
                    icon = 'verified';
                    color = 'text-success';
                    bg = 'bg-green-50';
                } else if (lowerNote.includes('unverified') || lowerNote.includes('no coverage')) {
                    icon = 'warning';
                    color = 'text-warning';
                    bg = 'bg-amber-50';
                }

                let html = `
                    <div class="flex items-start gap-3 p-3 rounded-lg ${bg} dark:bg-opacity-10 animate-in fade-in">
                        <span class="material-symbols-outlined ${color} mt-0.5 text-lg">${icon}</span>
                        <div>
                            <p class="text-sm font-bold text-text-light dark:text-text-dark mb-1">Source Status</p>
                            <p class="text-xs text-subtext-light dark:text-subtext-dark leading-relaxed font-medium">${verificationNote}</p>
                        </div>
                    </div>
                `;

                // Add articles list if available
                if (articles.length > 0) {
                    html += `<div class="mt-3 pl-2 border-l-2 border-border-light dark:border-border-dark space-y-2">`;
                    articles.slice(0, 3).forEach(article => {
                        html += `
                            <a href="${article.url}" target="_blank" class="block text-xs group">
                                <span class="font-bold text-text-light dark:text-text-dark group-hover:text-primary transition-colors">${article.source}</span>
                                <span class="block text-subtext-light dark:text-subtext-dark truncate">${article.title}</span>
                            </a>
                        `;
                    });
                    html += `</div>`;
                }

                sourceContent.innerHTML = html;
            } else {
                sourceContent.innerHTML = `<p class="text-sm text-subtext-light">No source verification data.</p>`;
            }
        }

        // 6. Response Time
        const responseTime = document.getElementById('response-time');
        if (responseTime && data.response_time_ms) {
            responseTime.textContent = `Response time: ${data.response_time_ms}ms`;
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // --- Utilities ---

    function showError(message) {
        errorContainer.innerHTML = `
            <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded-xl flex items-center gap-3 animate-in fade-in slide-in-from-bottom-2 shadow-sm">
                <span class="material-symbols-outlined">error</span>
                <p class="text-sm font-medium">${message}</p>
            </div>
        `;
        errorContainer.classList.remove('hidden');
    }

    function hideError() {
        errorContainer.classList.add('hidden');
        errorContainer.innerHTML = '';
    }

    function readFileAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start) + "%";
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
