// AI Resume Analyzer - Main JavaScript

// 💡 Theme Toggle Functionality
document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;

    const savedTheme = localStorage.getItem('theme') || 'light';
    htmlElement.setAttribute('data-bs-theme', savedTheme);
    if (themeToggle) {
        themeToggle.checked = savedTheme === 'dark';
    }

    if (themeToggle) {
        themeToggle.addEventListener('change', function() {
            const theme = this.checked ? 'dark' : 'light';
            htmlElement.setAttribute('data-bs-theme', theme);
            localStorage.setItem('theme', theme);
        });
    }

    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn && themeBtn.classList.contains('theme-btn')) {
        themeBtn.addEventListener('click', function() {
            const currentTheme = htmlElement.getAttribute('data-bs-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            htmlElement.setAttribute('data-bs-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
});

// 📂 File Upload Handler
const fileInput = document.getElementById('resume_files');
const dropZone = document.getElementById('dropZone');
const fileList = document.getElementById('fileList');

if (fileInput && dropZone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        displayFiles(files);
    }, false);

    fileInput.addEventListener('change', function() {
        displayFiles(this.files);
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function displayFiles(files) {
    if (!fileList) return;

    fileList.innerHTML = '';
    if (files.length === 0) {
        return;
    }

    const fileArray = Array.from(files);
    fileArray.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item d-flex align-items-center justify-content-between p-2 mb-2 bg-light rounded';
        fileItem.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-file-alt text-primary me-2"></i>
                <span>${file.name}</span>
                <small class="text-muted ms-2">(${formatFileSize(file.size)})</small>
            </div>
            <button type="button" class="btn btn-sm btn-outline-danger" onclick="removeFile(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        fileList.appendChild(fileItem);
    });
}

function removeFile(index) {
    const fileInput = document.getElementById('resume_files');
    const dt = new DataTransfer();
    const files = Array.from(fileInput.files);

    files.splice(index, 1);
    files.forEach(file => dt.items.add(file));

    fileInput.files = dt.files;
    displayFiles(fileInput.files);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// 📋 Form Validation
const resumeForm = document.getElementById('resumeForm');
if (resumeForm) {
    resumeForm.addEventListener('submit', function(e) {
        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        }
    });
}

/* ================= Email: Safe, Idempotent, Type-Guarded ================ */

// Track sent emails to avoid repeating
const sentEmails = new Set(); // key: `${email}-${type}`
// Track in-flight requests to stop multi-click loops
const inFlightEmails = new Set(); // key: `${email}-${type}`

// Validate that email type matches the card group
function isTypeAllowedForCard(button, type) {
    const card = button.closest('.candidate-card');
    if (!card) return true; // fallback: allow

    const isShortlistedCard = card.classList.contains('shortlisted-card');
    const isNotSelectedCard = card.classList.contains('not-selected-card');

    if (type === 'congratulations' && !isShortlistedCard) return false;
    if (type === 'rejection' && !isNotSelectedCard) return false;

    return true;
}

function markButtonAsSent(button) {
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-check me-1"></i>Sent';
    button.classList.remove('btn-primary', 'btn-danger', 'btn-outline-danger', 'btn-outline-success', 'btn-success');
    button.classList.add('btn-secondary', 'btn-disabled-custom');
}

// Central Email Sending Function (debounced + guarded)
function sendEmail(email, name, type, button) {
    if (!email || email === 'Not provided') {
        showToast('No email address available for this candidate', 'warning');
        return;
    }
    if (!button) return;

    // Type guard (prevents wrong emails to wrong lists)
    if (!isTypeAllowedForCard(button, type)) {
        showToast(`Invalid action: cannot send "${type}" email for this candidate.`, 'warning');
        return;
    }

    const key = `${email}-${type}`;

    // Prevent duplicate sends if already sent
    if (sentEmails.has(key)) {
        showToast(`${type === 'congratulations' ? 'Congratulatory' : 'Rejection'} email already sent to ${name}`, 'info');
        markButtonAsSent(button);
        return;
    }

    // Debounce multiple clicks / loops
    if (inFlightEmails.has(key) || button.dataset.sending === '1') {
        showToast('Please wait, email is already being sent...', 'info');
        return;
    }

    // UI lock
    const originalText = button.innerHTML;
    button.disabled = true;
    button.dataset.sending = '1';
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Sending...';
    inFlightEmails.add(key);

    fetch('/send_email', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, name, type })
    })
    .then(response => response.json())
    .then(data => {
        // Clear in-flight
        inFlightEmails.delete(key);
        delete button.dataset.sending;

        if (data && data.success) {
            sentEmails.add(key);
            const msg = type === 'congratulations'
                ? `Congratulatory email sent to ${name}`
                : `Rejection email sent to ${name}`;
            showToast(msg, 'success');
            markButtonAsSent(button);
        } else {
            const reason = (data && data.message) ? `: ${data.message}` : '';
            showToast(`Failed to send email${reason}`, 'danger');
            button.disabled = false;
            button.innerHTML = originalText;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        inFlightEmails.delete(key);
        delete button.dataset.sending;

        showToast('Error sending email', 'danger');
        button.disabled = false;
        button.innerHTML = originalText;
    });
}

// Individual Email Functions
function sendCongratulate(email, name, button) {
    sendEmail(email, name, 'congratulations', button);
}

function sendRejection(email, name, button) {
    sendEmail(email, name, 'rejection', button);
}

// 🔔 Toast notification helper function
function showToast(message, type) {
    const toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) return;

    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div class="toast align-items-center text-bg-${type} border-0" role="alert" id="${toastId}">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);

    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();

    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}
