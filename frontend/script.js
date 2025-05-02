let currentStatus = "Ready";
let statusEventSource = null;

function toggleTheme() {
    console.log('Toggle theme called');
    const html = document.documentElement;
    console.log('Current dark class:', html.classList.contains('dark'));

    if (html.classList.contains('dark')) {
        html.classList.remove('dark');
        localStorage.setItem('theme', 'light');
        console.log('Removed dark class');
    } else {
        html.classList.add('dark');
        localStorage.setItem('theme', 'dark');
        console.log('Added dark class');
    }

    console.log('New dark class state:', html.classList.contains('dark'));
}

// Check for saved theme preference or use system preference
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    // Remove dark mode by default
    document.documentElement.classList.remove('dark');

    // Add event listener to theme toggle button
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // Check if theme is stored in localStorage
    const savedTheme = localStorage.getItem('theme');
    console.log('Saved theme:', savedTheme);

    if (savedTheme === 'dark') {
        // Apply saved theme
        document.documentElement.classList.add('dark');
    } else if (!savedTheme) {
        // Check system preference
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        }
    }

    console.log('Initial dark mode state:', document.documentElement.classList.contains('dark'));
});

function startStatusStream() {
    if (statusEventSource) {
        statusEventSource.close();
    }

    statusEventSource = new EventSource('http://localhost:8000/status-stream');

    statusEventSource.onmessage = function (event) {
        updateStatus(event.data);
    };

    statusEventSource.onerror = function (error) {
        console.error('EventSource failed:', error);
        statusEventSource.close();
    };
}

function findLoops() {
    const fileInput = document.getElementById('audioFileInput');
    if (!fileInput || !fileInput.files[0]) {
        updateStatus('Please select an audio file');
        return;
    }

    startStatusStream(); // Start listening for status updates
    updateStatus('Uploading file...');

    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);

    fetch('http://localhost:8000/find-loop', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                updateStatus(`Error: ${data.error}`);
            }
            // Handle the response data
            console.log(data);
        })
        .catch(error => {
            updateStatus(`Error: ${error.message}`);
            console.error('Error:', error);
        })
        .finally(() => {
            if (statusEventSource) {
                statusEventSource.close();
            }
        });
}

function updateStatus(newStatus) {
    // Update current status
    const statusElement = document.getElementById('status');
    if (statusElement) {
        statusElement.textContent = `Status: ${newStatus}`;
    }

    // Add to history
    const historyElement = document.getElementById('statusHistory');
    if (historyElement) {
        const statusEntry = document.createElement('div');
        statusEntry.textContent = `${new Date().toLocaleTimeString()}: ${newStatus}`;
        historyElement.insertBefore(statusEntry, historyElement.firstChild);

        // Optional: Limit history to last 50 entries
        while (historyElement.children.length > 50) {
            historyElement.removeChild(historyElement.lastChild);
        }

        // Auto-scroll to top
        historyElement.scrollTop = 0;
    }
}
