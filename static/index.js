// --- Global App State ---
let activeCameraIndex = 0;
let activeCameraName = "Camera 1";
let currentRoiType = null; // 'tl' or 'people'
let roiPoints = [];
let isDrawingRoi = false;
let allCamerasData = []; // Cache for status updates
let globalSettings = {}; // Cache for settings

// --- Bootstrap Instances ---
let liveViewModal = null;
let drawZoneModal = null;
let activeTooltips = []; // To manage tooltips

// --- Bootstrap Badge Classes ---
const BADGE_CLASSES = {
  stop: 'badge text-bg-danger text-uppercase',
  go: 'badge text-bg-success text-uppercase',
  detected: 'badge text-bg-primary text-uppercase',
  none: 'badge text-bg-secondary text-uppercase',
  active: 'badge text-bg-success text-uppercase',
  inactive: 'badge text-bg-danger text-uppercase', // Changed to red for Disconnected
  default: 'badge bg-secondary-subtle text-secondary-emphasis',
  connecting: 'badge text-bg-secondary',
  recording: 'badge text-bg-danger',
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', function () {
  // Initialize Modals
  liveViewModal = new bootstrap.Modal(document.getElementById('liveViewModal'));
  drawZoneModal = new bootstrap.Modal(document.getElementById('drawZoneModal'));

  // Initialize Page Navigation
  initializeNavigation();

  // Initialize Theme Toggle
  initializeThemeToggle();

  // Initialize Dashboard Page
  initializeDashboard();

  // Initialize Settings Page
  initializeSettings();

  // Initialize Playback Page
  initializePlayback();

  // Start updating status from backend
  startStatusUpdates();

  // Initialize Low Power Mode
  initializeLowPowerMode();

  // Load settings initially to populate cache
  loadSettings();

  // Set default active camera
  const firstCard = document.querySelector('.camera-card');
  if (firstCard) {
    activeCameraIndex = parseInt(firstCard.dataset.cameraIndex);
    activeCameraName = firstCard.dataset.cameraName;
    document.getElementById('roi-cam-name').textContent = activeCameraName;
  }
});

// --- Page Initializers ---

let isAdminLoggedIn = false;

function initializeNavigation() {
  const navLinks = document.querySelectorAll('#main-nav .nav-link');
  const pages = {
    dashboard: document.getElementById('page-dashboard'),
    playback: document.getElementById('page-playback'),
    settings: document.getElementById('page-settings'),
  };

  // Admin Login Logic
  const loginModalEl = document.getElementById('adminLoginModal');
  const loginModal = loginModalEl ? new bootstrap.Modal(loginModalEl) : null;
  const loginBtn = document.getElementById('admin-login-btn');
  const usernameInput = document.getElementById('admin-username');
  const passwordInput = document.getElementById('admin-password');
  const loginError = document.getElementById('login-error');

  if (loginBtn) {
    loginBtn.addEventListener('click', () => {
      const username = usernameInput.value;
      const password = passwordInput.value;

      // Simple hardcoded check (as requested)
      // --- DEMO CREDENTIALS FOR PORTFOLIO ---
      // These are default credentials to allow access to the dashboard in this demo.
      if (username === 'admin' && password === 'spas@2024') {
        isAdminLoggedIn = true;
        loginModal.hide();
        // Clear inputs
        usernameInput.value = '';
        passwordInput.value = '';
        loginError.classList.add('d-none');

        // Trigger settings navigation
        const settingsLink = document.querySelector('a[data-page="settings"]');
        if (settingsLink) settingsLink.click();

        showNotification('Admin access granted', 'success');
      } else {
        loginError.classList.remove('d-none');
      }
    });
  }

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const page = link.getAttribute('data-page');

      // Check admin access for settings
      if (page === 'settings' && !isAdminLoggedIn) {
        if (loginModal) loginModal.show();
        return;
      }

      // Hide all pages
      Object.values(pages).forEach(p => p.classList.add('d-none'));

      // Show target page
      if (pages[page]) {
        pages[page].classList.remove('d-none');
      }

      // Update active link
      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      // Load data for page
      if (page === 'playback') {
        loadRecordings();
      } else if (page === 'settings') {
        loadSettings();
      }
    });
  });
}

function initializeThemeToggle() {
  const themeToggle = document.getElementById('theme-toggle');
  const themeToggleIcon = themeToggle.querySelector('i');

  const setLightMode = () => {
    document.documentElement.setAttribute('data-bs-theme', 'light');
    themeToggleIcon.classList.remove('fa-sun');
    themeToggleIcon.classList.add('fa-moon');
  };

  const setDarkMode = () => {
    document.documentElement.setAttribute('data-bs-theme', 'dark');
    themeToggleIcon.classList.remove('fa-moon');
    themeToggleIcon.classList.add('fa-sun');
  };

  // Set initial theme
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (prefersDark) {
    setDarkMode();
  } else {
    setLightMode();
  }

  themeToggle.addEventListener('click', () => {
    if (document.documentElement.getAttribute('data-bs-theme') === 'dark') {
      setLightMode();
    } else {
      setDarkMode();
    }
  });
}

function initializeDashboard() {
  // Camera card selection
  document.getElementById('camera-grid').addEventListener('click', (e) => {
    const card = e.target.closest('.camera-card');
    if (card && !e.target.closest('.expand-btn')) { // Don't select if expand is clicked
      activeCameraIndex = parseInt(card.dataset.cameraIndex);
      activeCameraName = card.dataset.cameraName;

      // Update active card styling
      document.querySelectorAll('.camera-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');

      // Update ROI card header
      document.getElementById('roi-cam-name').textContent = activeCameraName;
    }
  });

  // Expand button listener
  document.querySelectorAll('.expand-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const card = e.target.closest('.camera-card');
      const camIndex = card.dataset.cameraIndex;
      const camName = card.dataset.cameraName;

      document.getElementById('liveViewModalLabel').querySelector('span').textContent = camName;
      document.getElementById('live-view-stream-img').src = `/video_feed/${camIndex}?t=${Date.now()}`;
    });
  });

  // Stop stream when live view modal closes
  document.getElementById('liveViewModal').addEventListener('hidden.bs.modal', () => {
    document.getElementById('live-view-stream-img').src = '';
  });

  // FIX: Add listeners to hide placeholder icons on video load
  document.querySelectorAll('.camera-card').forEach(card => {
    const videoStream = card.querySelector('.video-stream');
    const placeholderIcon = card.querySelector('.placeholder-icon');

    if (videoStream && placeholderIcon) {
      // Hide placeholder on successful load
      videoStream.addEventListener('load', () => {
        placeholderIcon.style.display = 'none';
      });

      // Show placeholder if video fails to load
      videoStream.addEventListener('error', () => {
        placeholderIcon.style.display = 'block';
      });

      // Check if already loaded (for cached images)
      if (videoStream.complete && videoStream.naturalHeight > 0) {
        placeholderIcon.style.display = 'none';
      }
    }
  });

  // Initialize ROI Drawing Modal
  initializeRoiModal();
}

function initializeSettings() {
  // Settings form
  document.getElementById('settings-form').addEventListener('submit', saveSettings);
  document.getElementById('reset-settings').addEventListener('click', resetSettings);

  // Audio uploads
  document.getElementById('berhenti-upload').addEventListener('change', (e) => handleAudioUpload(e, 'berhenti'));
  document.getElementById('melintas-upload').addEventListener('change', (e) => handleAudioUpload(e, 'melintas'));

  // Audio reset
  document.getElementById('reset-berhenti-audio').addEventListener('click', () => resetAudio('berhenti'));
  document.getElementById('reset-melintas-audio').addEventListener('click', () => resetAudio('melintas'));

  // Hardware test buttons
  document.getElementById('test-speaker-go').addEventListener('click', () => testSpeaker('melintas'));
  document.getElementById('test-speaker-stop').addEventListener('click', () => testSpeaker('berhenti'));
  document.getElementById('test-led-on').addEventListener('click', () => testLED('on'));
  document.getElementById('test-led-off').addEventListener('click', () => testLED('off'));
  document.getElementById('test-led-blink').addEventListener('click', () => testLED('blink'));

  // Sliders
  setupSlider('confidence-threshold', 'confidence-display', '%');
  setupSlider('led-blink-interval', 'led-blink-display', 's');

  // Camera Config
  initializeCameraConfig();
}

function initializePlayback() {
  document.getElementById('refresh-recordings').addEventListener('click', loadRecordings);

  // Add listeners for filters
  document.getElementById('cam-filter').addEventListener('change', filterRecordings);
  document.getElementById('date-filter').addEventListener('change', filterRecordings);
}

function setupSlider(sliderId, displayId, unit) {
  const slider = document.getElementById(sliderId);
  const display = document.getElementById(displayId);
  if (slider && display) {
    slider.addEventListener('input', () => {
      display.textContent = slider.value + unit;
    });
    // Set initial value
    display.textContent = slider.value + unit;
  }
}

// --- Status Updates ---

function startStatusUpdates() {
  updateAllStatus(); // Run immediately
  // FIX: Slow down interval to 5 seconds (5000ms)
  setInterval(updateAllStatus, 5000);

  // Start sensor updates
  updateSensorData(); // Run immediately
  setInterval(updateSensorData, 5000); // Update every 5 seconds
}

function updateSensorData() {
  fetch(`/sensor_data?t=${Date.now()}`)
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        // Update temperature
        const tempEl = document.getElementById('sensor-temperature');
        if (data.temperature !== null) {
          tempEl.textContent = `${data.temperature}°C`;
          tempEl.classList.remove('text-muted');
        }

        // Update humidity
        const humidityEl = document.getElementById('sensor-humidity');
        if (data.humidity !== null) {
          humidityEl.textContent = `${data.humidity}%`;
          humidityEl.classList.remove('text-muted');
        }

        // Update voltage
        const voltageEl = document.getElementById('sensor-voltage');
        if (data.voltage !== null) {
          voltageEl.textContent = `${data.voltage}V`;
          voltageEl.classList.remove('text-muted');

          // Color code based on voltage level (12V battery)
          if (data.voltage < 11.5) {
            voltageEl.classList.add('text-danger');
            voltageEl.classList.remove('text-warning', 'text-success');
          } else if (data.voltage < 12.0) {
            voltageEl.classList.add('text-warning');
            voltageEl.classList.remove('text-danger', 'text-success');
          } else {
            voltageEl.classList.add('text-success');
            voltageEl.classList.remove('text-danger', 'text-warning');
          }
        } else {
          voltageEl.textContent = '--';
          voltageEl.classList.add('text-muted');
          voltageEl.classList.remove('text-danger', 'text-warning', 'text-success');
        }
      }
    })
    .catch(error => {
      console.error('Error fetching sensor data:', error);
    });
}

function updateAllStatus() {
  // FIX: Add a timeout controller
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4800); // Abort fetch if it takes too long

  fetch(`/status?t=${Date.now()}`, { signal: controller.signal })
    .then(response => {
      clearTimeout(timeoutId); // Clear timeout if fetch is successful
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    })
    .then(data => {
      allCamerasData = data.cameras; // Cache data
      let anyRecording = false;

      allCamerasData.forEach((cameraData, index) => {
        const recIndicator = document.getElementById(`recording-indicator-${index}`);
        const statusBadge = document.getElementById(`camera-status-badge-${index}`);

        const isRecording = cameraData.recording && cameraData.recording.active;
        if (recIndicator) {
          recIndicator.classList.toggle('d-none', !isRecording);
          recIndicator.classList.toggle('d-flex', isRecording);
          if (isRecording) anyRecording = true;
        }

        if (statusBadge) {
          if (isRecording) {
            statusBadge.textContent = 'Recording';
            statusBadge.className = BADGE_CLASSES.recording;
          } else if (cameraData.is_initialized) {
            statusBadge.textContent = 'Active';
            statusBadge.className = BADGE_CLASSES.active;
          } else {
            statusBadge.textContent = 'Disconnected';
            statusBadge.className = BADGE_CLASSES.inactive;
          }
        }
      });

      const statusDot = document.getElementById('system-status-dot');
      const statusText = document.getElementById('system-status-text');
      statusDot.className = `rounded-circle me-2 ${anyRecording ? 'bg-danger recording-dot' : 'bg-success'}`;
      statusText.textContent = anyRecording ? 'System Recording' : 'System Active';

      // Update the global status panel
      if (document.getElementById('page-dashboard').classList.contains('d-none') === false) {
        updateGlobalStatusPanel(allCamerasData);
      }
    })
    .catch(error => {
      clearTimeout(timeoutId); // Clear timeout on error too
      if (error.name === 'AbortError') {
        console.warn('Status fetch timed out.');
      } else {
        console.error('Error fetching status:', error);
      }
      // Show system as offline
      const statusDot = document.getElementById('system-status-dot');
      const statusText = document.getElementById('system-status-text');
      statusDot.className = 'rounded-circle me-2 bg-danger';
      statusText.textContent = 'System Offline';
    });
}


//
// --- GLOBAL STATUS PANEL FUNCTION ---
//
function updateGlobalStatusPanel(camerasData) {
  const bodyEl = document.getElementById('global-status-body');

  // If no data, show placeholder
  if (!camerasData || camerasData.length === 0) {
    bodyEl.innerHTML = `<div class="text-center p-4 text-muted">No camera data available...</div>`;
    return;
  }

  // Aggregate state across all cameras (OR logic)
  let globalPersonDetected = false;
  let globalTrafficLight = 'unknown';
  let anyInitialized = false;

  camerasData.forEach(cameraData => {
    if (cameraData.is_initialized) {
      anyInitialized = true;

      // Person detection: if ANY camera detects person
      if (cameraData.person_in_roi) {
        globalPersonDetected = true;
      }

      // Traffic light: priority is 'go' > 'stop' > 'unknown'
      if (cameraData.traffic_light === 'go') {
        globalTrafficLight = 'go';
      } else if (cameraData.traffic_light === 'stop' && globalTrafficLight !== 'go') {
        globalTrafficLight = 'stop';
      }
    }
  });

  if (!anyInitialized) {
    bodyEl.innerHTML = `<div class="text-center p-4 text-muted">Waiting for cameras to initialize...</div>`;
    return;
  }

  const trafficClass = globalTrafficLight === 'stop' ? 'danger' : (globalTrafficLight === 'go' ? 'success' : 'secondary');
  const trafficText = globalTrafficLight === 'unknown' ? '-' : globalTrafficLight.toUpperCase();

  const peopleClass = globalPersonDetected ? 'primary' : 'secondary';
  const peopleText = globalPersonDetected ? 'Detected' : 'Clear';

  // Display global status
  bodyEl.innerHTML = `
    <div class="row g-3">
      <div class="col-6">
        <div class="card text-center text-bg-${trafficClass} bg-opacity-75">
          <div class="card-header fs-sm text-uppercase fw-semibold" style="font-size: 0.8rem;">Traffic Light</div>
          <div class="card-body p-3">
            <h4 class="card-title fw-bold mb-0" style="font-size: 1.75rem;">${trafficText}</h4>
          </div>
        </div>
      </div>
      <div class="col-6">
        <div class="card text-center text-bg-${peopleClass} bg-opacity-75">
          <div class="card-header fs-sm text-uppercase fw-semibold" style="font-size: 0.8rem;">People</div>
          <div class="card-body p-3">
            <h4 class="card-title fw-bold mb-0" style="font-size: 1.75rem;">${peopleText}</h4>
          </div>
        </div>
      </div>
    </div>
  `;
}
// --- END OF GLOBAL STATUS PANEL FUNCTION ---
//


// --- ROI Drawing (New Modal Logic) ---

// FIX: New function to resize the overlay to match the 'object-fit: contain' video
function resizeRoiOverlay() {
  const videoStream = document.getElementById('drawZoneModal-video-stream');
  const overlay = document.getElementById('drawZoneModal-overlay');
  const container = document.getElementById('drawZoneModal-video-container');

  if (!videoStream || !overlay || !container || !videoStream.naturalWidth || videoStream.naturalWidth === 0) {
    // Video not loaded yet, reset overlay to container size
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.top = '0px';
    overlay.style.left = '0px';
    return;
  }

  const containerRect = container.getBoundingClientRect();
  const naturalRatio = videoStream.naturalWidth / videoStream.naturalHeight;
  const containerRatio = containerRect.width / containerRect.height;

  let renderedWidth, renderedHeight, topOffset, leftOffset;

  if (naturalRatio > containerRatio) {
    // Video is wider than container (letterbox top/bottom)
    renderedWidth = containerRect.width;
    renderedHeight = containerRect.width / naturalRatio;
    topOffset = (containerRect.height - renderedHeight) / 2;
    leftOffset = 0;
  } else {
    // Video is narrower than container (letterbox left/right)
    renderedHeight = containerRect.height;
    renderedWidth = containerRect.height * naturalRatio;
    topOffset = 0;
    leftOffset = (containerRect.width - renderedWidth) / 2;
  }

  // Apply the dimensions and position to the overlay
  overlay.style.width = `${renderedWidth}px`;
  overlay.style.height = `${renderedHeight}px`;
  overlay.style.top = `${topOffset}px`;
  overlay.style.left = `${leftOffset}px`;

  // Redraw any existing points
  drawRoiVisuals();
}


function initializeRoiModal() {
  const modalEl = document.getElementById('drawZoneModal');
  const videoStream = document.getElementById('drawZoneModal-video-stream');

  modalEl.addEventListener('show.bs.modal', () => {
    document.getElementById('drawZoneModalLabel').querySelector('span').textContent = activeCameraName;
    // Reset overlay size before loading
    const overlay = document.getElementById('drawZoneModal-overlay');
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.top = '0';
    overlay.style.left = '0';

    videoStream.src = `/video_feed/${activeCameraIndex}?t=${Date.now()}`;
    isDrawingRoi = false;
    currentRoiType = null;
    roiPoints = [];
    clearRoiVisualization();
    resetRoiButtons();
  });

  // FIX: Add listeners to resize overlay when video loads or window resizes
  videoStream.onload = () => {
    // Use a small timeout to ensure rendering is complete
    setTimeout(resizeRoiOverlay, 50);
  };
  window.addEventListener('resize', resizeRoiOverlay);

  // Also resize when the modal is *fully* shown
  modalEl.addEventListener('shown.bs.modal', () => {
    setTimeout(resizeRoiOverlay, 50); // Give it a moment to stabilize
  });

  modalEl.addEventListener('hidden.bs.modal', () => {
    videoStream.src = '';
    isDrawingRoi = false;
    currentRoiType = null;
    window.removeEventListener('resize', resizeRoiOverlay); // Clean up listener
  });

  // Modal drawing controls - 3 zone types: tl_red, tl_green, people
  document.getElementById('modal-draw-tl-red').addEventListener('click', () => startRoiDefinition('tl_red'));
  document.getElementById('modal-draw-tl-green').addEventListener('click', () => startRoiDefinition('tl_green'));
  document.getElementById('modal-draw-people').addEventListener('click', () => startRoiDefinition('people'));
  document.getElementById('modal-clear-last').addEventListener('click', handleRightClick);
  // document.getElementById('modal-save-roi').addEventListener('click', finishRoiDefinition); // Removed
  document.getElementById('modal-reset-all-roi').addEventListener('click', resetAllRois);

  videoStream.addEventListener('click', handleVideoClick);
}

function startRoiDefinition(type) {
  currentRoiType = type;
  roiPoints = [];
  isDrawingRoi = true;

  resetRoiButtons();

  // Get the correct button based on type
  let button;
  if (type === 'tl_red') {
    button = document.getElementById('modal-draw-tl-red');
  } else if (type === 'tl_green') {
    button = document.getElementById('modal-draw-tl-green');
  } else {
    button = document.getElementById('modal-draw-people');
  }

  button.innerHTML = '<i class="fas fa-mouse-pointer me-1"></i> Drawing... (4 points)';
  button.classList.remove('btn-outline-danger', 'btn-outline-success', 'btn-outline-primary');
  if (type === 'tl_red') {
    button.classList.add('btn-danger');
  } else if (type === 'tl_green') {
    button.classList.add('btn-success');
  } else {
    button.classList.add('btn-primary');
  }

  const typeNames = { 'tl_red': 'Red Light', 'tl_green': 'Green Light', 'people': 'People' };
  showNotification(`Drawing ${typeNames[type] || type} ROI: Click 4 points on the video.`, 'info');
}

function resetRoiButtons() {
  const tlRedBtn = document.getElementById('modal-draw-tl-red');
  const tlGreenBtn = document.getElementById('modal-draw-tl-green');
  const peopleBtn = document.getElementById('modal-draw-people');

  tlRedBtn.innerHTML = '<i class="fas fa-circle me-2" style="color: #dc3545;"></i>Draw Red Light Zone';
  tlGreenBtn.innerHTML = '<i class="fas fa-circle me-2" style="color: #198754;"></i>Draw Green Light Zone';
  peopleBtn.innerHTML = '<i class="fas fa-users me-2"></i>Draw People Zone';

  tlRedBtn.className = 'btn btn-outline-danger';
  tlGreenBtn.className = 'btn btn-outline-success';
  peopleBtn.className = 'btn btn-outline-primary';
}

function handleVideoClick(event) {
  if (!isDrawingRoi || !currentRoiType) return;

  // FIX: Use the overlay for coordinate calculation
  const overlay = document.getElementById('drawZoneModal-overlay');
  const overlayRect = overlay.getBoundingClientRect();

  // Calculate click position relative to the overlay
  const x = (event.clientX - overlayRect.left) / overlayRect.width;
  const y = (event.clientY - overlayRect.top) / overlayRect.height;

  // Check if click is inside the overlay (not in black bars)
  if (x < 0 || x > 1 || y < 0 || y > 1) return;

  roiPoints.push([x, y]);
  drawRoiVisuals();

  if (roiPoints.length === 4) {
    isDrawingRoi = false; // Stop drawing
    // Auto-save
    finishRoiDefinition();
  }
}

function handleRightClick() {
  if (roiPoints.length > 0) {
    roiPoints.pop();
    drawRoiVisuals();

    // Allow drawing again if user was "finished"
    if (!isDrawingRoi && currentRoiType) {
      isDrawingRoi = true;

      // Get the correct button based on type
      let button;
      if (currentRoiType === 'tl_red') {
        button = document.getElementById('modal-draw-tl-red');
      } else if (currentRoiType === 'tl_green') {
        button = document.getElementById('modal-draw-tl-green');
      } else {
        button = document.getElementById('modal-draw-people');
      }

      button.innerHTML = `<i class="fas fa-mouse-pointer me-1"></i> Drawing... (${4 - roiPoints.length} left)`;
      button.classList.remove('btn-outline-danger', 'btn-outline-success', 'btn-outline-primary');
      if (currentRoiType === 'tl_red') {
        button.classList.add('btn-danger');
      } else if (currentRoiType === 'tl_green') {
        button.classList.add('btn-success');
      } else {
        button.classList.add('btn-primary');
      }
    }
  }
}

//
// --- THIS IS THE FIXED drawRoiVisuals FUNCTION ---
//
function drawRoiVisuals() {
  clearRoiVisualization();
  const overlay = document.getElementById('drawZoneModal-overlay');
  if (!overlay || !currentRoiType) return;

  // FIX: Use overlay's clientWidth/Height, which are the rendered pixel dimensions
  const overlayWidth = overlay.clientWidth;
  const overlayHeight = overlay.clientHeight;
  const colorClass = currentRoiType === 'tl' ? 'bg-danger' : 'bg-success';

  roiPoints.forEach(point => {
    const pointEl = document.createElement('div');
    pointEl.className = `roi-point ${colorClass}`;
    // FIX: Use pixels relative to the overlay
    pointEl.style.left = `${point[0] * overlayWidth}px`;
    pointEl.style.top = `${point[1] * overlayHeight}px`;
    overlay.appendChild(pointEl);
  });

  if (roiPoints.length < 2) return;
  for (let i = 0; i < roiPoints.length - 1; i++) {
    drawLine(roiPoints[i], roiPoints[i + 1], colorClass);
  }
  // Show closing line if 4 points
  if (roiPoints.length === 4) {
    drawLine(roiPoints[3], roiPoints[0], colorClass);
  }
}
// --- END OF FIXED FUNCTION ---
//

//
// --- THIS IS THE FIXED drawLine FUNCTION ---
//
function drawLine(point1, point2, colorClass) {
  const overlay = document.getElementById('drawZoneModal-overlay');
  // FIX: Use overlay's clientWidth/Height
  const overlayWidth = overlay.clientWidth;
  const overlayHeight = overlay.clientHeight;

  // Calculate coordinates relative to the overlay
  const x1 = point1[0] * overlayWidth;
  const y1 = point1[1] * overlayHeight;
  const x2 = point2[0] * overlayWidth;
  const y2 = point2[1] * overlayHeight;

  const line = document.createElement('div');
  line.className = `roi-line ${colorClass}`;

  const length = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

  // Position line relative to the overlay
  line.style.left = `${x1}px`;
  line.style.top = `${y1}px`;
  line.style.width = `${length}px`;
  line.style.transform = `rotate(${angle}deg)`;
  overlay.appendChild(line);
}
// --- END OF FIXED FUNCTION ---
//

function clearRoiVisualization() {
  const overlay = document.getElementById('drawZoneModal-overlay');
  if (overlay) overlay.innerHTML = '';
}

function finishRoiDefinition() {
  const videoStream = document.getElementById('drawZoneModal-video-stream');

  if (!currentRoiType) {
    showNotification('Error: Please select "Draw Traffic Zone" or "Draw People Zone" first.', 'danger');
    return;
  }

  if (roiPoints.length !== 4) {
    showNotification('Error: You must click 4 points to define the zone.', 'danger');
    return;
  }
  if (!videoStream || videoStream.naturalWidth === 0) {
    showNotification('Error: Video stream not loaded. Please wait and try again.', 'danger');
    return;
  }

  // The percentages in roiPoints are now correct (relative to the visible image).
  // We send them as normalized coordinates (0-1) to the backend.
  // The backend will scale them to the current frame resolution.
  const actualPoints = roiPoints;

  fetch(`/set_roi/${activeCameraIndex}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type: currentRoiType, points: actualPoints })
  })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification('ROI saved successfully! You can now draw another zone.', 'success');
        // Keep modal open for continuous drawing
        // drawZoneModal.hide(); 

        // Reset state for next drawing
        roiPoints = [];
        clearRoiVisualization();
        resetRoiButtons();
        isDrawingRoi = false;
        currentRoiType = null;

        updateAllStatus(); // Refresh status panel
      } else {
        showNotification(data.message || 'Failed to save ROI.', 'danger');
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
}

function resetAllRois() {
  if (!confirm(`Are you sure you want to reset all ROIs for ${activeCameraName}?`)) return;

  fetch(`/reset_roi/${activeCameraIndex}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type: 'all' })
  })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification('All ROIs reset!', 'success');
        drawZoneModal.hide();
        updateAllStatus(); // Refresh status panel
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
}


// --- Settings Page ---

function loadSettings() {
  fetch(`/settings?t=${Date.now()}`)
    .then(response => response.json())
    .then(data => {
      globalSettings = data; // Cache settings

      // AI & Alerts
      document.getElementById('confidence-threshold').value = data.confidence_threshold;
      document.getElementById('confidence-display').textContent = data.confidence_threshold + '%';
      document.getElementById('audio-cooldown').value = data.audio_cooldown;
      document.getElementById('led-enabled').checked = data.led_enabled;
      document.getElementById('led-blink-interval').value = data.led_blink_interval;
      document.getElementById('led-blink-display').textContent = data.led_blink_interval + 's';

      // Recording
      // Convert seconds back to minutes for the form
      document.getElementById('recording-duration').value = data.recording_duration / 60;
      document.getElementById('retention-days').value = data.retention_days;
      document.getElementById('enable-recording').checked = data.enable_recording;
    })
    .catch(error => console.error('Error loading settings:', error));
}

function saveSettings(event) {
  event.preventDefault();
  const settings = {
    // AI & Alerts
    confidence_threshold: parseInt(document.getElementById('confidence-threshold').value),
    audio_cooldown: parseInt(document.getElementById('audio-cooldown').value),
    led_enabled: document.getElementById('led-enabled').checked,
    led_blink_interval: parseFloat(document.getElementById('led-blink-interval').value),

    // Recording
    // Convert minutes to seconds for the backend
    recording_duration: parseInt(document.getElementById('recording-duration').value) * 60,
    retention_days: parseInt(document.getElementById('retention-days').value),
    enable_recording: document.getElementById('enable-recording').checked,

    // Pass-through values from the original settings cache
    message_interval: globalSettings.message_interval || 5,
    message_duration: globalSettings.message_duration || 3,
  };

  fetch('/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings)
  })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification('Settings saved successfully!', 'success');
        loadSettings(); // Reload settings cache
      } else {
        showNotification('Error saving settings.', 'danger');
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
}

function resetSettings() {
  if (!confirm('Are you sure you want to reset all settings to default?')) return;

  fetch('/reload_settings', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification('Settings reset to default!', 'success');
        loadSettings(); // Reload settings into form
      } else {
        showNotification('Error resetting settings.', 'danger');
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
}

// --- Audio Upload ---

function handleAudioUpload(event, audioType) {
  const file = event.target.files[0];
  const inputEl = event.target;
  if (!file) return;

  if (file.size > 5 * 1024 * 1024) { // 5MB
    showNotification('File too large. Max 5MB.', 'danger');
    inputEl.value = '';
    return;
  }
  if (file.type !== 'audio/wav' && file.type !== 'audio/x-wav') {
    showNotification('Invalid file. Only .wav files are allowed.', 'danger');
    inputEl.value = '';
    return;
  }

  showNotification(`Uploading ${audioType} audio...`, 'info');
  const formData = new FormData();
  formData.append('audio_file', file);
  formData.append('audio_type', audioType);

  fetch('/upload_audio', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification(`${audioType} audio uploaded!`, 'success');
      } else {
        showNotification(data.error || 'Failed to upload.', 'danger');
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'))
    .finally(() => inputEl.value = ''); // Clear input
}

function resetAudio(audioType) {
  if (!confirm(`Are you sure you want to reset the ${audioType} audio to default?`)) return;

  fetch('/reset_audio', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_type: audioType })
  })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification(`${audioType} audio reset to default.`, 'success');
        // Clear file inputs
        if (audioType === 'berhenti') {
          document.getElementById('berhenti-upload').value = '';
        } else {
          document.getElementById('melintas-upload').value = '';
        }
      } else {
        showNotification(data.error || 'Failed to reset audio.', 'danger');
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
}


// --- Camera Configuration ---

function initializeCameraConfig() {
  const cameraSelect = document.getElementById('camera-select');
  const zoomLevel = document.getElementById('zoom-level');
  const zoomDisplay = document.getElementById('zoom-display');
  const applyZoomBtn = document.getElementById('apply-zoom');
  const cameraPreview = document.getElementById('camera-preview');
  const previewPlaceholder = document.getElementById('preview-placeholder');
  const zoomMarker = document.getElementById('zoom-marker');

  let currentX = 0.5;
  let currentY = 0.5;

  // Populate camera select when tab is shown
  const tabEl = document.getElementById('camera-config-tab');
  if (tabEl) {
    tabEl.addEventListener('shown.bs.tab', () => {
      // Clear existing options
      cameraSelect.innerHTML = '';

      if (allCamerasData.length === 0) {
        const option = document.createElement('option');
        option.text = "No cameras detected";
        cameraSelect.add(option);
        return;
      }

      allCamerasData.forEach((cam, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.text = cam.camera_name || `Camera ${index + 1}`;
        cameraSelect.add(option);
      });

      // Trigger change to load settings for first camera
      if (cameraSelect.options.length > 0) {
        cameraSelect.dispatchEvent(new Event('change'));
      }
    });

    // Stop preview when tab is hidden to save bandwidth
    tabEl.addEventListener('hidden.bs.tab', () => {
      if (cameraPreview) {
        cameraPreview.src = "";
        cameraPreview.style.display = 'none';
      }
      if (previewPlaceholder) previewPlaceholder.style.display = 'flex';
      if (zoomMarker) zoomMarker.style.display = 'none';
    });
  }

  // Handle camera selection change
  if (cameraSelect) {
    cameraSelect.addEventListener('change', () => {
      const camIndex = cameraSelect.value;
      if (camIndex === undefined || camIndex === null) return;

      // Load current zoom config
      const zoomConfigs = globalSettings.camera_zoom_configs || {};
      const config = zoomConfigs[camIndex] || {};

      // Handle old format or new format
      let level = 1.0;
      if (typeof config === 'number') {
        level = config;
        currentX = 0.5;
        currentY = 0.5;
      } else {
        level = config.level || 1.0;
        currentX = config.x !== undefined ? config.x : 0.5;
        currentY = config.y !== undefined ? config.y : 0.5;
      }

      zoomLevel.value = level;
      zoomDisplay.textContent = level + 'x';

      // Update preview
      if (cameraPreview) {
        cameraPreview.src = `/video_feed/${camIndex}?t=${Date.now()}`; // Add timestamp to prevent caching
        cameraPreview.style.display = 'block';
      }
      if (previewPlaceholder) previewPlaceholder.style.display = 'none';

      updateMarkerPosition();
    });
  }

  function updateMarkerPosition() {
    if (zoomMarker && cameraPreview && cameraPreview.style.display !== 'none') {
      zoomMarker.style.left = (currentX * 100) + '%';
      zoomMarker.style.top = (currentY * 100) + '%';
      zoomMarker.style.display = 'block';
    }
  }

  // --- Digital Zoom Controls ---

  // Handle click on preview to set center
  if (cameraPreview) {
    cameraPreview.addEventListener('click', (e) => {
      const rect = cameraPreview.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Normalize
      currentX = x / rect.width;
      currentY = y / rect.height;

      // Clamp
      currentX = Math.max(0, Math.min(currentX, 1));
      currentY = Math.max(0, Math.min(currentY, 1));

      updateMarkerPosition();
    });
  }

  // Handle zoom slider change
  if (zoomLevel) {
    zoomLevel.addEventListener('input', () => {
      zoomDisplay.textContent = zoomLevel.value + 'x';
    });
  }

  // Handle apply zoom
  if (applyZoomBtn) {
    applyZoomBtn.addEventListener('click', () => {
      const camIndex = cameraSelect.value;
      const zoom = parseFloat(zoomLevel.value);

      if (!camIndex && camIndex !== 0) return;

      fetch(`/set_zoom/${camIndex}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          zoom: zoom,
          x: currentX,
          y: currentY
        })
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification(`Zoom set to ${zoom}x at (${currentX.toFixed(2)}, ${currentY.toFixed(2)})`, 'success');
            // Update global settings cache
            if (!globalSettings.camera_zoom_configs) globalSettings.camera_zoom_configs = {};
            globalSettings.camera_zoom_configs[camIndex] = {
              level: zoom,
              x: currentX,
              y: currentY
            };
          } else {
            showNotification(data.message || 'Failed to set zoom', 'danger');
          }
        })
        .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
    });
  }

  // Reset zoom button
  const resetZoomBtn = document.getElementById('reset-zoom');
  if (resetZoomBtn) {
    resetZoomBtn.addEventListener('click', () => {
      const camIndex = cameraSelect.value;
      if (!camIndex && camIndex !== 0) return;

      // Reset to defaults
      currentX = 0.5;
      currentY = 0.5;
      zoomLevel.value = 1.0;
      zoomDisplay.textContent = '1.0x';

      // Hide zoom marker
      if (zoomMarker) zoomMarker.style.display = 'none';

      // Apply the reset
      fetch(`/set_zoom/${camIndex}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          zoom: 1.0,
          x: 0.5,
          y: 0.5
        })
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification('Zoom reset to default (1.0x)', 'success');
            // Update global settings cache
            if (!globalSettings.camera_zoom_configs) globalSettings.camera_zoom_configs = {};
            globalSettings.camera_zoom_configs[camIndex] = {
              level: 1.0,
              x: 0.5,
              y: 0.5
            };
          } else {
            showNotification(data.message || 'Failed to reset zoom', 'danger');
          }
        })
        .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
    });
  }
}

// --- Hardware Testing ---

function testSpeaker(audioType) {
  showNotification(`Testing ${audioType} speaker...`, 'info');

  fetch('/test_speaker', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_type: audioType })
  })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification(data.message, 'success');
      } else {
        showNotification(data.error || 'Failed to test speaker.', 'danger');
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
}

function testLED(action) {
  const actionText = action === 'on' ? 'Turning LED ON' : action === 'off' ? 'Turning LED OFF' : 'Blinking LED';
  showNotification(`${actionText}...`, 'info');

  fetch('/test_led', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: action })
  })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification(data.message, 'success');
      } else {
        showNotification(data.error || 'Failed to test LED.', 'danger');
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
}


// --- Playback Page ---

let allRecordingsData = []; // Cache for filtering

function loadRecordings() {
  const tableBody = document.getElementById('recordings-table-body');
  const loadingEl = document.getElementById('recordings-loading');
  const emptyEl = document.getElementById('recordings-empty');

  loadingEl.classList.remove('d-none');
  emptyEl.classList.add('d-none');
  tableBody.innerHTML = '';

  // Clear old tooltips
  clearTooltips();

  fetch(`/recordings?t=${Date.now()}`) // Use the original /recordings endpoint
    .then(response => response.json())
    .then(data => {
      loadingEl.classList.add('d-none');
      allRecordingsData = data.recordings || [];

      if (allRecordingsData.length === 0) {
        emptyEl.classList.remove('d-none');
      } else {
        populateDateFilter(allRecordingsData);
        filterRecordings(); // Display all recordings initially
      }
    })
    .catch(error => {
      loadingEl.classList.add('d-none');
      tableBody.innerHTML = `<tr><td colspan="5" class="text-center text-danger"><i class="fas fa-exclamation-triangle me-1"></i> Error loading recordings.</td></tr>`;
      console.error('Error loading recordings:', error);
    });
}

function populateDateFilter(recordings) {
  const dateFilter = document.getElementById('date-filter');
  const dates = new Set(recordings.map(rec => rec.date_folder));

  // Clear old options (except 'All Dates')
  while (dateFilter.options.length > 1) {
    dateFilter.remove(1);
  }

  // Sort dates descending
  const sortedDates = Array.from(dates).sort().reverse();

  sortedDates.forEach(date => {
    const option = document.createElement('option');
    option.value = date;
    option.textContent = date;
    dateFilter.appendChild(option);
  });
}

function filterRecordings() {
  const tableBody = document.getElementById('recordings-table-body');
  const emptyEl = document.getElementById('recordings-empty');
  tableBody.innerHTML = '';

  const cameraFilter = document.getElementById('cam-filter').value;
  const dateFilter = document.getElementById('date-filter').value;

  const filteredRecordings = allRecordingsData.filter(rec => {
    const camMatch = (cameraFilter === 'all' || rec.camera === cameraFilter);
    const dateMatch = (dateFilter === 'all' || rec.date_folder === dateFilter);
    return camMatch && dateMatch;
  });

  if (filteredRecordings.length === 0) {
    emptyEl.classList.remove('d-none');
    return;
  }

  emptyEl.classList.add('d-none');

  filteredRecordings.forEach(rec => {
    const row = document.createElement('tr');
    const isActive = rec.is_active;
    const buttonDisabled = isActive ? 'disabled' : '';
    const tooltip = isActive ? 'data-bs-toggle="tooltip" data-bs-title="Recording in progress..."' : '';
    const fileType = rec.filetype || 'MP4'; // Default to MP4

    row.innerHTML = `
          <td class="fw-medium">${rec.camera}</td>
          <td class="text-muted">${rec.created}</td>
          <td class="text-muted">${formatDuration(rec.duration)}</td>
          <td class="text-muted">${formatFileSize(rec.size)}</td>
          <td class="text-end">
            <button class="btn btn-sm btn-outline-success download-btn" ${buttonDisabled}>
              <i class="fas fa-download"></i>
            </button>
            <button class="btn btn-sm btn-outline-danger delete-btn ms-1" ${buttonDisabled}>
              <i class="fas fa-trash"></i>
            </button>
          </td>
        `;

    // Add event listeners
    row.querySelector('.download-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      downloadRecording(rec.filename);
    });

    row.querySelector('.delete-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteRecording(row, rec.filename);
    });

    tableBody.appendChild(row);
  });

  // Initialize new tooltips
  initializeTooltips();
}


function downloadRecording(filename) {
  const a = document.createElement('a');
  a.href = `/recordings/${filename}?t=${Date.now()}`;
  a.download = filename.split('/')[1] || filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function deleteRecording(rowElement, filename) {
  if (!confirm(`Are you sure you want to delete ${filename}?`)) return;

  fetch(`/recordings/${filename}`, { method: 'DELETE' })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        showNotification('Recording deleted!', 'success');
        rowElement.remove(); // Remove row from table
        // Re-check if table is now empty
        if (document.getElementById('recordings-table-body').childElementCount === 0) {
          document.getElementById('recordings-empty').classList.remove('d-none');
        }
      } else {
        showNotification(data.error || 'Failed to delete.', 'danger');
      }
    })
    .catch(error => showNotification(`Error: ${error.message}`, 'danger'));
}


// --- Utility Functions ---

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
  if (isNaN(seconds) || seconds < 0) return '00:00';
  const min = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

function clearTooltips() {
  activeTooltips.forEach(t => t.dispose());
  activeTooltips = [];
}

function initializeTooltips() {
  clearTooltips();
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  activeTooltips = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
}

function showNotification(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toastEl = document.createElement('div');
  toastEl.className = `toast align-items-center text-bg-${type} border-0`;
  toastEl.setAttribute('role', 'alert');
  toastEl.setAttribute('aria-live', 'assertive');
  toastEl.setAttribute('aria-atomic', 'true');

  let iconClass = 'fa-info-circle';
  if (type === 'success') iconClass = 'fa-check-circle';
  if (type === 'danger') iconClass = 'fa-exclamation-triangle';

  toastEl.innerHTML = `
    <div class="d-flex">
      <div class="toast-body d-flex align-items-center gap-2">
        <i class="fas ${iconClass}"></i>
        <span>${message}</span>
      </div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>
  `;

  container.appendChild(toastEl);

  const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
  toast.show();

  toastEl.addEventListener('hidden.bs.toast', () => {
    toastEl.remove();
  });
}

function initializeLowPowerMode() {
  const btn = document.getElementById('low-power-toggle');
  if (!btn) return;

  // Check initial state from global settings if available
  // We can check globalSettings if it's populated (it might not be yet, but updates will fix it)
  // Or we can rely on the user clicking it, but it's better to sync.
  // Since globalSettings is populated by startStatusUpdates -> updateGlobalStatusPanel -> but that's status, not settings.
  // We should probably fetch settings once.
  fetch('/settings')
    .then(res => res.json())
    .then(settings => {
      if (settings.processing_width === 320) {
        btn.classList.remove('btn-outline-secondary');
        btn.classList.add('btn-success');
      }
    })
    .catch(err => console.error("Error fetching settings for low power init:", err));

  btn.addEventListener('click', async () => {
    const isEnabled = btn.classList.contains('btn-success');
    const newState = !isEnabled; // Toggle

    try {
      const response = await fetch('/toggle_low_power', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enabled: newState }),
      });

      const data = await response.json();

      if (data.success) {
        if (newState) {
          btn.classList.remove('btn-outline-secondary');
          btn.classList.add('btn-success');
          showNotification('Low Power Mode ENABLED. Cameras restarting...', 'success');
        } else {
          btn.classList.remove('btn-success');
          btn.classList.add('btn-outline-secondary');
          showNotification('Low Power Mode DISABLED. Cameras restarting...', 'info');
        }

        // Refresh settings page if open
        if (document.getElementById('page-settings') && !document.getElementById('page-settings').classList.contains('d-none')) {
          loadSettings();
        }

      } else {
        showNotification('Failed to toggle Low Power Mode: ' + data.error, 'danger');
      }
    } catch (error) {
      console.error('Error toggling low power mode:', error);
      showNotification('Error toggling Low Power Mode', 'danger');
    }
  });
}