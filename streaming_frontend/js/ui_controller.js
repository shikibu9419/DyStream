/**
 * UI Controller — orchestrates WebRTC client, audio metering, and video display.
 */

class UIController {
    constructor() {
        // Components
        this.rtcClient = new WebRTCClient();
        this.audioCapture = new AudioCaptureManager();
        this.videoRenderer = new VideoRenderer('videoElement');

        // State
        this.currentMode = 'speaker';
        this.audioMode = 'mic';           // 'mic' or 'file'
        this.isStreaming = false;
        this.uploadedAudioBuffer = null;  // decoded AudioBuffer from file upload
        this._fileAudioCtx = null;        // AudioContext used during file playback
        this._fileSource = null;          // AudioBufferSourceNode for file playback

        // UI elements
        this.elements = {
            // Status
            statusText: document.getElementById('statusText'),
            connectionStatus: document.getElementById('connectionStatus'),

            // Mode buttons
            speakerModeBtn: document.getElementById('speakerModeBtn'),
            listenerModeBtn: document.getElementById('listenerModeBtn'),
            modeDescription: document.getElementById('modeDescription'),

            // Avatar selection
            avatarSelect: document.getElementById('avatarSelect'),

            // Control buttons
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),

            // Stats
            fpsValue: document.getElementById('fpsValue'),
            latencyValue: document.getElementById('latencyValue'),
            framesValue: document.getElementById('framesValue'),

            // Audio source tabs
            micTabBtn: document.getElementById('micTabBtn'),
            fileTabBtn: document.getElementById('fileTabBtn'),
            micTabPanel: document.getElementById('micTabPanel'),
            fileTabPanel: document.getElementById('fileTabPanel'),
            micStatus: document.getElementById('micStatus'),

            // Audio source (file upload)
            audioFileInput: document.getElementById('audioFileInput'),
            clearAudioBtn: document.getElementById('clearAudioBtn'),
            audioSourceStatus: document.getElementById('audioSourceStatus'),

            // Audio level
            audioLevelFill: document.getElementById('audioLevelFill'),

            // Settings
            denoisingSteps: document.getElementById('denoisingSteps'),
            denoisingValue: document.getElementById('denoisingValue'),
            cfgAudio: document.getElementById('cfgAudio'),
            cfgAudioValue: document.getElementById('cfgAudioValue'),
            cfgAudioOther: document.getElementById('cfgAudioOther'),
            cfgAudioOtherValue: document.getElementById('cfgAudioOtherValue'),
            cfgAll: document.getElementById('cfgAll'),
            cfgAllValue: document.getElementById('cfgAllValue'),
            applySettingsBtn: document.getElementById('applySettingsBtn'),
        };

        this._setupEventListeners();
        this._setupRTCCallbacks();
        this._startStatsUpdate();
    }

    // ── Event Listeners ──

    _setupEventListeners() {
        // Mode buttons
        this.elements.speakerModeBtn.addEventListener('click', () => this._switchMode('speaker'));
        this.elements.listenerModeBtn.addEventListener('click', () => this._switchMode('listener'));

        // Audio source tabs
        this.elements.micTabBtn.addEventListener('click', () => this._switchAudioTab('mic'));
        this.elements.fileTabBtn.addEventListener('click', () => this._switchAudioTab('file'));

        // Audio file upload
        this.elements.audioFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this._handleAudioUpload(file);
            }
        });
        this.elements.clearAudioBtn.addEventListener('click', () => this._clearAudioFile());

        // Control buttons
        this.elements.startBtn.addEventListener('click', () => this._startStreaming());
        this.elements.stopBtn.addEventListener('click', () => this._stopStreaming());

        // Settings sliders
        this.elements.denoisingSteps.addEventListener('input', (e) => {
            this.elements.denoisingValue.textContent = e.target.value;
        });
        this.elements.cfgAudio.addEventListener('input', (e) => {
            this.elements.cfgAudioValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        this.elements.cfgAudioOther.addEventListener('input', (e) => {
            this.elements.cfgAudioOtherValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        this.elements.cfgAll.addEventListener('input', (e) => {
            this.elements.cfgAllValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        this.elements.applySettingsBtn.addEventListener('click', () => this._applySettings());
    }

    // ── WebRTC Callbacks ──

    _setupRTCCallbacks() {
        this.rtcClient.onConnected((info) => {
            console.log('WebRTC connected, session:', info.session_id);
            this._updateStatus('Connected', 'connected');
            this.elements.startBtn.disabled = false;

            // Initialize audio level metering from the local stream
            const localStream = this.rtcClient.getLocalStream();
            if (localStream) {
                this.audioCapture.initFromStream(localStream);
                this.audioCapture.onAudioLevel((level) => {
                    this.elements.audioLevelFill.style.width = `${level * 100}%`;
                });
                this.audioCapture.start();
            }
        });

        this.rtcClient.onDisconnected(() => {
            console.log('WebRTC disconnected');
            this._updateStatus('Disconnected', 'disconnected');
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = true;

            if (this.isStreaming) {
                this._stopStreaming();
            }
        });

        this.rtcClient.onVideoTrack((stream) => {
            console.log('Received remote video stream');
            this.videoRenderer.setStream(stream);
            this.videoRenderer.start();
        });

        this.rtcClient.onStatus((msg) => {
            console.log('Status from server:', msg);
            this._handleStatusUpdate(msg);
        });

        this.rtcClient.onError((error) => {
            console.error('WebRTC error:', error);
            this._showError(error);
        });
    }

    // ── Audio file handling ──

    async _handleAudioUpload(file) {
        try {
            const audioCtx = new AudioContext();
            const arrayBuf = await file.arrayBuffer();
            this.uploadedAudioBuffer = await audioCtx.decodeAudioData(arrayBuf);
            await audioCtx.close();

            const duration = this.uploadedAudioBuffer.duration.toFixed(1);
            this.elements.audioSourceStatus.textContent =
                `${file.name} (${duration}s) — file audio will be streamed`;
            this.elements.audioSourceStatus.classList.add('file-loaded');
            this.elements.clearAudioBtn.style.display = '';
            console.log(`Audio file loaded: ${file.name}, ${duration}s, ` +
                `${this.uploadedAudioBuffer.numberOfChannels}ch, ` +
                `${this.uploadedAudioBuffer.sampleRate}Hz`);
        } catch (err) {
            console.error('Failed to decode audio file:', err);
            this._showError('Failed to decode audio file: ' + err.message);
            this._clearAudioFile();
        }
    }

    _clearAudioFile() {
        this.uploadedAudioBuffer = null;
        this.elements.audioFileInput.value = '';
        this.elements.audioSourceStatus.textContent = 'No file selected';
        this.elements.audioSourceStatus.classList.remove('file-loaded');
        this.elements.clearAudioBtn.style.display = 'none';
    }

    _switchAudioTab(tab) {
        if (tab === this.audioMode || this.isStreaming) return;

        this.audioMode = tab;

        if (tab === 'mic') {
            this.elements.micTabBtn.classList.add('active');
            this.elements.fileTabBtn.classList.remove('active');
            this.elements.micTabPanel.style.display = '';
            this.elements.fileTabPanel.style.display = 'none';
        } else {
            this.elements.micTabBtn.classList.remove('active');
            this.elements.fileTabBtn.classList.add('active');
            this.elements.micTabPanel.style.display = 'none';
            this.elements.fileTabPanel.style.display = '';
        }

        console.log('Switched audio tab to:', tab);
    }

    // ── Actions ──

    async _startStreaming() {
        if (this.isStreaming) return;

        // Validate file mode has a file selected
        if (this.audioMode === 'file' && !this.uploadedAudioBuffer) {
            this._showError('Please select an audio file first, or switch to Microphone mode.');
            return;
        }

        console.log('Starting streaming... audioMode:', this.audioMode);
        this._updateStatus('Connecting...', 'connecting');

        try {
            let audioStream = null;

            if (this.audioMode === 'file' && this.uploadedAudioBuffer) {
                // File audio → MediaStream via AudioContext
                this._fileAudioCtx = new AudioContext();
                this._fileSource = this._fileAudioCtx.createBufferSource();
                this._fileSource.buffer = this.uploadedAudioBuffer;
                const dest = this._fileAudioCtx.createMediaStreamDestination();
                this._fileSource.connect(dest);
                this._fileSource.start();
                audioStream = dest.stream;

                // Auto-stop when file playback ends
                this._fileSource.onended = () => {
                    console.log('Audio file playback ended');
                    if (this.isStreaming) {
                        this._stopStreaming();
                    }
                };
            }
            // audioMode === 'mic': audioStream stays null → getUserMedia in rtcClient.connect()

            // Connect WebRTC (uses audioStream if provided, else microphone)
            await this.rtcClient.connect(audioStream);

            // Tell server to start inference
            this.rtcClient.startStreaming();

            this.isStreaming = true;
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            // Disable tab switching and file input while streaming
            this.elements.micTabBtn.disabled = true;
            this.elements.fileTabBtn.disabled = true;
            this.elements.audioFileInput.disabled = true;
            this.elements.clearAudioBtn.disabled = true;
            this._updateStatus('Streaming', 'streaming');

            console.log('Streaming started');

        } catch (error) {
            console.error('Failed to start streaming:', error);
            this._showError('Failed to start streaming: ' + error.message);
            this._cleanupFileAudio();
        }
    }

    _stopStreaming() {
        if (!this.isStreaming) return;

        console.log('Stopping streaming...');

        this._cleanupFileAudio();
        this.rtcClient.stopStreaming();
        this.rtcClient.disconnect();
        this.videoRenderer.stop();
        this.audioCapture.cleanup();

        this.isStreaming = false;
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        // Re-enable tab switching and file input
        this.elements.micTabBtn.disabled = false;
        this.elements.fileTabBtn.disabled = false;
        this.elements.audioFileInput.disabled = false;
        this.elements.clearAudioBtn.disabled = false;
        this._updateStatus('Disconnected', 'disconnected');

        console.log('Streaming stopped');
    }

    _cleanupFileAudio() {
        if (this._fileSource) {
            try { this._fileSource.stop(); } catch (_) { /* already stopped */ }
            this._fileSource.onended = null;
            this._fileSource = null;
        }
        if (this._fileAudioCtx) {
            this._fileAudioCtx.close();
            this._fileAudioCtx = null;
        }
    }

    _switchMode(mode) {
        if (mode === this.currentMode) return;

        this.currentMode = mode;

        if (mode === 'speaker') {
            this.elements.speakerModeBtn.classList.add('active');
            this.elements.listenerModeBtn.classList.remove('active');
            this.elements.modeDescription.textContent =
                'AI behaves as speaker (your voice drives speaking motion)';
        } else {
            this.elements.speakerModeBtn.classList.remove('active');
            this.elements.listenerModeBtn.classList.add('active');
            this.elements.modeDescription.textContent =
                'AI behaves as listener (your voice elicits reactive listening motion)';
        }

        this.rtcClient.switchMode(mode);
        console.log('Switched to mode:', mode);
    }

    _applySettings() {
        const config = {
            denoising_steps: parseInt(this.elements.denoisingSteps.value),
            cfg_audio: parseFloat(this.elements.cfgAudio.value),
            cfg_audio_other: parseFloat(this.elements.cfgAudioOther.value),
            cfg_all: parseFloat(this.elements.cfgAll.value),
        };

        console.log('Applying settings:', config);
        this.rtcClient.updateConfig(config);

        this._updateStatus('Settings applied', 'success');
        setTimeout(() => {
            if (this.isStreaming) {
                this._updateStatus('Streaming', 'streaming');
            } else {
                this._updateStatus('Connected', 'connected');
            }
        }, 2000);
    }

    // ── Status handling ──

    _handleStatusUpdate(message) {
        const status = message.status;

        if (status === 'start_ok') {
            this._updateStatus('Streaming', 'streaming');
        } else if (status === 'stop_ok') {
            this._updateStatus('Connected', 'connected');
        } else if (status === 'mode_switch_ok') {
            console.log('Mode switch acknowledged');
        } else if (status === 'config_update_ok') {
            console.log('Config update acknowledged');
        }
    }

    _updateStatus(text, className = '') {
        this.elements.statusText.textContent = text;
        this.elements.connectionStatus.textContent = text;
        this.elements.connectionStatus.className = className;
    }

    _showError(message) {
        alert('Error: ' + message);
    }

    // ── Stats update loop ──

    _startStatsUpdate() {
        setInterval(async () => {
            if (!this.isStreaming) return;

            // Video renderer stats (FPS via requestVideoFrameCallback)
            const renderStats = this.videoRenderer.getStats();
            this.elements.fpsValue.textContent = renderStats.fps;
            this.elements.framesValue.textContent = renderStats.framesRendered;

            // WebRTC stats (jitter as proxy for latency)
            const rtcStats = await this.rtcClient.getStats();
            if (rtcStats) {
                const jitterMs = Math.round((rtcStats.jitter || 0) * 1000);
                this.elements.latencyValue.textContent = jitterMs + 'ms';
            }
        }, 1000);
    }
}
