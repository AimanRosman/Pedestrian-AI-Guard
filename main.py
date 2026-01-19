import time
import cv2
import numpy as np
import os
import sys
import torch
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import mimetypes
import threading
import logging
from datetime import datetime
import glob
import gc
import queue
import re
import subprocess
import shutil

# --- Hardware & OS Abstraction ---
# Try to import hardware-specific libraries, but don't fail if they're missing
try:
    import Jetson.GPIO as GPIO
    JETSON_GPIO_AVAILABLE = True
except ImportError:
    JETSON_GPIO_AVAILABLE = False

try:
    import RPi.GPIO as RPiGPIO
    RPI_GPIO_AVAILABLE = True
except ImportError:
    RPI_GPIO_AVAILABLE = False

try:
    import gpiod
    from gpiod.line import Direction, Value
    GPIOD_AVAILABLE = True
except ImportError:
    GPIOD_AVAILABLE = False
    
# Import for new audio player
try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except ImportError:
    SIMPLEAUDIO_AVAILABLE = False

# Import for DHT11 sensor
try:
    import adafruit_dht
    import board
    DHT11_AVAILABLE = True
except ImportError:
    DHT11_AVAILABLE = False

# Import for ADS1115 ADC
try:
    import board
    import busio
    from adafruit_ads1x15.ads1115 import ADS1115
    from adafruit_ads1x15.analog_in import AnalogIn
    ADS1115_AVAILABLE = True
except ImportError:
    ADS1115_AVAILABLE = False
    
# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not JETSON_GPIO_AVAILABLE:
    logger.warning("Jetson.GPIO library not found.")
if not RPI_GPIO_AVAILABLE:
    logger.warning("RPi.GPIO library not found.")
if not GPIOD_AVAILABLE:
    logger.warning("gpiod library not found.")
if not SIMPLEAUDIO_AVAILABLE:
    logger.warning("simpleaudio library not found. Audio playback will be disabled.")
if not DHT11_AVAILABLE:
    logger.warning("Adafruit_DHT library not found. DHT11 sensor will be disabled.")
if not ADS1115_AVAILABLE:
    logger.warning("ADS1115 library not found. Voltage sensor will be disabled.")


class LEDController:
    """Abstract base class for LED controllers."""
    def __init__(self, pin):
        self.pin = pin
        self.blinking = False
        self.blink_thread = None
        self.stop_event = threading.Event()
        logger.warning(f"Initializing LEDController (Pin: {self.pin})")

    def turn_on(self):
        raise NotImplementedError

    def turn_off(self):
        raise NotImplementedError

    def start_blink(self, interval):
        self.stop_blink() # Stop any existing blink
        self.blinking = True
        self.stop_event.clear()
        self.blink_thread = threading.Thread(target=self._blink_loop, args=(interval,), daemon=True)
        self.blink_thread.start()

    def _blink_loop(self, interval):
        while not self.stop_event.is_set():
            try:
                self.turn_on()
                if self.stop_event.wait(interval):
                    break
                self.turn_off()
                if self.stop_event.wait(interval):
                    break
            except Exception as e:
                logger.error(f"Error in blink loop: {e}")
                break
        self.turn_off() # Ensure LED is off when stopping

    def stop_blink(self):
        self.blinking = False
        if self.blink_thread and self.blink_thread.is_alive():
            self.stop_event.set()
            self.blink_thread.join(timeout=1.0)
        self.blink_thread = None
        self.turn_off() # Ensure it's off

    def cleanup(self):
        self.stop_blink()
        logger.warning("LED Controller cleanup complete.")

class JetsonLEDController(LEDController):
    """LED controller specifically for Jetson.GPIO library."""
    def __init__(self, pin):
        if not JETSON_GPIO_AVAILABLE:
            raise ImportError("Cannot create JetsonLEDController, Jetson.GPIO library not found.")
        super().__init__(pin)
        # Use BOARD mode to match original script
        GPIO.setmode(GPIO.BOARD) 
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.HIGH) # HIGH is off for this logic
        logger.warning(f"JetsonLEDController initialized on BOARD pin {self.pin}")

    def turn_on(self):
        GPIO.output(self.pin, GPIO.LOW) # LOW is ON

    def turn_off(self):
        GPIO.output(self.pin, GPIO.HIGH) # HIGH is OFF

    def cleanup(self):
        super().cleanup()
        GPIO.cleanup(self.pin)
        logger.warning(f"JetsonLEDController cleaned up pin {self.pin}")

class RaspberryPiLEDController(LEDController):
    """LED controller specifically for RPi.GPIO library."""
    def __init__(self, pin):
        if not RPI_GPIO_AVAILABLE:
            raise ImportError("Cannot create RaspberryPiLEDController, RPi.GPIO library not found.")
        super().__init__(pin)
        # Use BOARD mode
        RPiGPIO.setmode(RPiGPIO.BOARD) 
        RPiGPIO.setup(self.pin, RPiGPIO.OUT, initial=RPiGPIO.HIGH) # HIGH is off
        logger.warning(f"RaspberryPiLEDController initialized on BOARD pin {self.pin}")

    def turn_on(self):
        RPiGPIO.output(self.pin, RPiGPIO.LOW) # LOW is ON

    def turn_off(self):
        RPiGPIO.output(self.pin, RPiGPIO.HIGH) # HIGH is OFF

    def cleanup(self):
        super().cleanup()
        RPiGPIO.cleanup(self.pin)
        logger.warning(f"RaspberryPiLEDController cleaned up pin {self.pin}")

class GpiodLEDController(LEDController):
    """LED controller for Raspberry Pi 5 using gpiod library (v2.x API)."""
    def __init__(self, pin):
        if not GPIOD_AVAILABLE:
            raise ImportError("Cannot create GpiodLEDController, gpiod library not found.")
        super().__init__(pin)
        
        # For Raspberry Pi 5, use gpiochip4
        self.chip_path = "/dev/gpiochip4"
        
        # gpiod 2.x API
        self.line_request = gpiod.request_lines(
            self.chip_path,
            consumer="spas-gpio",
            config={
                self.pin: gpiod.LineSettings(
                    direction=Direction.OUTPUT,
                    output_value=Value.ACTIVE  # Start HIGH (off for active-low relay)
                )
            }
        )
        
        logger.warning(f"GpiodLEDController initialized on {self.chip_path} pin {self.pin}")

    def turn_on(self):
        # Set to LOW (INACTIVE) for active-low relay
        self.line_request.set_value(self.pin, Value.INACTIVE)

    def turn_off(self):
        # Set to HIGH (ACTIVE) for active-low relay
        self.line_request.set_value(self.pin, Value.ACTIVE)

    def cleanup(self):
        super().cleanup()
        if hasattr(self, 'line_request') and self.line_request:
            self.line_request.release()
        logger.warning(f"GpiodLEDController cleaned up pin {self.pin}")

class DummyLEDController(LEDController):
    """Dummy LED controller for when no GPIO is available (e.g., Windows)."""
    def __init__(self, pin):
        # We don't call super().__init__ because it logs/creates threading events which is fine
        # but we want to be minimal. Actually, super init is fine.
        self.pin = pin
        self.blinking = False
        self.blink_thread = None
        self.stop_event = threading.Event()
        logger.warning(f"DummyLEDController initialized on pin {self.pin} (No real hardware)")

    def turn_on(self):
        # logger.info(f"Dummy LED {self.pin} ON") # Reduce spam
        pass

    def turn_off(self):
        # logger.info(f"Dummy LED {self.pin} OFF")
        pass

    def cleanup(self):
        self.stop_blink()
        logger.warning(f"DummyLEDController cleaned up pin {self.pin}")

#
# ----------------------------------------------------
# REFACTORED AUDIO PLAYER (using simpleaudio)
# ----------------------------------------------------
#
class AudioPlayer:
    """Cross-platform audio player using simpleaudio (for .wav files only)."""
    
    def __init__(self, speaker_relay_controller=None):
        self.audio_lock = threading.Lock()
        self.speaker_relay_controller = speaker_relay_controller
        self.current_process = None  # Track current audio process for cutoff
        if not SIMPLEAUDIO_AVAILABLE:
            logger.error("AudioPlayer disabled: simpleaudio library not found.")

    def play_audio(self, audio_path):
        if not SIMPLEAUDIO_AVAILABLE:
            logger.warning(f"Skipping audio (simpleaudio disabled): {audio_path}")
            return
            
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return
            
        if not audio_path.lower().endswith('.wav'):
            logger.error(f"AudioPlayer Error: Only .wav files are supported. Cannot play {audio_path}")
            return
            
        # Try to acquire lock with a short timeout
        # This allows waiting briefly for current audio to finish
        if not self.audio_lock.acquire(blocking=True, timeout=0.1):
            logger.debug(f"Audio playback skipped, already playing.")
            return

        # Start playback in a new thread to avoid blocking the main thread
        threading.Thread(target=self._play_in_thread, args=(audio_path,), daemon=True).start()

    def stop_audio(self):
        """Stop currently playing audio immediately"""
        if self.current_process is not None:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=1)
                logger.warning("Audio playback stopped/cutoff")
            except Exception as e:
                logger.error(f"Error stopping audio: {e}")
            finally:
                self.current_process = None
                # Release lock if held
                try:
                    self.audio_lock.release()
                except RuntimeError:
                    pass  # Lock wasn't held
                # Turn off speaker relay
                if self.speaker_relay_controller:
                    self.speaker_relay_controller.turn_off()

    def _get_usb_audio_device(self):
        """Find the ALSA card index for USB Audio Device"""
        try:
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                # Match various USB audio device names
                if ('USB Audio' in line or 'USB audio' in line or 'usb-audio' in line.lower()) and 'card' in line:
                    # Extract card number: "card 2: Device ..." -> "2"
                    parts = line.split(':')
                    if len(parts) > 0:
                        card_part = parts[0] # "card 2"
                        card_num = card_part.split()[1]
                        logger.warning(f"Found USB audio on card {card_num}")
                        return f"plughw:{card_num},0"
        except Exception as e:
            logger.error(f"Error finding USB audio device: {e}")
        return None

    def _play_in_thread(self, audio_path):
        try:
            # Turn on speaker relay before playing
            if self.speaker_relay_controller:
                self.speaker_relay_controller.turn_on()
                logger.warning("Speaker relay activated")
            
            logger.warning(f"Playing audio: {audio_path}")
            
            # Find USB audio device dynamically
            device = self._get_usb_audio_device()
            
            cmd = ['aplay', '-q', audio_path]
            if device:
                logger.warning(f"Using audio device: {device}")
                cmd = ['aplay', '-D', device, '-q', audio_path]
            else:
                logger.warning("USB Audio Device not found, using default")
            
            # Use Popen to allow process termination
            self.current_process = subprocess.Popen(cmd)
            self.current_process.wait()  # Wait for playback to complete
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
        finally:
            self.current_process = None
            # Turn off speaker relay after playing
            if self.speaker_relay_controller:
                self.speaker_relay_controller.turn_off()
                logger.warning("Speaker relay deactivated")
            
            self.audio_lock.release()
            logger.warning(f"Audio playback finished, lock released.")

# --- End Hardware & OS Abstraction ---


class DHT11Sensor:
    """DHT11 Temperature and Humidity Sensor using adafruit_dht"""
    
    def __init__(self, gpio_pin=4):
        self.gpio_pin = gpio_pin
        self.dht_device = None
        
        if DHT11_AVAILABLE:
            try:
                # Map integer pin to board pin object (e.g., 4 -> board.D4)
                pin_name = f"D{gpio_pin}"
                if hasattr(board, pin_name):
                    board_pin = getattr(board, pin_name)
                    self.dht_device = adafruit_dht.DHT11(board_pin)
                    logger.warning(f"DHT11 sensor initialized on {pin_name}")
                else:
                    logger.error(f"Invalid board pin: {pin_name}")
            except Exception as e:
                logger.error(f"Failed to initialize DHT11: {e}")
        else:
            logger.warning("DHT11 sensor disabled: adafruit_dht library not available")
    
    def read(self):
        """Read temperature and humidity from DHT11 sensor"""
        if not self.dht_device:
            return None, None
        
        # Try up to 3 times to get a reading
        for _ in range(3):
            try:
                # adafruit_dht raises RuntimeError for read errors (common with DHT11)
                temperature = self.dht_device.temperature
                humidity = self.dht_device.humidity
                
                if temperature is not None and humidity is not None:
                    return round(temperature, 1), round(humidity, 1)
                
                # If we got None, wait a bit and try again
                time.sleep(2.0)
                
            except RuntimeError as e:
                # Expected error, just return None to skip this reading
                time.sleep(2.0)
                continue
            except Exception as e:
                logger.error(f"Error reading DHT11 sensor: {e}")
                return None, None
        
        return None, None


class VoltageSensor:
    """ADS1115 ADC for Battery Voltage Monitoring"""
    
    def __init__(self, channel=0, voltage_divider_ratio=5.05):
        """
        Initialize voltage sensor
        
        Args:
            channel: ADC channel (0-3)
            voltage_divider_ratio: Ratio of voltage divider (default ~5.05 from battery.py)
        """
        self.channel = channel
        self.voltage_divider_ratio = voltage_divider_ratio
        self.ads = None
        self.analog_in = None
        
        if not ADS1115_AVAILABLE:
            logger.warning("Voltage sensor disabled: ADS1115 library not available")
            return
        
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS1115(i2c)
            self.ads.gain = 1 # Set gain to 1 (+/- 4.096V) as per battery.py
            
            # Select channel
            if channel == 0:
                self.analog_in = AnalogIn(self.ads, 0)
            elif channel == 1:
                self.analog_in = AnalogIn(self.ads, 1)
            elif channel == 2:
                self.analog_in = AnalogIn(self.ads, 2)
            elif channel == 3:
                self.analog_in = AnalogIn(self.ads, 3)
            
            logger.warning(f"ADS1115 voltage sensor initialized on channel {channel} with gain 1")
        except Exception as e:
            logger.error(f"Failed to initialize ADS1115: {e}")
            self.ads = None
    
    def read(self):
        """Read voltage from ADC and convert to actual battery voltage"""
        if not ADS1115_AVAILABLE or self.ads is None or self.analog_in is None:
            return None
        
        try:
            # Read voltage from ADC
            adc_voltage = self.analog_in.voltage
            
            # Convert to actual battery voltage using divider ratio
            battery_voltage = adc_voltage * self.voltage_divider_ratio
            
            return round(battery_voltage, 2)
        except Exception as e:
            logger.error(f"Error reading voltage sensor: {e}")
            return None


class Config:
    """Application configuration constants for general deployment"""
    
    # Set the model path to the 'models' directory
    YOLO_MODEL_PATH = 'models/yolo26n.pt' 
    
    RECORDINGS_DIR = '/media/pi/rootfs/recordings'
    AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'audio')
    
    # General deployment settings
    DEFAULT_SETTINGS = {
        'confidence_threshold': 50,
        'message_interval': 5,
        'message_duration': 3,
        'show_bboxes': True,
        'show_dots': True,
        'tl_roi_label': 'Traffic Light ROI',
        'people_roi_label': 'People ROI',
        'show_timestamp': True,
        'show_camera_name': True,
        'debug_show_all_people': False,
        'force_cpu': False,
        'max_cameras': 3,
        'camera_test_timeout': 2.0,
        'camera_retry_count': 3,
        'camera_indices': [0],
        # 'camera_device_paths': [
        #     '/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4.1:1.0-video-index0',
        #     '/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4.3:1.0-video-index0',
        #     '/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4.4:1.0-video-index0'
        # ],
        'processing_width': 640,
        'processing_height': 480,
        'target_fps': 15,
        'skip_frames': 1,
        'recording_duration': 300,
        'retention_days': 30,
        'preferred_codec': 'H264',
        'preferred_container': 'mp4',
        'validate_recordings': True,
        'enable_recording': True,
        'berhenti_audio_filename': 'berhenti_default.wav',
        'melintas_audio_filename': 'melintas_default.wav',
        'audio_cooldown': 3,
        'led_enabled': True,
        'led_controller_type': 'auto',
        'led_pin': 17,  # GPIO 17 for LED
        'led_blink_interval': 0.5,
        'speaker_relay_enabled': True,
        'speaker_relay_pin': 27,  # GPIO 27 for speaker relay
        'camera_indices': [0, 2, 4],  # Actual USB camera indices on this system
        'camera_device_paths': [],  # List of device paths e.g. ["/dev/video0", "/dev/video2"]
        'max_cameras': 3,
        'camera_retry_count': 5,  # Increased from 2
        'camera_test_timeout': 5.0,  # Increased from 2.0
        'camera_init_delay': 3.0,  # Increased from 1.0
        'camera_zoom_configs': {},  # Dict: camera_index -> {'level': float, 'x': float, 'y': float}
        'camera_links': {
            '0': ['2', '4'],  # Camera 0 linked with cameras 2 and 4
            '2': ['0', '4'],  # Camera 2 linked with cameras 0 and 4
            '4': ['0', '2']   # Camera 4 linked with cameras 0 and 2
        },  # Link all USB cameras together
    }
    
    ROI_COLORS = {
        'tl': (0, 255, 255),      # Yellow (legacy)
        'tl_red': (0, 0, 255),    # Red
        'tl_green': (0, 255, 0),  # Green
        'people': (255, 255, 0)   # Cyan
    }


class CameraDetector:
    """Detects available cameras and their indices or device paths (Simplified for general use)"""

    @staticmethod
    def detect_available_cameras(max_cameras=10, test_timeout=2.0, retry_count=2, explicit_indices=None, device_paths=None):
        """Detect available cameras using cv2.CAP_ANY

        Args:
            max_cameras: Maximum camera indices to test (if using indices)
            test_timeout: Timeout for testing each camera
            retry_count: Number of retry attempts per camera
            explicit_indices: List of camera indices to test (e.g., [0, 1, 2])
            device_paths: List of device paths to test (e.g., ['/dev/v4l/by-path/...'])

        Returns:
            List of available cameras (either indices or device paths)
        """
        available_cameras = []
        logger.warning("Detecting available cameras...")

        # Prioritize device paths if provided (Linux V4L paths)
        if device_paths:
            logger.warning(f"Testing {len(device_paths)} device paths...")
            for device_path in device_paths:
                logger.warning(f"Testing device path {device_path}...")

                for attempt in range(retry_count):
                    cap = None
                    try:
                        # For V4L paths on Linux, use CAP_V4L2 backend
                        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

                        if cap.isOpened():
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                available_cameras.append(device_path)
                                logger.warning(f"Camera at {device_path} is available")
                                cap.release()
                                break  # Success, move to next device

                            cap.release()

                        logger.warning(f"Camera at {device_path} not responsive (attempt {attempt+1})")

                    except Exception as e:
                        logger.error(f"Error testing camera {device_path} (attempt {attempt+1}): {e}")
                        if cap is not None:
                            cap.release()

                    time.sleep(0.5)
        if device_paths:
            # ... (keep existing device path logic) ...
            pass # Simplified for this diff, assume existing logic is fine or irrelevant for Windows
        else:
            # Fall back to index-based detection
            # REDUCED from 10 to 2 to avoid "weird" index out of range errors on typical laptops
            indices_to_test = explicit_indices if explicit_indices else range(2) 

            for camera_index in indices_to_test:
                logger.warning(f"Testing camera index {camera_index}...")

                for attempt in range(retry_count):
                    cap = None
                    try:
                        # Use cv2.CAP_DSHOW on Windows to avoid some delays/errors, or CAP_ANY
                        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
                        cap = cv2.VideoCapture(camera_index, backend)

                        if cap.isOpened():
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                available_cameras.append(camera_index)
                                logger.warning(f"Camera {camera_index} is available")
                                cap.release()
                                break # Success, move to next index

                            cap.release()

                        logger.warning(f"Camera {camera_index} not responsive (attempt {attempt+1})")

                    except Exception as e:
                        logger.error(f"Error testing camera {camera_index} (attempt {attempt+1}): {e}")
                        if cap is not None:
                            cap.release()

                    time.sleep(0.5)

        logger.warning(f"Found {len(available_cameras)} available cameras: {available_cameras}")
        return available_cameras


class ModelManager:
    """Handles model loading and inference using YOLOv8 (Generalized)"""
    
    def __init__(self):
        self.device = None
        self.yolo_model = None
        self.use_gpu = True
        self.frame_count = 0
        self._setup_device()
        self._load_model()
    
    def _setup_device(self):
        """Setup the device for inference - Forced to CPU for Raspberry Pi"""
        # User requested to remove CUDA check for potential performance/compatibility on Pi
        self.device = torch.device('cpu')
        self.use_gpu = False
        logger.warning(f"Device set to CPU (Raspberry Pi Mode)")
    
    def _load_model(self):
        """
        Load the YOLO model.
        The ultralytics library will automatically download the model 
        if Config.YOLO_MODEL_PATH is not found.
        """
        try:
            from ultralytics import YOLO
            
            # --- ADDED THIS LINE ---
            # Ensure the 'models/' directory exists before YOLO tries to save there
            os.makedirs(os.path.dirname(Config.YOLO_MODEL_PATH), exist_ok=True)
            # ---
            
            # This single line handles checking for, downloading, and loading the model.
            logger.warning(f"Loading model: {Config.YOLO_MODEL_PATH}. (Will download if missing)")
            self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
            
            self._test_model()
            
            logger.warning(f"YOLO model loaded successfully on {self.device.type}")
            
        except Exception as e:
            logger.error(f"ERROR loading YOLO model: {e}")
            if self.use_gpu:
                logger.warning("Attempting to fall back to CPU...")
                self.device = torch.device('cpu')
                self.use_gpu = False
                try:
                    # Retry loading on CPU
                    self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
                    self._test_model()
                    logger.warning("YOLO model loaded successfully on CPU after fallback")
                except Exception as e2:
                    logger.error(f"ERROR loading YOLO model on CPU: {e2}")
                    sys.exit(1)
            else:
                sys.exit(1)
    
    def _test_model(self):
        """Test the model with a dummy inference"""
        try:
            dummy_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            self.yolo_model(dummy_image, verbose=False, device=self.device)
            logger.warning("Model test successful")
            return True
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            raise e
    
    def detect_objects(self, frame, confidence_threshold=0.5):
        """Run object detection on the frame"""
        try:
            self.frame_count += 1
            if self.frame_count % Config.DEFAULT_SETTINGS.get('skip_frames', 1) != 0:
                return []
            
            device = 'cpu' if Config.DEFAULT_SETTINGS.get('force_cpu', False) else self.device
            
            processing_width = Config.DEFAULT_SETTINGS.get('processing_width', 640)
            processing_height = Config.DEFAULT_SETTINGS.get('processing_height', 480)
            
            if frame.shape[1] != processing_width or frame.shape[0] != processing_height:
                resized_frame = cv2.resize(frame, (processing_width, processing_height))
            else:
                resized_frame = frame
            
            results = self.yolo_model(resized_frame, verbose=False, device=device)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        orig_h, orig_w = frame.shape[:2]
                        x1 = int(x1 * orig_w / processing_width)
                        y1 = int(y1 * orig_h / processing_height)
                        x2 = int(x2 * orig_w / processing_width)
                        y2 = int(y2 * orig_h / processing_height)
                        
                        if conf > confidence_threshold:
                            detections.append([x1, y1, x2, y2, conf, cls])
            
            if self.frame_count % 30 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return detections
        
        except Exception as e:
            logger.error(f"People detection error: {e}")
            return []

class TrafficLightDetector:
    """Handles traffic light state detection (Unchanged)"""
    
    @staticmethod
    def detect_state(roi):
        """Detect traffic light state using color analysis (red and green only)
        
        OUTDOOR OPTIMIZED: Uses multiple HSV ranges to handle various lighting conditions:
        - Bright sunlight (lower saturation)
        - Cloudy/overcast (normal saturation)  
        - Dusk/dawn (lower value)
        - Multiple shades of red and green from different traffic light manufacturers
        
        Only returns 'stop' if red is specifically detected above a threshold.
        Returns 'go' if green is detected above threshold.
        Returns 'unknown' if neither color meets the minimum threshold.
        """
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # ===== RED COLOR RANGES (expanded for better detection) =====
            # Standard bright red (hue 0-10)
            lower_red1 = np.array([0, 40, 40])       # Very low S/V for washed out colors
            upper_red1 = np.array([10, 255, 255])
            
            # Wrap-around red (hue 160-180) - captures pink-reds
            lower_red2 = np.array([160, 40, 40])
            upper_red2 = np.array([180, 255, 255])
            
            # Orange-red (hue 10-25) - some traffic lights appear orange
            lower_red3 = np.array([10, 40, 40])
            upper_red3 = np.array([25, 255, 255])
            
            # Dark red / maroon (very low value for dim conditions)
            lower_red4 = np.array([0, 20, 20])
            upper_red4 = np.array([15, 255, 180])
            
            # Wrap-around dark red (hue 170-180)
            lower_red5 = np.array([170, 20, 20])
            upper_red5 = np.array([180, 255, 180])
            
            # Very bright/saturated red (high value, for LED lights)
            lower_red6 = np.array([0, 100, 150])
            upper_red6 = np.array([12, 255, 255])
            
            # ===== GREEN COLOR RANGES (STRICT - pedestrian light specific) =====
            # Pedestrian walk lights are typically bright, saturated green
            # Using higher saturation thresholds to avoid false positives
            
            # Standard bright green (hue 45-75, high saturation)
            lower_green1 = np.array([45, 80, 80])     # Higher S/V for accurate detection
            upper_green1 = np.array([75, 255, 255])
            
            # Bright LED green (hue 50-70, very saturated)
            lower_green2 = np.array([50, 100, 100])   # Very saturated green
            upper_green2 = np.array([70, 255, 255])
            
            # Cyan-green LED (some pedestrian lights have cyan tint)
            lower_green3 = np.array([75, 80, 80])
            upper_green3 = np.array([90, 255, 255])
            
            # Create masks for all red ranges
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red3 = cv2.inRange(hsv, lower_red3, upper_red3)
            mask_red4 = cv2.inRange(hsv, lower_red4, upper_red4)
            mask_red5 = cv2.inRange(hsv, lower_red5, upper_red5)
            mask_red6 = cv2.inRange(hsv, lower_red6, upper_red6)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_red = cv2.bitwise_or(mask_red, mask_red3)
            mask_red = cv2.bitwise_or(mask_red, mask_red4)
            mask_red = cv2.bitwise_or(mask_red, mask_red5)
            mask_red = cv2.bitwise_or(mask_red, mask_red6)
            
            # Create masks for green ranges (only 3 strict ranges now)
            mask_green1 = cv2.inRange(hsv, lower_green1, upper_green1)
            mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)
            mask_green3 = cv2.inRange(hsv, lower_green3, upper_green3)
            mask_green = cv2.bitwise_or(mask_green1, mask_green2)
            mask_green = cv2.bitwise_or(mask_green, mask_green3)
            
            red_count = cv2.countNonZero(mask_red)
            green_count = cv2.countNonZero(mask_green)
            
            # Calculate total ROI pixels for percentage threshold
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # Minimum threshold: at least 1% of ROI must be the detected color
            # (increased from 0.2% to reduce false positives)
            min_threshold = total_pixels * 0.01
            
            # Only detect if color meets minimum threshold
            red_detected = red_count >= min_threshold
            green_detected = green_count >= min_threshold
            
            # Decision logic: prioritize the color with more pixels if both detected
            if green_detected and (green_count > red_count):
                return 'go', green_count
            elif red_detected and (red_count > green_count):
                return 'stop', red_count
            elif green_detected:
                return 'go', green_count
            elif red_detected:
                return 'stop', red_count
            else:
                # Neither color meets threshold - return unknown
                return 'unknown', 0
                
        except Exception as e:
            logger.error(f"Traffic light detection error: {e}")
            return 'unknown', 0


class ROIManager:
    """Manages regions of interest with persistence support"""
    
    ROI_CONFIG_FILE = 'roi_config.json'
    
    def __init__(self):
        self.tl_roi = None  # Legacy - kept for compatibility
        self.tl_red_roi = None  # Red light ROI
        self.tl_green_roi = None  # Green light ROI
        self.people_roi = None
        # Auto-load saved ROIs on startup
        self.load_rois_from_file()
    
    def set_roi(self, roi_type, points):
        """Set ROI for traffic light or people detection"""
        if roi_type == 'tl':
            self.tl_roi = points
        elif roi_type == 'tl_red':
            self.tl_red_roi = points
        elif roi_type == 'tl_green':
            self.tl_green_roi = points
        elif roi_type == 'people':
            self.people_roi = points
        else:
            return False
        # Auto-save whenever ROI is set
        self.save_rois_to_file()
        return True
    
    def reset_roi(self, roi_type):
        """Reset ROI for traffic light or people detection"""
        if roi_type == 'all' or roi_type == 'tl':
            self.tl_roi = None
        if roi_type == 'all' or roi_type == 'tl_red':
            self.tl_red_roi = None
        if roi_type == 'all' or roi_type == 'tl_green':
            self.tl_green_roi = None
        if roi_type == 'all' or roi_type == 'people':
            self.people_roi = None
        # Auto-save after reset
        self.save_rois_to_file()
        return True
    
    def get_roi_status(self):
        """Get the status of ROIs"""
        return {
            'tl_roi_defined': self.tl_roi is not None and len(self.tl_roi) >= 4,
            'tl_red_roi_defined': self.tl_red_roi is not None and len(self.tl_red_roi) >= 4,
            'tl_green_roi_defined': self.tl_green_roi is not None and len(self.tl_green_roi) >= 4,
            'people_roi_defined': self.people_roi is not None and len(self.people_roi) >= 4
        }
    
    def save_rois_to_file(self):
        """Save current ROIs to JSON file for persistence"""
        import json
        config = {
            'tl': self.tl_roi,
            'tl_red': self.tl_red_roi,
            'tl_green': self.tl_green_roi,
            'people': self.people_roi
        }
        try:
            config_path = os.path.join(os.path.dirname(__file__), self.ROI_CONFIG_FILE)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.warning(f"ROIs saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save ROIs: {e}")
    
    def load_rois_from_file(self):
        """Load ROIs from JSON file on startup"""
        import json
        try:
            config_path = os.path.join(os.path.dirname(__file__), self.ROI_CONFIG_FILE)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.tl_roi = config.get('tl')
                self.tl_red_roi = config.get('tl_red')
                self.tl_green_roi = config.get('tl_green')
                self.people_roi = config.get('people')
                logger.warning(f"ROIs loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load ROIs: {e}")
    
    @staticmethod
    def draw_roi(frame, roi, color, label):
        """Draw a single ROI on the frame with label"""
        if roi is not None and len(roi) >= 4:
            # Scale normalized points to frame dimensions
            h, w = frame.shape[:2]
            scaled_roi = []
            for point in roi:
                # Check if point is normalized (float <= 1.0) or absolute
                if isinstance(point[0], float) and point[0] <= 1.0:
                    scaled_roi.append([int(point[0] * w), int(point[1] * h)])
                else:
                    scaled_roi.append([int(point[0]), int(point[1])])
            
            pts = np.array(scaled_roi, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, 2)
            
            min_x = min(point[0] for point in scaled_roi)
            min_y = min(point[1] for point in scaled_roi)
            
            # Dynamic font scale for ROI label
            base_width = 640.0
            scale_factor = frame.shape[1] / base_width
            font_scale = max(0.4, 0.6 * scale_factor)
            font_thickness = max(1, int(2 * scale_factor))
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            label_bg_x1 = min_x - 5
            label_bg_y1 = min_y - label_size[1] - 15
            label_bg_x2 = min_x + label_size[0] + 5
            label_bg_y2 = min_y - 5
            
            cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
            
            cv2.putText(frame, label, (min_x, min_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    @staticmethod
    def get_bounding_rect(points, frame_shape=None):
        """Get the bounding rectangle from a list of points"""
        if len(points) < 2:
            return None
        
        # Scale points if frame_shape is provided and points are normalized
        if frame_shape is not None:
            h, w = frame_shape[:2]
            scaled_points = []
            for p in points:
                if isinstance(p[0], float) and p[0] <= 1.0:
                    scaled_points.append([int(p[0] * w), int(p[1] * h)])
                else:
                    scaled_points.append([int(p[0]), int(p[1])])
            x_coords = [p[0] for p in scaled_points]
            y_coords = [p[1] for p in scaled_points]
        else:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
        
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        
        return (x1, y1, x2, y2)
    
    @staticmethod
    def create_mask_from_points(frame_shape, points):
        """Create a binary mask from polygon points"""
        if len(points) < 4:
            return None
        
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        h, w = frame_shape[:2]
        
        # Scale normalized points
        scaled_points = []
        for point in points:
            if isinstance(point[0], float) and point[0] <= 1.0:
                scaled_points.append([int(point[0] * w), int(point[1] * h)])
            else:
                scaled_points.append([int(point[0]), int(point[1])])
                
        pts = np.array(scaled_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        
        return mask


class MessageManager:
    """Manages message display, audio alerts, and LED control via abstractions."""
    
    def __init__(self, settings, led_controller: LEDController, audio_player: AudioPlayer, global_state, linked_cameras=None):
        self.current_tl_state = None
        self.message_start_time = 0
        self.show_message = False
        self.message_text = ""
        self.last_periodic_message_time = 0
        self.person_in_roi = False
        
        self.led_controller = led_controller
        self.audio_player = audio_player
        self.global_state = global_state
        self.linked_cameras = linked_cameras
        
        # Audio settings
        self.last_berhenti_play_time = 0
        self.last_melintas_play_time = 0
        
        # LED settings
        self.led_enabled = False
        self.led_blink_interval = 0.5
        self.led_thread = None
        self.stop_led_event = threading.Event()
        
        # State tracking for continuous mode
        self.go_mode_active = False
        
        # Create audio directory if it doesn't exist
        os.makedirs(Config.AUDIO_DIR, exist_ok=True)
        
        # Create default audio files if they don't exist
        self._create_default_audio_files()

        # Apply initial settings
        self.update_settings(settings)
        logger.warning(f"MessageManager initialized with linked cameras: {self.linked_cameras}")
    
    def _create_default_audio_files(self):
        """Create default audio files if they don't exist"""
        # NOTE: This function still requires 'ffmpeg' to be installed on the system.
        default_berhenti_path = os.path.join(Config.AUDIO_DIR, 'berhenti_default.wav')
        if not os.path.exists(default_berhenti_path):
            try:
                subprocess.run([
                    'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=800:duration=0.5',
                    '-ar', '48000', '-c:a', 'pcm_s16le', '-y', default_berhenti_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.warning(f"Created default berhenti_default.wav audio file")
            except Exception as e:
                logger.error(f"Failed to create berhenti_default.wav: {e}. 'ffmpeg' must be installed.")
        
        default_melintas_path = os.path.join(Config.AUDIO_DIR, 'melintas_default.wav')
        if not os.path.exists(default_melintas_path):
            try:
                subprocess.run([
                    'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=1200:duration=0.5',
                    '-ar', '48000', '-c:a', 'pcm_s16le', '-y', default_melintas_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.warning(f"Created default melintas_default.wav audio file")
            except Exception as e:
                logger.error(f"Failed to create melintas_default.wav: {e}. 'ffmpeg' must be installed.")

    
    def _start_continuous_led_blink(self):
        """Start a thread to blink the LED continuously"""
        if not self.led_enabled:
            return
            
        if self.led_thread is None or not self.led_thread.is_alive():
            self.stop_led_event.clear()
            self.led_thread = threading.Thread(target=self._blink_led_loop)
            self.led_thread.daemon = True
            self.led_thread.start()
            logger.warning("Started continuous LED blinking")
    
    def _blink_led_loop(self):
        """Loop to blink LED"""
        while not self.stop_led_event.is_set():
            if not self.led_enabled:
                break
                
            self.led_controller.turn_on()
            # Wait for interval or stop event
            if self.stop_led_event.wait(self.led_blink_interval):
                break
                
            if not self.led_enabled:
                break
                
            self.led_controller.turn_off()
            # Wait for interval or stop event
            if self.stop_led_event.wait(self.led_blink_interval):
                break
        self.led_controller.turn_off()
    
    def _stop_continuous_led_blink(self):
        """Stop the continuous LED blinking thread"""
        if not self.led_enabled:
            return
        # logger.warning("Stopping continuous LED blinking") # Reduce log spam
        self.stop_led_event.set()
        if self.led_thread is not None:
            self.led_thread.join(timeout=1.0)
            self.led_thread = None
        self.led_controller.stop_blink() # Ensure it's off
    
    def _play_audio_only(self, audio_path):
        """Play audio using the abstracted audio_player"""
        self.audio_player.play_audio(audio_path)
    
    def update_settings(self, settings):
        """Update message manager settings"""
        # These are now initialized in __init__, but we keep this method for runtime updates if needed
        self.message_display_duration = settings.get('message_duration', 3)
        self.periodic_message_interval = settings.get('message_interval', 5)
        self.audio_cooldown = settings.get('audio_cooldown', 3)
        
        # Update audio filenames and paths
        self.berhenti_audio_filename = settings.get('berhenti_audio_filename', 'berhenti_default.wav')
        self.melintas_audio_filename = settings.get('melintas_audio_filename', 'melintas_default.wav')
        self.berhenti_audio_path = os.path.join(Config.AUDIO_DIR, self.berhenti_audio_filename)
        self.melintas_audio_path = os.path.join(Config.AUDIO_DIR, self.melintas_audio_filename)
        
        # Update LED settings
        self.led_enabled = settings.get('led_enabled', False) # Default to False
        self.led_blink_interval = settings.get('led_blink_interval', 0.5)
        
        logger.warning(f"MessageManager settings updated. LED Enabled: {self.led_enabled}")

    def update_state(self):
        """Update the state based on GLOBAL state and trigger audio alerts.
        
        NEW STATE MACHINE:
        - IDLE: Waiting for person in ROI for 2 seconds
        - WAITING_FOR_GREEN: Person confirmed, playing "Sila Tunggu", checking for green
        - CROSSING_MODE: Green detected, playing "Sila Melintas" + LED for 20 seconds
        
        Key behaviors:
        - Person must stay in ROI for 2 seconds to trigger
        - If person leaves before 2 sec, timer resets
        - Single-person lock prevents double triggers
        - Green detected anytime -> immediate switch to crossing mode
        - Red without person in ROI is ignored
        """
        current_time = time.time()
        
        # Initialize state variables on first run
        if not hasattr(self, '_state'):
            self._state = 'IDLE'
            self._person_first_seen_time = None
            self._person_detection_locked = False
            self._wait_start_time = None
            self._crossing_start_time = None
            self._sila_tunggu_play_count = 0
            self._audio_stop_requested = False
        
        # Get current global state
        global_state = self.global_state.get_state()
        person_in_roi = global_state['person_detected']
        
        # Check light states from separate ROIs (tl_red and tl_green)
        # For now, we use the existing traffic_light_state until CameraProcessor is updated
        tl_state = global_state['traffic_light_state']
        is_green = tl_state == 'go'
        is_red = tl_state == 'stop'
        
        # ==================== STATE: IDLE ====================
        if self._state == 'IDLE':
            # Check for person in ROI
            if person_in_roi and not self._person_detection_locked:
                if self._person_first_seen_time is None:
                    # Person just entered ROI, start timer
                    self._person_first_seen_time = current_time
                    logger.debug("Person entered ROI, starting 2-second timer")
                elif current_time - self._person_first_seen_time >= 2.0:
                    # Person has been in ROI for 2 seconds!
                    logger.warning("PERSON CONFIRMED (2 sec) - Checking light state")
                    self._person_detection_locked = True  # Lock to prevent re-trigger
                    
                    # Check light immediately
                    if is_green:
                        # Green light - go directly to crossing mode
                        logger.warning("GREEN LIGHT DETECTED - Starting crossing mode")
                        self._start_crossing_mode(current_time)
                    else:
                        # Red light or unknown - play "Sila Tunggu" and wait
                        logger.warning("RED/UNKNOWN LIGHT - Playing Sila Tunggu")
                        self._play_audio_only(self.berhenti_audio_path)  # Sila Tunggu
                        self._sila_tunggu_play_count = 1
                        self._wait_start_time = current_time
                        self._state = 'WAITING_FOR_GREEN'
                        
                        self.show_message = True
                        self.message_start_time = current_time
                        self.message_text = "Sila Tunggu"
            else:
                # Person not in ROI or left - reset timer
                if self._person_first_seen_time is not None:
                    logger.debug("Person left ROI before 2 seconds - timer reset")
                self._person_first_seen_time = None
        
        # ==================== STATE: WAITING_FOR_GREEN ====================
        elif self._state == 'WAITING_FOR_GREEN':
            # Continuously check for green light
            if is_green:
                # Green detected! Cut off Sila Tunggu audio and switch to crossing mode
                logger.warning("GREEN DETECTED during wait - Cutting off audio and switching to crossing mode")
                # Stop Sila Tunggu audio immediately
                if self.audio_player:
                    self.audio_player.stop_audio()
                self._start_crossing_mode(current_time)
            
            # Check if still in this state (not switched to crossing)
            elif self._state == 'WAITING_FOR_GREEN':
                elapsed = current_time - self._wait_start_time
                
                # Every 10 seconds, check light and repeat audio if still red and person present
                if elapsed >= 10.0:
                    if person_in_roi and is_red:
                        # Still red AND person in ROI - play "Sila Tunggu" again
                        logger.warning(f"10 sec elapsed, still RED + person - Playing Sila Tunggu (#{self._sila_tunggu_play_count + 1})")
                        self._play_audio_only(self.berhenti_audio_path)
                        self._sila_tunggu_play_count += 1
                        self._wait_start_time = current_time  # Reset wait timer
                        
                        self.show_message = True
                        self.message_start_time = current_time
                        self.message_text = "Sila Tunggu"
                    elif not person_in_roi:
                        # Person left ROI during wait - reset to IDLE
                        logger.warning("Person left ROI during wait - Resetting to IDLE")
                        self._reset_to_idle()
        
        # ==================== STATE: CROSSING_MODE ====================
        elif self._state == 'CROSSING_MODE':
            elapsed = current_time - self._crossing_start_time
            
            # Check for red light detection - cut off crossing immediately (no person check needed)
            if is_red:
                logger.warning("RED detected during crossing - Cutting off audio and LED immediately")
                self._stop_continuous_led_blink()
                # Stop audio playback immediately
                if self.audio_player:
                    self.audio_player.stop_audio()
                
                # Reset to IDLE
                self._state = 'IDLE'
                self._person_detection_locked = False  # Unlock for re-detection
                self._crossing_start_time = None
                
                # If person is in ROI, start their timer for Sila Tunggu
                if person_in_roi:
                    self._person_first_seen_time = current_time
                    logger.warning("Person in ROI - Starting 2-second timer for Sila Tunggu")
                else:
                    self._person_first_seen_time = None
                    logger.warning("No person in ROI - Going back to IDLE")
                return  # Exit early, next update_state call will handle the IDLE state
            
            # After 20 seconds, end crossing mode normally
            if elapsed >= 20.0:
                logger.warning("Crossing mode ended (20 seconds)")
                self._stop_continuous_led_blink()
                self._reset_to_idle()
        
        # Update message visibility
        if self.show_message:
            if current_time - self.message_start_time >= self.message_display_duration:
                self.show_message = False
    
    def _start_crossing_mode(self, current_time):
        """Start crossing mode with audio and LED"""
        self._state = 'CROSSING_MODE'
        self._crossing_start_time = current_time
        
        # Play "Sila Melintas" once
        self._play_audio_only(self.melintas_audio_path)
        
        # Start LED blinking for 20 seconds
        if self.led_enabled:
            self._start_continuous_led_blink()
        
        self.show_message = True
        self.message_start_time = current_time
        self.message_text = "Sila Melintas"
    
    def _reset_to_idle(self):
        """Reset all state to IDLE"""
        self._state = 'IDLE'
        self._person_first_seen_time = None
        self._person_detection_locked = False  # Unlock for next person
        self._wait_start_time = None
        self._crossing_start_time = None
        self._sila_tunggu_play_count = 0
        self._audio_stop_requested = False
    
    def draw_message(self, frame):
        """Draw the message on the frame if needed (Unchanged)"""
        if not self.show_message:
            return frame
        
        display_frame = frame.copy()
        overlay = display_frame.copy()
        
        # Dynamic font scale
        base_width = 640.0
        scale_factor = display_frame.shape[1] / base_width
        font_scale = max(0.8, 1.5 * scale_factor)
        font_thickness = max(2, int(3 * scale_factor))
        
        text_size = cv2.getTextSize(self.message_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        text_x = display_frame.shape[1] - text_size[0] - 20
        text_y = text_size[1] + 20
        
        bg_x1 = text_x - 10
        bg_y1 = text_y - text_size[1] - 10
        bg_x2 = text_x + text_size[0] + 10
        bg_y2 = text_y + 10
        
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        text_color = (0, 0, 255) if self.current_tl_state == 'stop' else (0, 255, 0)
        
        cv2.putText(display_frame, self.message_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        
        return display_frame
    
    def cleanup(self):
        """Clean up resources"""
        # Stop LED blinking
        self._stop_continuous_led_blink()


class GlobalDetectionState:
    """Thread-safe global state for detection results across all cameras"""
    def __init__(self):
        self.lock = threading.Lock()
        # Store state per camera: {cam_index: {'person': bool, 'tl': str, 'last_person': float, 'last_tl': float}}
        self.camera_states = {}

    def update_person(self, detected, cam_index):
        with self.lock:
            if cam_index not in self.camera_states:
                self.camera_states[cam_index] = {
                    'person': False, 'tl': 'unknown', 
                    'last_person': 0, 'last_tl': 0
                }
            
            state = self.camera_states[cam_index]
            if detected:
                state['person'] = True
                state['last_person'] = time.time()
            else:
                # Only clear if enough time has passed
                if time.time() - state['last_person'] > 2.0:
                    state['person'] = False

    def update_tl(self, tl_state, cam_index):
        with self.lock:
            if cam_index not in self.camera_states:
                self.camera_states[cam_index] = {
                    'person': False, 'tl': 'unknown', 
                    'last_person': 0, 'last_tl': 0
                }
            
            state = self.camera_states[cam_index]
            # TrafficLightDetector returns 'go', 'stop', or 'unknown'
            if tl_state in ['go', 'stop', 'unknown']:
                state['tl'] = tl_state
                state['last_tl'] = time.time()
            elif time.time() - state['last_tl'] > 5.0:
                state['tl'] = 'unknown'

    def get_state(self, cam_indices=None):
        """
        Get aggregated state.
        If cam_indices is provided, aggregate only for those cameras.
        Otherwise, aggregate all cameras.
        """
        with self.lock:
            person_detected = False
            traffic_light_state = 'unknown'
            
            # Determine which cameras to check
            targets = cam_indices if cam_indices is not None else self.camera_states.keys()
            
            # Check person detection (OR logic - any camera detects person)
            for cam_idx in targets:
                if cam_idx in self.camera_states:
                    if self.camera_states[cam_idx]['person']:
                        person_detected = True
                        break
            
            # Check traffic light state
            # Priority: 'go' > 'stop' > 'unknown'
            # If any linked camera sees 'go', use 'go'
            found_states = []
            for cam_idx in targets:
                if cam_idx in self.camera_states:
                    s = self.camera_states[cam_idx]['tl']
                    if s in ['go', 'stop']:
                        found_states.append(s)
            
            # Use 'go' if any camera sees it, otherwise use 'stop' if seen
            if 'go' in found_states:
                traffic_light_state = 'go'
            elif 'stop' in found_states:
                traffic_light_state = 'stop'
            
            return {
                'person_detected': person_detected,
                'traffic_light_state': traffic_light_state
            }


class CameraProcessor:
    """Handles camera capture, recording, and processing (Simplified)"""
    
    def __init__(self, model_manager, settings, led_controller, audio_player, global_state, camera_index=0, camera_name="Camera 1"):
        self.model_manager = model_manager
        self.settings = settings
        self.led_controller = led_controller
        self.audio_player = audio_player
        self.global_state = global_state
        self.camera_index = camera_index  # Can be int (index) or str (device path)
        self.camera_name = camera_name
        self.cap = None
        self.stop_event = threading.Event()
        self.is_initialized = False
        self.restart_requested = False # Flag for thread-safe restart
        self.retry_count = settings.get('camera_retry_count', 2)
        
        self.video_writer = None
        self.recording_start_time = None
        self.current_recording_path = None
        self.recording_fps = settings.get('target_fps', 10)
        self.recording_frame_count = 0
        self.recording_segment_count = 0
        
        self.recording_thread = None
        self.web_thread = None
        self.frame_queue = queue.Queue(maxsize=30)
        
        self.roi_manager = ROIManager()
    
        # Determine linked cameras for this camera
        linked_cameras = None
        camera_links = self.settings.get('camera_links', {})
        cam_key = str(self.camera_index)
        
        if cam_key in camera_links:
            # Start with self
            linked_cameras = [self.camera_index]
            # Add linked cameras
            for link in camera_links[cam_key]:
                # Match type of camera_index (int or str)
                if isinstance(self.camera_index, int):
                    try:
                        linked_cameras.append(int(link))
                    except (ValueError, TypeError):
                        linked_cameras.append(link)
                else:
                    linked_cameras.append(link)
            logger.warning(f"Camera {self.camera_index} linked with: {linked_cameras}")

        # Pass the hardware controllers AND global_state to the MessageManager
        self.message_manager = MessageManager(self.settings, led_controller, audio_player, global_state, linked_cameras=linked_cameras)
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.fps_update_interval = 5
        
        # Use the general RECORDINGS_DIR from Config
        self.camera_recordings_dir = os.path.join(Config.RECORDINGS_DIR, camera_name.replace(' ', '_'))
        os.makedirs(self.camera_recordings_dir, exist_ok=True)
        logger.warning(f"Created camera recordings directory: {self.camera_recordings_dir}")
    

    def initialize_camera(self):
        """Initialize the camera with general settings"""
        logger.warning(f"Attempting to initialize camera {self.camera_index} ({self.camera_name})")

        # Determine backend based on camera_index type
        is_device_path = isinstance(self.camera_index, str)
        backend = cv2.CAP_V4L2 if is_device_path else cv2.CAP_ANY
        backend_name = "CAP_V4L2" if is_device_path else "CAP_ANY"

        logger.warning(f"Using backend: {backend_name} for camera {self.camera_index}")

        for attempt in range(self.retry_count):
            try:
                if self.cap is not None:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                
                if self.cap.isOpened():
                    # Set resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.get('processing_width', 640))
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.get('processing_height', 480))
                    self.cap.set(cv2.CAP_PROP_FPS, self.settings.get('target_fps', 10))
                    
                    # Read back actual properties
                    self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                    
                    # Test read
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        logger.warning(f"Camera {self.camera_index} initialized successfully")
                        self.is_initialized = True
                        return True
                    else:
                        logger.warning(f"Camera {self.camera_index} opened but failed to read frame")
                else:
                    logger.warning(f"Failed to open camera {self.camera_index}")
                
            except Exception as e:
                logger.error(f"Error initializing camera {self.camera_index}: {e}")
            
            time.sleep(self.settings.get('camera_init_delay', 3.0))
        
        logger.error(f"Failed to initialize camera {self.camera_index} after {self.retry_count} attempts")
        self.is_initialized = False
        return False
    
    def start_recording_if_enabled(self):
        """Start recording if enabled in settings"""
        if self.settings.get('enable_recording', True):
            self.start_recording()
            logger.warning(f"Recording enabled for {self.camera_name}")
        else:
            logger.warning(f"Recording disabled for {self.camera_name}")
    
    def start_threads(self):
        """Start all background processing threads"""
        # Start the main capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        logger.warning(f"Started capture thread for {self.camera_name}")

        # Start the recording thread
        self.recording_thread = threading.Thread(target=self.recording_worker, daemon=True)
        self.recording_thread.start()
        logger.warning(f"Started recording thread for {self.camera_name}")
        
        # Start the web processing thread
        self.web_thread = threading.Thread(target=self.web_worker, daemon=True)
        self.web_thread.start()
        logger.warning(f"Started web processing thread for {self.camera_name}")
    
    def release_camera(self):
        """Release the camera and stop recording"""
        self.stop_event.set()

        # ADD THIS CHECK:
        if hasattr(self, 'capture_thread') and self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.recording_thread is not None and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        if self.web_thread is not None and self.web_thread.is_alive():
            self.web_thread.join(timeout=1.0)
        
        self.stop_recording()
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def stop(self):
        """Stop the camera processor"""
        self.release_camera()
    
    def _is_codec_available(self, codec):
        """Check if a specific codec is available"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_path = os.path.join(self.camera_recordings_dir, 'temp_test.mp4')
            test_writer = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))
            if test_writer.isOpened():
                test_writer.release()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return True
        except Exception as e:
            logger.warning(f"Codec {codec} test failed: {e}")
        return False
    
    def start_recording(self):
        """Start a new recording session with MP4 format and daily folder organization"""
        self.stop_recording()
        
        # Create daily folder based on current date
        current_date = datetime.now()
        daily_folder = current_date.strftime("%Y-%m-%d")
        daily_path = os.path.join(self.camera_recordings_dir, daily_folder)
        os.makedirs(daily_path, exist_ok=True)
        logger.warning(f"Created/using daily folder: {daily_path}")
        
        # Generate filename with timestamp
        timestamp = current_date.strftime("%Y%m%d_%H%M%S")
        
        # More universal codec list. 'mp4v' is often the most reliable.
        codecs_to_try = [
            (self.settings.get('preferred_codec', 'mp4v'), 'mp4'),
            ('mp4v', 'mp4'),
            ('H264', 'mp4'),
            ('XVID', 'mp4'), # XVID in MP4 container
            ('XVID', 'avi'), # Fallback to AVI if MP4 fails
        ]
        
        fourcc = None
        for codec, ext in codecs_to_try:
            if self._is_codec_available(codec):
                filename = f"{timestamp}.{ext}"
                self.current_recording_path = os.path.join(daily_path, filename)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                logger.warning(f"Using codec {codec} for {ext} recording")
                break
        
        if fourcc is None:
            logger.error("No suitable video codec found. Recording is disabled.")
            return

        target_fps = self.settings.get('target_fps', 10)
        logger.warning(f"Setting recording FPS to: {target_fps}")
        logger.warning(f"Recording will be saved to: {self.current_recording_path}")
        
        try:
            self.video_writer = cv2.VideoWriter(
                self.current_recording_path, 
                fourcc, 
                target_fps,
                (self.frame_width, self.frame_height)
            )
            
            if self.video_writer.isOpened():
                self.recording_start_time = time.time()
                self.recording_frame_count = 0
                logger.warning(f"Started recording for {self.camera_name}: {self.current_recording_path} at {target_fps} FPS")
            else:
                logger.error("Failed to open video writer")
                self.video_writer = None
        except Exception as e:
            logger.error(f"Error initializing video writer: {e}")
            self.video_writer = None
    
    def stop_recording(self):
        """Stop the current recording session"""
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception as e:
                logger.error(f"Error releasing video writer: {e}")
            finally:
                self.video_writer = None
            
            if self.current_recording_path:
                logger.warning(f"Stopped recording for {self.camera_name}: {self.current_recording_path}")
                logger.warning(f"Recorded {self.recording_frame_count} frames")
                
                if self.settings.get('validate_recordings', True):
                    if self.validate_recording(self.current_recording_path):
                        logger.warning(f"Recording validation successful for {self.current_recording_path}")
                    else:
                        logger.error(f"Recording validation failed for {self.current_recording_path}")
                        self._fix_corrupted_recording(self.current_recording_path)
                
                # Run faststart optimization for web streaming
                try:
                    if self.current_recording_path.endswith('.mp4') and os.path.exists(self.current_recording_path):
                        self._faststart_mp4(self.current_recording_path)
                except Exception as e:
                    logger.error(f"Faststart optimization failed: {e}")
                
                self.current_recording_path = None
            
            self.cleanup_old_recordings()
    
    def validate_recording(self, filepath):
        """Validate if a recorded video file is playable"""
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return False
            
            for _ in range(5):
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    return False
            
            cap.release()
            return True
        except Exception as e:
            logger.error(f"Error validating recording {filepath}: {e}")
            return False
    
    def _fix_corrupted_recording(self, input_path):
        """Attempt to fix a corrupted recording using ffmpeg to MP4"""
        try:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            daily_folder = os.path.basename(os.path.dirname(input_path))
            fixed_path = os.path.join(self.camera_recordings_dir, daily_folder, f"{base_name}_fixed.mp4")
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y',
                fixed_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.warning(f"Fixed corrupted recording to MP4: {input_path} -> {fixed_path}")
                try:
                    os.remove(input_path)
                except:
                    pass
                return True
            else:
                logger.error(f"Failed to fix recording to MP4: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error fixing recording to MP4: {e}")
            return False
    
    def _faststart_mp4(self, input_path):
        """Run ffmpeg faststart remux to improve web playback without re-encoding."""
        try:
            # --- NEW CROSS-PLATFORM BINARY FINDER ---
            script_dir = os.path.dirname(os.path.realpath(__file__))
            
            # Determine OS and the correct path/filename
            if os.name == 'nt': # Windows
                platform_folder = 'win'
                ffmpeg_name = 'ffmpeg.exe'
            elif sys.platform == 'darwin': # macOS
                platform_folder = 'mac'
                ffmpeg_name = 'ffmpeg'
            else: # Assume Linux
                platform_folder = 'linux'
                ffmpeg_name = 'ffmpeg'
            
            # Build the path to the bundled executable
            ffmpeg_path = os.path.join(script_dir, 'bin', platform_folder, ffmpeg_name)
            
            if not os.path.exists(ffmpeg_path):
                # Fallback: try to find 'ffmpeg' in the system's PATH
                ffmpeg_path = 'ffmpeg'
                if not shutil.which(ffmpeg_path): # shutil.which checks the system PATH
                    logger.error("FATAL: 'ffmpeg' executable not found.")
                    logger.error(f"Please place it in 'bin/{platform_folder}/' or install it system-wide.")
                    return False
                else:
                    logger.warning("Using system-wide 'ffmpeg' found in PATH.")
            # --- END NEW CODE ---

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            daily_folder = os.path.basename(os.path.dirname(input_path))
            temp_path = os.path.join(self.camera_recordings_dir, daily_folder, f"{base_name}_faststart.mp4")
            
            cmd = [
                ffmpeg_path,
                '-i', input_path,
                '-c', 'copy',
                '-movflags', '+faststart',
                '-y',
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                try:
                    os.replace(temp_path, input_path)
                    logger.warning(f"Applied faststart to {input_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed replacing original with faststart copy: {e}")
            else:
                logger.error(f"Faststart remux failed: {result.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error during faststart remux: {e}")
            return False
    
    def cleanup_old_recordings(self):
        """Remove recordings older than the retention period (default 30 days)"""
        retention_days = self.settings.get('retention_days', 30)
        current_time = time.time()
        cutoff_time = current_time - (retention_days * 24 * 60 * 60)

        try:
            daily_folders = [f for f in os.listdir(self.camera_recordings_dir) 
                           if os.path.isdir(os.path.join(self.camera_recordings_dir, f))]
            
            for folder in daily_folders:
                folder_path = os.path.join(self.camera_recordings_dir, folder)
                
                # Check all supported video extensions
                for ext in ('*.mp4', '*.avi'):
                    recordings = glob.glob(os.path.join(folder_path, ext))
                    
                    for recording in recordings:
                        try:
                            file_mtime = os.path.getmtime(recording)
                            if file_mtime < cutoff_time:
                                os.remove(recording)
                                logger.warning(f"Removed old recording for {self.camera_name}: {recording}")
                        except Exception as e:
                            logger.error(f"Failed to remove recording {recording}: {e}")
                
                try:
                    if not os.listdir(folder_path):
                        os.rmdir(folder_path)
                        logger.warning(f"Removed empty folder: {folder_path}")
                except Exception as e:
                    logger.error(f"Failed to remove folder {folder_path}: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup for {self.camera_name}: {e}")

    def recording_worker(self):
        """Worker thread for continuous recording"""
        logger.warning(f"Recording worker started for {self.camera_name}")
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                recording_frame = self.add_timestamp(frame.copy(), position="bottom-left")
                
                if self.video_writer is not None and self.video_writer.isOpened():
                    try:
                        if recording_frame.dtype != np.uint8:
                            recording_frame = recording_frame.astype(np.uint8)
                        
                        self.video_writer.write(recording_frame)
                        self.recording_frame_count += 1
                    except Exception as e:
                        logger.error(f"Error writing frame to recording: {e}")
                        self.stop_recording()
                        self.start_recording()
                else:
                    if self.settings.get('enable_recording', True) and self.is_initialized:
                        logger.warning(f"Video writer not open, attempting to restart recording for {self.camera_name}")
                        self.start_recording()
                
                recording_duration = self.settings.get('recording_duration', 300)
                if self.recording_start_time and (time.time() - self.recording_start_time >= recording_duration):
                    logger.warning(f"{recording_duration} seconds elapsed, starting new recording segment")
                    self.recording_segment_count += 1
                    self.start_recording()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in recording worker for {self.camera_name}: {e}")
        
        logger.warning(f"Recording worker stopped for {self.camera_name}")
    
    def web_worker(self):
        """Worker thread for web processing"""
        logger.warning(f"Web worker started for {self.camera_name}")
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                web_frame = self.process_web_frame(frame)
                
                self.frame_count += 1
                current_time = time.time()
                
                if self.frame_count % self.fps_update_interval == 0:
                    elapsed = current_time - self.last_fps_time
                    if elapsed > 0:
                        self.current_fps = self.fps_update_interval / elapsed
                    self.last_fps_time = current_time
                
                with threading.Lock():
                    self.latest_web_frame = web_frame.copy()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in web worker for {self.camera_name}: {e}")
        
        logger.warning(f"Web worker stopped for {self.camera_name}")
    
    def add_timestamp(self, frame, position="bottom-left"):
        """Draw timestamp on the frame"""
        if not self.settings.get('show_timestamp', True):
            return frame
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Dynamic font scaling based on frame width
        # Base: 0.7 scale, 2 thickness for 640px width
        frame_width = frame.shape[1]
        base_width = 640.0
        scale_factor = frame_width / base_width
        
        font_scale = max(0.4, 0.7 * scale_factor) # Minimum scale 0.4
        font_thickness = max(1, int(2 * scale_factor))
        
        text_size = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        if position == "bottom-left":
            x = 10
            y = frame.shape[0] - 10
        elif position == "bottom-right":
            x = frame.shape[1] - text_size[0] - 10
            y = frame.shape[0] - 10
        elif position == "top-left":
            x = 10
            y = text_size[1] + 10
        elif position == "top-right":
            x = frame.shape[1] - text_size[0] - 10
            y = text_size[1] + 10
        else:
            x = 10
            y = frame.shape[0] - 10
        
        bg_x1 = x - 5
        bg_y1 = y - text_size[1] - 5
        bg_x2 = x + text_size[0] + 5
        bg_y2 = y + 5
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        cv2.putText(frame, current_time, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 255, 255), font_thickness)
        
        return frame
    
    def apply_digital_zoom(self, frame):
        """Apply digital zoom to the frame based on settings"""
        try:
            # Get zoom config for this camera
            zoom_configs = self.settings.get('camera_zoom_configs', {})
            # Try both string and int keys
            config = zoom_configs.get(str(self.camera_index), zoom_configs.get(self.camera_index, {}))
            
            # Handle old format (just a float) or new format (dict)
            if isinstance(config, (int, float)):
                zoom_factor = float(config)
                center_x = 0.5
                center_y = 0.5
            else:
                zoom_factor = float(config.get('level', 1.0))
                center_x = float(config.get('x', 0.5))
                center_y = float(config.get('y', 0.5))
            
            if zoom_factor <= 1.0:
                return frame
            
            # Limit max zoom
            zoom_factor = min(zoom_factor, 5.0)
            
            h, w = frame.shape[:2]
            
            # Calculate new dimensions
            new_h = int(h / zoom_factor)
            new_w = int(w / zoom_factor)
            
            # Calculate top-left corner based on center coordinates
            # center_x/y are normalized (0.0-1.0)
            cx = int(center_x * w)
            cy = int(center_y * h)
            
            left = cx - (new_w // 2)
            top = cy - (new_h // 2)
            
            # Clamp to boundaries
            left = max(0, min(left, w - new_w))
            top = max(0, min(top, h - new_h))
            
            # Crop
            cropped = frame[top:top+new_h, left:left+new_w]
            
            # Resize back to original size
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return zoomed
            
        except Exception as e:
            logger.error(f"Error applying digital zoom: {e}")
            return frame

    def process_web_frame(self, frame):
        """Process the frame for detection and dashboard display"""
        # Apply digital zoom first - affects both detection and display
        frame = self.apply_digital_zoom(frame)
        
        display_frame = frame.copy()
        
        display_frame = self.add_timestamp(display_frame, "bottom-left")
        
        # Draw all ROIs - Red Light, Green Light, and People
        self.roi_manager.draw_roi(display_frame, self.roi_manager.tl_red_roi, 
                                Config.ROI_COLORS['tl_red'], "Red Light ROI")
        self.roi_manager.draw_roi(display_frame, self.roi_manager.tl_green_roi, 
                                Config.ROI_COLORS['tl_green'], "Green Light ROI")
        self.roi_manager.draw_roi(display_frame, self.roi_manager.people_roi, 
                                Config.ROI_COLORS['people'], self.settings['people_roi_label'])
        
        # Legacy single ROI (keep for backward compatibility)
        self.roi_manager.draw_roi(display_frame, self.roi_manager.tl_roi, 
                                Config.ROI_COLORS['tl'], self.settings['tl_roi_label'])
        
        # === SEPARATE RED/GREEN LIGHT DETECTION ===
        tl_state = None
        red_detected = False
        green_detected = False
        
        # Check RED light ROI
        if self.roi_manager.tl_red_roi is not None and len(self.roi_manager.tl_red_roi) >= 4:
            try:
                mask = self.roi_manager.create_mask_from_points(frame.shape, self.roi_manager.tl_red_roi)
                if mask is not None:
                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                    rect = self.roi_manager.get_bounding_rect(self.roi_manager.tl_red_roi, frame.shape)
                    if rect:
                        x1, y1, x2, y2 = rect
                        roi_frame = masked_frame[y1:y2, x1:x2]
                        if roi_frame.size > 0:
                            # Check specifically for RED in this ROI
                            tl_class, count = TrafficLightDetector.detect_state(roi_frame)
                            if tl_class == 'stop':  # Red detected
                                red_detected = True
            except Exception as e:
                logger.error(f"Red light ROI detection error: {e}")
        
        # Check GREEN light ROI
        if self.roi_manager.tl_green_roi is not None and len(self.roi_manager.tl_green_roi) >= 4:
            try:
                mask = self.roi_manager.create_mask_from_points(frame.shape, self.roi_manager.tl_green_roi)
                if mask is not None:
                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                    rect = self.roi_manager.get_bounding_rect(self.roi_manager.tl_green_roi, frame.shape)
                    if rect:
                        x1, y1, x2, y2 = rect
                        roi_frame = masked_frame[y1:y2, x1:x2]
                        if roi_frame.size > 0:
                            # Check specifically for GREEN in this ROI
                            tl_class, count = TrafficLightDetector.detect_state(roi_frame)
                            if tl_class == 'go':  # Green detected
                                green_detected = True
            except Exception as e:
                logger.error(f"Green light ROI detection error: {e}")
        
        # Determine final state based on separate ROI detections
        # Priority: Green > Red > Unknown (to favor crossing when safe)
        if green_detected:
            tl_state = 'go'
        elif red_detected:
            tl_state = 'stop'
        else:
            tl_state = 'unknown'
        
        # Display detection status
        # Dynamic font scale
        base_width = 640.0
        scale_factor = frame.shape[1] / base_width
        font_scale = max(0.4, 0.7 * scale_factor)
        font_thickness = max(1, int(2 * scale_factor))
        
        status_color = (0, 255, 0) if tl_state == 'go' else (0, 0, 255) if tl_state == 'stop' else (128, 128, 128)
        cv2.putText(display_frame, f"Light: {tl_state.upper()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, font_thickness)
        
        person_in_roi = False
        
        try:
            confidence_threshold = self.settings['confidence_threshold'] / 100.0
            
            detections = self.model_manager.detect_objects(frame, confidence_threshold)
            
            people = [d for d in detections if int(d[5]) == 0]
            
            for person in people:
                px1, py1, px2, py2 = map(int, person[:4])
                
                dot_x = px1
                dot_y = py2
                
                if self.roi_manager.people_roi is not None and len(self.roi_manager.people_roi) >= 4:
                    # Scale normalized points to frame dimensions
                    h, w = frame.shape[:2]
                    scaled_people_roi = []
                    for p in self.roi_manager.people_roi:
                        if isinstance(p[0], float) and p[0] <= 1.0:
                            scaled_people_roi.append([int(p[0] * w), int(p[1] * h)])
                        else:
                            scaled_people_roi.append([int(p[0]), int(p[1])])
                            
                    if cv2.pointPolygonTest(np.array(scaled_people_roi, np.int32), (dot_x, dot_y), False) >= 0:
                        person_in_roi = True
                        
                        if self.settings['show_bboxes']:
                            cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        
                        if self.settings['show_dots']:
                            cv2.circle(display_frame, (dot_x, dot_y), 4, (0, 0, 255), -1)
                        
                        # Dynamic font scale for person label
                        person_font_scale = max(0.3, 0.5 * scale_factor)
                        person_thickness = max(1, int(2 * scale_factor))
                        
                        cv2.putText(display_frame, "Person", (px1, py1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, person_font_scale, (0, 0, 255), person_thickness)
                    else:
                        if self.settings.get('debug_show_all_people', False):
                            cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 255, 0), 1)
                else:
                    pass
        except Exception as e:
            logger.error(f"People detection error: {e}")
        # Update GLOBAL state with local findings
        self.global_state.update_tl(tl_state if tl_state else 'unknown', self.camera_index)
        self.global_state.update_person(person_in_roi, self.camera_index)
        
        # Update message manager (it will check global state internally)
        self.message_manager.update_state()
        
        display_frame = self.message_manager.draw_message(display_frame)
        
        self.frame_count += 1
        current_time = time.time()
        
        if self.frame_count % self.fps_update_interval == 0:
            elapsed = current_time - self.last_fps_time
            if elapsed > 0:
                self.current_fps = self.fps_update_interval / elapsed
            self.last_fps_time = current_time
        
        fps_text = f"FPS: {self.current_fps:.1f}"
        fps_text = f"FPS: {self.current_fps:.1f}"
        # Dynamic font scale for FPS
        base_width = 640.0
        scale_factor = frame.shape[1] / base_width
        font_scale = max(0.4, 0.5 * scale_factor)
        font_thickness = max(1, int(1 * scale_factor))
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        x = display_frame.shape[1] - text_size[0] - 10
        y = text_size[1] + 10
        
        bg_x1 = x - 5
        bg_y1 = y - text_size[1] - 5
        bg_x2 = x + text_size[0] + 5
        bg_y2 = y + 5
        
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        cv2.putText(display_frame, fps_text, (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        return display_frame
    
    def close_camera(self):
        """Close the camera connection but keep threads running (for retry)"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_initialized = False
        logger.warning(f"Camera {self.camera_name} connection closed (threads kept alive)")

    def release_camera(self):
        """Release the camera and stop recording"""
        self.stop_event.set()

        if hasattr(self, 'capture_thread') and self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.recording_thread is not None and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        if self.web_thread is not None and self.web_thread.is_alive():
            self.web_thread.join(timeout=1.0)
        
        self.stop_recording()
        
        self.close_camera()

    def capture_frames(self):
        """Capture frames from camera and feed to processing queues with continuous retry"""
        
        self.message_manager.last_periodic_message_time = time.time()
        self.last_frame_time = time.time()
        self.last_fps_time = time.time()
        
        target_fps = self.settings.get('target_fps', 10)
        frame_interval = 1.0 / target_fps
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while not self.stop_event.is_set():
            # Check for restart request (thread-safe)
            if self.restart_requested:
                logger.warning(f"Restart requested for {self.camera_name}. Re-initializing...")
                self.stop_recording() # Stop recording before closing camera to ensure writer is recreated
                self.close_camera()
                self.restart_requested = False
                # Loop will continue and hit the initialization check below
                continue

            # Check if camera is initialized
            if not self.is_initialized or self.cap is None or not self.cap.isOpened():
                logger.warning(f"Camera {self.camera_name} not initialized. Attempting to connect...")
                if self.initialize_camera():
                    logger.warning(f"Camera {self.camera_name} connected successfully.")
                    consecutive_failures = 0
                else:
                    # Wait before retrying
                    time.sleep(5.0)
                    continue

            start_time = time.time()
            
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    logger.warning(f"Failed to capture frame from {self.camera_name} (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Max failures reached for {self.camera_name}. Closing connection for retry...")
                        self.close_camera() # Use close_camera instead of release_camera to keep thread alive
                        consecutive_failures = 0
                        time.sleep(1.0)
                    else:
                        time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    # logger.warning(f"Frame queue full for {self.camera_name}, dropping frame") # Reduce spam
                    pass
                
                elapsed = time.time() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                if self.frame_count % 100 == 0: # Log less frequently
                    actual_fps = 1.0 / elapsed if elapsed > 0 else 0
                    # logger.warning(f"Camera {self.camera_name} - Actual FPS: {actual_fps:.2f}")
                    
            except Exception as e:
                logger.error(f"Error in capture loop for {self.camera_name}: {e}")
                self.close_camera() # Use close_camera to allow retry
                time.sleep(1.0)
        
        self.close_camera()
    
    def generate_frames(self):
        """Generate frames for the video stream"""
        
        while not self.stop_event.is_set():
            try:
                with threading.Lock():
                    if hasattr(self, 'latest_web_frame'):
                        frame = self.latest_web_frame.copy()
                    else:
                        frame = np.zeros((self.settings.get('processing_height', 480), self.settings.get('processing_width', 640), 3), dtype=np.uint8)
                        cv2.putText(frame, f"Initializing {self.camera_name}...", 
                                    (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.error("ERROR: Failed to encode frame")
                    break
                
                frame_bytes = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                
                time.sleep(1.0 / self.settings.get('target_fps', 10))
                
            except Exception as e:
                logger.error(f"Error generating frames for {self.camera_name}: {e}")
                time.sleep(0.1)
        
        self.release_camera()
    
    def stop(self):
        """Stop the camera processing"""
        self.stop_event.set()


class WebApp:
    """Main Flask application class with generalized hardware."""
    
    def __init__(self):
        self.app = Flask(__name__)
        
        CORS(self.app, resources={
            r"/recordings/*": {
                "origins": "*",
                "methods": ["GET", "OPTIONS"],
                "allow_headers": ["Content-Type", "Range"]
            }
        })
        
        self.model_manager = ModelManager()
        self.settings = Config.DEFAULT_SETTINGS.copy()
        self.settings_lock = threading.Lock()
        
        # Create audio directory if it doesn't exist
        os.makedirs(Config.AUDIO_DIR, exist_ok=True)
        
        # Initialize hardware controllers
        self.led_controller = self._setup_led_controller()
        self.speaker_relay_controller = self._setup_speaker_relay_controller()
        
        # Initialize audio player
        self.audio_player = AudioPlayer(self.speaker_relay_controller)
        
        # Initialize global detection state
        self.global_state = GlobalDetectionState()
        
        # Initialize sensors
        self.dht11_sensor = DHT11Sensor(gpio_pin=4)
        self.voltage_sensor = VoltageSensor(channel=0, voltage_divider_ratio=5.05)  # Ratio from battery.py
        # ---
        
        # Detect cameras using device paths (if provided) or indices
        self.camera_processors = []
        
        # Use configured device paths if available, otherwise use indices
        camera_device_paths = self.settings.get('camera_device_paths', [])
        camera_indices = self.settings.get('camera_indices', [0])
        
        if camera_device_paths:
            logger.warning(f"Initializing cameras using device paths: {camera_device_paths}")
            for i, device_path in enumerate(camera_device_paths):
                cam_name = f"Camera {i+1}"
                processor = CameraProcessor(
                    self.model_manager, 
                    self.settings, 
                    self.led_controller, 
                    self.audio_player,
                    self.global_state, # Pass global state
                    camera_index=device_path, 
                    camera_name=cam_name
                )
                self.camera_processors.append(processor)
        else:
            logger.warning(f"Initializing cameras using indices: {camera_indices}")
            for i, cam_idx in enumerate(camera_indices):
                cam_name = f"Camera {i+1}"
                processor = CameraProcessor(
                    self.model_manager, 
                    self.settings, 
                    self.led_controller, 
                    self.audio_player,
                    self.global_state, # Pass global state
                    camera_index=cam_idx, 
                    camera_name=cam_name
                )
                self.camera_processors.append(processor)
        
        # Build camera_names list for templates
        self.camera_names = [proc.camera_name for proc in self.camera_processors]
        
        # Start threads for all camera processors (they will retry connection internally)
        for processor in self.camera_processors:
            processor.start_threads()
            logger.warning(f"Started threads for {processor.camera_name}")
        
        logger.warning(f"Initialized {len(self.camera_processors)} camera processors")
        logger.warning(f"Initial settings: {self.settings}")
        
        self._setup_routes()

    def _setup_led_controller(self):
        """Factory function to create the correct LED controller."""
        controller_type = self.settings.get('led_controller_type', 'auto').lower()
        pin = self.settings.get('led_pin', 17)
        
        if controller_type == 'jetson':
            if JETSON_GPIO_AVAILABLE:
                logger.warning("Forcing JetsonLEDController.")
                return JetsonLEDController(pin)
            else:
                logger.error("Jetson controller forced, but Jetson.GPIO library not found!")
                raise ImportError("Jetson.GPIO library is required but not installed")
        
        # Auto-detect or default fallbacks
        if GPIOD_AVAILABLE:
            logger.warning("Auto-detected gpiod. Using GpiodLEDController (Raspberry Pi 5).")
            return GpiodLEDController(pin)
        elif JETSON_GPIO_AVAILABLE:
            logger.warning("Auto-detected Jetson.GPIO. Using JetsonLEDController.")
            return JetsonLEDController(pin)
        elif RPI_GPIO_AVAILABLE:
            logger.warning("Auto-detected RPi.GPIO. Using RaspberryPiLEDController.")
            return RaspberryPiLEDController(pin)
        else:
            logger.warning("No GPIO library found. Using DummyLEDController (Simulation Mode).")
            return DummyLEDController(pin)

    def _setup_speaker_relay_controller(self):
        """Factory function to create the speaker relay controller."""
        if not self.settings.get('speaker_relay_enabled', False):
            logger.warning("Speaker relay disabled in settings. Returning None.")
            return None  # No controller needed
        
        pin = self.settings.get('speaker_relay_pin', 27)
        
        # Use the same auto-detection logic as LED - prioritize gpiod for Pi 5
        # Use the same auto-detection logic as LED - prioritize gpiod for Pi 5
        if GPIOD_AVAILABLE:
            logger.warning(f"Auto-detected gpiod. Using GpiodLEDController for speaker relay on pin {pin} (Raspberry Pi 5).")
            return GpiodLEDController(pin)
        elif RPI_GPIO_AVAILABLE:
            logger.warning(f"Auto-detected RPi.GPIO. Using RaspberryPiLEDController for speaker relay on pin {pin}.")
            return RaspberryPiLEDController(pin)
        elif JETSON_GPIO_AVAILABLE:
            logger.warning(f"Auto-detected Jetson.GPIO. Using JetsonLEDController for speaker relay on pin {pin}.")
            return JetsonLEDController(pin)
        else:
            logger.warning("No GPIO library found for speaker relay. Using DummyLEDController.")
            return DummyLEDController(pin)

    def update_settings(self, new_settings):
        """Thread-safe settings update"""
        with self.settings_lock:
            logger.warning(f"Received settings update: {new_settings}")
            
            for key, value in new_settings.items():
                if key in self.settings:
                    old_value = self.settings[key]
                    self.settings[key] = value
                    logger.warning(f"Updated setting {key} from {old_value} to {value}")
            
            # Re-check LED controller if settings changed
            if 'led_controller_type' in new_settings or 'led_pin' in new_settings:
                self.led_controller.cleanup()
                self.led_controller = self._setup_led_controller()
                logger.warning("Re-initialized LED controller due to settings change.")

            # Update all camera processors
            for camera_processor in self.camera_processors:
                camera_processor.settings = self.settings.copy()
                camera_processor.message_manager.update_settings(self.settings)
                
                # Pass the (potentially new) LED controller
                camera_processor.message_manager.led_controller = self.led_controller
                
                logger.warning(f"Updated {camera_processor.camera_name} settings")
    
    def reload_settings(self):
        """Force reload all settings from the default configuration"""
        with self.settings_lock:
            self.settings = Config.DEFAULT_SETTINGS.copy()
            logger.warning(f"Reloaded settings: {self.settings}")
            
            # Re-init hardware
            self.led_controller.cleanup()
            self.led_controller = self._setup_led_controller()
            
            # Update all camera processors
            for camera_processor in self.camera_processors:
                camera_processor.settings = self.settings.copy()
                camera_processor.message_manager.update_settings(self.settings)
                camera_processor.message_manager.led_controller = self.led_controller

    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Render the main dashboard page"""
            return render_template('index.html', cameras=self.camera_names)
        
        # Video Feed Routes
        for i, camera_processor in enumerate(self.camera_processors):
            route_name = f'video_feed_{i}'
            self.app.add_url_rule(
                f'/video_feed/{i}',
                route_name,
                self.create_video_feed_generator(camera_processor)
            )
        
        @self.app.route('/set_roi/<int:camera_index>', methods=['POST'])
        def set_roi(camera_index):
            """Set ROI for traffic light or people detection for a specific camera"""
            if camera_index >= len(self.camera_processors):
                return jsonify({'success': False, 'message': 'Invalid camera index'})
            
            data = request.json
            roi_type = data.get('type')
            points = data.get('points')
            
            camera_processor = self.camera_processors[camera_index]
            success = camera_processor.roi_manager.set_roi(roi_type, points)
            if success:
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'message': 'Invalid ROI type'})

        @self.app.route('/set_zoom/<int:camera_index>', methods=['POST'])
        def set_zoom(camera_index):
            """Set digital zoom level and position for a specific camera"""
            if camera_index >= len(self.camera_processors):
                return jsonify({'success': False, 'message': 'Invalid camera index'})
            
            try:
                data = request.json
                zoom_level = float(data.get('zoom', 1.0))
                center_x = float(data.get('x', 0.5))
                center_y = float(data.get('y', 0.5))
                
                # Validate
                zoom_level = max(1.0, min(zoom_level, 5.0))
                center_x = max(0.0, min(center_x, 1.0))
                center_y = max(0.0, min(center_y, 1.0))
                
                # Update settings
                with self.settings_lock:
                    if 'camera_zoom_configs' not in self.settings:
                        self.settings['camera_zoom_configs'] = {}
                    
                    self.settings['camera_zoom_configs'][str(camera_index)] = {
                        'level': zoom_level,
                        'x': center_x,
                        'y': center_y
                    }
                    
                    # Re-initialize all cameras to apply new resolution
                for cp in self.camera_processors:
                    cp.settings = self.settings.copy()
                    # Signal the capture thread to restart safely
                    cp.restart_requested = True
            
                return jsonify({'success': True, 'zoom': zoom_level, 'x': center_x, 'y': center_y})
                
            except Exception as e:
                logger.error(f"Error setting zoom: {e}")
                return jsonify({'success': False, 'error': str(e)})
            
        @self.app.route('/reset_roi/<int:camera_index>', methods=['POST'])
        def reset_roi(camera_index):
            """Reset ROI for traffic light or people detection for a specific camera"""
            if camera_index >= len(self.camera_processors):
                return jsonify({'success': False, 'message': 'Invalid camera index'})
            
            data = request.json
            roi_type = data.get('type')
            
            camera_processor = self.camera_processors[camera_index]
            success = camera_processor.roi_manager.reset_roi(roi_type)
            if success:
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'message': 'Invalid ROI type'})
        
        @self.app.route('/settings', methods=['GET', 'POST'])
        def handle_settings():
            """Get or update settings"""
            if request.method == 'GET':
                if request.args.get('t'):
                    logger.warning(f"Settings requested with cache-busting parameter")
                return jsonify(self.settings)
            
            elif request.method == 'POST':
                data = request.json
                self.update_settings(data)
                return jsonify({'success': True})
        
        @self.app.route('/upload_audio', methods=['POST'])
        def upload_audio():
            """Handle audio file upload"""
            try:
                if 'audio_file' not in request.files:
                    return jsonify({'success': False, 'error': 'No file provided'})
                
                file = request.files['audio_file']
                audio_type = request.form.get('audio_type')
                
                if audio_type not in ['berhenti', 'melintas']:
                    return jsonify({'success': False, 'error': 'Invalid audio type'})
                
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'})
                
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                
                max_size = 5 * 1024 * 1024  # 5MB
                if file_size > max_size:
                    return jsonify({'success': False, 'error': 'File too large. Maximum size is 5MB'})
                
                # We can now only accept .wav files
                allowed_extensions = {'wav'}
                allowed_mimetypes = {'audio/wav', 'audio/x-wav'}
                
                file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                mime_type = mimetypes.guess_type(file.filename)[0]
                
                if file_ext not in allowed_extensions or mime_type not in allowed_mimetypes:
                    return jsonify({'success': False, 'error': 'Invalid file type. Only .wav files are allowed.'})
                
                os.makedirs(Config.AUDIO_DIR, exist_ok=True)
                
                filename = f"{audio_type}_user.wav"
                filepath = os.path.join(Config.AUDIO_DIR, filename)
                
                file.save(filepath)
                logger.warning(f"Uploaded {audio_type} audio: {filepath}")
                
                # No conversion necessary, we only accept wav
                
                if audio_type == 'berhenti':
                    self.settings['berhenti_audio_filename'] = filename
                else:
                    self.settings['melintas_audio_filename'] = filename
                
                # Update all camera processors
                for camera_processor in self.camera_processors:
                    camera_processor.settings = self.settings.copy()
                    camera_processor.message_manager.update_settings(self.settings)
                
                return jsonify({
                    'success': True,
                    'audio_filename': filename,
                    'message': f'{audio_type} audio uploaded successfully'
                })
                
            except Exception as e:
                logger.error(f"Error uploading audio: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/reset_audio', methods=['POST'])
        def reset_audio():
            """Reset audio to default"""
            try:
                data = request.json
                audio_type = data.get('audio_type')
                
                if audio_type not in ['berhenti', 'melintas']:
                    return jsonify({'success': False, 'error': 'Invalid audio type'})
                
                os.makedirs(Config.AUDIO_DIR, exist_ok=True)
                
                user_filename = f"{audio_type}_user.wav"
                user_filepath = os.path.join(Config.AUDIO_DIR, user_filename)
                if os.path.exists(user_filepath):
                    os.remove(user_filepath)
                    logger.warning(f"Removed user {audio_type} audio file: {user_filepath}")
                
                default_filename = f"{audio_type}_default.wav"
                
                if audio_type == 'berhenti':
                    self.settings['berhenti_audio_filename'] = default_filename
                else:
                    self.settings['melintas_audio_filename'] = default_filename
                
                for camera_processor in self.camera_processors:
                    camera_processor.settings = self.settings.copy()
                    camera_processor.message_manager.update_settings(self.settings)
                
                return jsonify({
                    'success': True,
                    'audio_filename': default_filename,
                    'message': f'{audio_type} audio reset to default'
                })
                
            except Exception as e:
                logger.error(f"Error resetting audio: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/sensor_data', methods=['GET'])
        def sensor_data():
            """Get current sensor readings"""
            try:
                # Read DHT11 sensor
                temperature, humidity = self.dht11_sensor.read()
                
                # Read voltage sensor
                voltage = self.voltage_sensor.read()
                
                return jsonify({
                    'success': True,
                    'temperature': temperature,
                    'humidity': humidity,
                    'voltage': voltage
                })
            except Exception as e:
                logger.error(f"Error reading sensors: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'temperature': None,
                    'humidity': None,
                    'voltage': None
                })
        
        @self.app.route('/test_speaker', methods=['POST'])
        def test_speaker():
            """Test speaker by playing an audio file"""
            try:
                data = request.json
                audio_type = data.get('audio_type', 'melintas')  # 'melintas' or 'berhenti'
                
                if audio_type not in ['berhenti', 'melintas']:
                    return jsonify({'success': False, 'error': 'Invalid audio type'})
                
                # Get the audio filename from settings
                if audio_type == 'berhenti':
                    audio_filename = self.settings.get('berhenti_audio_filename', 'berhenti_default.wav')
                else:
                    audio_filename = self.settings.get('melintas_audio_filename', 'melintas_default.wav')
                
                audio_path = os.path.join(Config.AUDIO_DIR, audio_filename)
                
                if not os.path.exists(audio_path):
                    return jsonify({'success': False, 'error': f'Audio file not found: {audio_filename}'})
                
                # Play the audio using the audio player
                self.audio_player.play_audio(audio_path)
                logger.warning(f"Test speaker: Playing {audio_type} audio")
                
                return jsonify({
                    'success': True,
                    'message': f'Playing {audio_type} audio',
                    'audio_file': audio_filename
                })
                
            except Exception as e:
                logger.error(f"Error testing speaker: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/test_led', methods=['POST'])
        def test_led():
            """Test LED by turning it on/off or blinking"""
            try:
                data = request.json
                action = data.get('action', 'on')  # 'on', 'off', or 'blink'
                
                if action not in ['on', 'off', 'blink']:
                    return jsonify({'success': False, 'error': 'Invalid action'})
                
                if action == 'on':
                    self.led_controller.turn_on()
                    logger.warning("Test LED: Turned ON")
                    message = 'LED turned ON'
                    
                elif action == 'off':
                    self.led_controller.turn_off()
                    logger.warning("Test LED: Turned OFF")
                    message = 'LED turned OFF'
                    
                elif action == 'blink':
                    # Blink for 5 seconds
                    blink_interval = self.settings.get('led_blink_interval', 0.5)
                    self.led_controller.start_blink(blink_interval)
                    logger.warning(f"Test LED: Blinking at {blink_interval}s interval")
                    
                    # Stop blinking after 5 seconds
                    def stop_blink_after_delay():
                        time.sleep(5)
                        self.led_controller.stop_blink()
                        logger.warning("Test LED: Stopped blinking")
                    
                    threading.Thread(target=stop_blink_after_delay, daemon=True).start()
                    message = 'LED blinking for 5 seconds'
                
                return jsonify({
                    'success': True,
                    'message': message,
                    'action': action
                })
                
            except Exception as e:
                logger.error(f"Error testing LED: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/audio/<path:filename>')
        def serve_audio(filename):
            """Serve audio files"""
            try:
                if '..' in filename or filename.startswith('/'):
                    return jsonify({'error': 'Invalid filename'}), 400
                return send_from_directory(Config.AUDIO_DIR, filename)
            except Exception as e:
                logger.error(f"Error serving audio {filename}: {e}")
                return jsonify({'error': str(e)}), 404
        
        @self.app.route('/reload_settings', methods=['POST'])
        def handle_reload_settings():
            """Force reload settings from default configuration"""
            try:
                self.reload_settings()
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Error reloading settings: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/validate_settings', methods=['GET'])
        def handle_validate_settings():
            """Validate that settings are properly applied to all components"""
            try:
                validation_results = {
                    'webapp_settings': self.settings,
                    'camera_processors': []
                }
                
                for camera_processor in self.camera_processors:
                    processor_result = {
                        'camera_name': camera_processor.camera_name,
                        'settings': camera_processor.settings,
                        'message_manager': {
                            'audio_cooldown': camera_processor.message_manager.audio_cooldown,
                            'led_enabled': camera_processor.message_manager.led_enabled,
                            'led_blink_interval': camera_processor.message_manager.led_blink_interval,
                            'berhenti_audio_filename': camera_processor.message_manager.berhenti_audio_filename,
                            'melintas_audio_filename': camera_processor.message_manager.melintas_audio_filename,
                            'berhenti_audio_path': camera_processor.message_manager.berhenti_audio_path,
                            'melintas_audio_path': camera_processor.message_manager.melintas_audio_path
                        }
                    }
                    validation_results['camera_processors'].append(processor_result)
                
                return jsonify(validation_results)
            except Exception as e:
                logger.error(f"Error validating settings: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/toggle_recording/<int:camera_index>', methods=['POST'])
        def toggle_recording(camera_index):
            """Toggle recording for a specific camera"""
            if camera_index >= len(self.camera_processors):
                return jsonify({'success': False, 'message': 'Invalid camera index'})
            
            camera_processor = self.camera_processors[camera_index]
            
            try:
                if camera_processor.video_writer is not None and camera_processor.video_writer.isOpened():
                    camera_processor.stop_recording()
                    return jsonify({
                        'success': True, 
                        'recording_active': False,
                        'message': f"Recording stopped for {camera_processor.camera_name}"
                    })
                else:
                    camera_processor.start_recording()
                    return jsonify({
                        'success': True, 
                        'recording_active': True,
                        'message': f"Recording started for {camera_processor.camera_name}"
                    })
            except Exception as e:
                logger.error(f"Error toggling recording for camera {camera_index}: {e}")
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/recording_status/<int:camera_index>', methods=['GET'])
        def recording_status(camera_index):
            """Get recording status for a specific camera"""
            if camera_index >= len(self.camera_processors):
                return jsonify({'success': False, 'message': 'Invalid camera index'})
            
            camera_processor = self.camera_processors[camera_index]
            
            recording_status = {
                'active': camera_processor.video_writer is not None and camera_processor.video_writer.isOpened(),
                'current_recording': os.path.basename(camera_processor.current_recording_path) if camera_processor.current_recording_path else None,
                'recording_start_time': camera_processor.recording_start_time,
                'recording_frame_count': camera_processor.recording_frame_count,
                'recording_segment_count': camera_processor.recording_segment_count
            }
            
            return jsonify({
                'success': True,
                'camera_name': camera_processor.camera_name,
                'recording_status': recording_status
            })
        
        @self.app.route('/all_recording_status', methods=['GET'])
        def all_recording_status():
            """Get recording status for all cameras"""
            recording_status = []
            
            for camera_processor in self.camera_processors:
                status = {
                    'camera_name': camera_processor.camera_name,
                    'camera_index': camera_processor.camera_index,
                    'active': camera_processor.video_writer is not None and camera_processor.video_writer.isOpened(),
                    'current_recording': os.path.basename(camera_processor.current_recording_path) if camera_processor.current_recording_path else None,
                    'recording_start_time': camera_processor.recording_start_time,
                    'recording_frame_count': camera_processor.recording_frame_count,
                    'recording_segment_count': camera_processor.recording_segment_count
                }
                recording_status.append(status)
            
            return jsonify({'recording_status': recording_status})
        
        @self.app.route('/status')
        def status():
            """Get system status with recording validation"""
            cameras_status = []
            for i, camera_processor in enumerate(self.camera_processors):
                roi_status = camera_processor.roi_manager.get_roi_status()
                
                recording_status = {
                    'active': camera_processor.video_writer is not None and camera_processor.video_writer.isOpened(),
                    'current_recording': os.path.basename(camera_processor.current_recording_path) if camera_processor.current_recording_path else None,
                    'validated': False
                }
                
                # Get state from global detection state for this camera's linked cameras
                state = self.global_state.get_state(camera_processor.message_manager.linked_cameras)
                
                cameras_status.append({
                    'camera_name': camera_processor.camera_name,
                    'camera_index': camera_processor.camera_index,
                    'traffic_light': state['traffic_light_state'] if state['traffic_light_state'] != 'unknown' else 'Not detected',
                    'person_in_roi': state['person_detected'],
                    'device': f"{self.model_manager.device.type.upper()}{' - ' + torch.cuda.get_device_name(0) if self.model_manager.device.type == 'cuda' else ''}",
                    'tl_roi_defined': roi_status['tl_roi_defined'],
                    'people_roi_defined': roi_status['people_roi_defined'],
                    'recording': recording_status,
                    'is_initialized': camera_processor.is_initialized,
                    'fps': camera_processor.current_fps,
                    'berhenti_audio_filename': camera_processor.message_manager.berhenti_audio_filename,
                    'melintas_audio_filename': camera_processor.message_manager.melintas_audio_filename
                })
            
            return jsonify({'cameras': cameras_status})
        
        @self.app.route('/recordings')
        def list_recordings():
            """List all available recordings from all cameras"""
            try:
                recordings = []
                for camera_processor in self.camera_processors:
                    camera_name = camera_processor.camera_name
                    camera_dir = camera_processor.camera_recordings_dir
                    
                    if os.path.exists(camera_dir):
                        daily_folders = [f for f in os.listdir(camera_dir) 
                                       if os.path.isdir(os.path.join(camera_dir, f))]
                        
                        for folder in daily_folders:
                            folder_path = os.path.join(camera_dir, folder)
                            
                            for filename in os.listdir(folder_path):
                                if filename.endswith(('.mp4', '.avi')): # Support both mp4 and avi
                                    filepath = os.path.join(folder_path, filename)
                                    stat = os.stat(filepath)
                                    
                                    duration = self._get_video_duration(filepath)
                                    
                                    is_active = filepath == camera_processor.current_recording_path
                                    
                                    is_valid = False
                                    if self.settings.get('validate_recordings', True) and not is_active:
                                        is_valid = camera_processor.validate_recording(filepath)
                                    
                                    recordings.append({
                                        'filename': f"{folder}/{filename}",
                                        'date_folder': folder,
                                        'camera': camera_name,
                                        'size': stat.st_size,
                                        'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                                        'duration': duration,
                                        'is_active': is_active,
                                        'is_valid': is_valid,
                                        'filetype': 'MP4' if filename.endswith('.mp4') else 'AVI'
                                    })
                
                recordings.sort(key=lambda x: x['created'], reverse=True)
                
                return jsonify({'recordings': recordings})
            except Exception as e:
                logger.error(f"Error listing recordings: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/recordings/by_camera')
        def list_recordings_by_camera():
            """List recordings grouped by camera, each with date-folder and filename."""
            try:
                result = {}
                for camera_processor in self.camera_processors:
                    camera_name = camera_processor.camera_name
                    camera_dir = camera_processor.camera_recordings_dir
                    camera_list = []
                    if os.path.exists(camera_dir):
                        daily_folders = [f for f in os.listdir(camera_dir) 
                                       if os.path.isdir(os.path.join(camera_dir, f))]
                        for folder in daily_folders:
                            folder_path = os.path.join(camera_dir, folder)
                            for filename in os.listdir(folder_path):
                                if filename.endswith(('.mp4', '.avi')):
                                    filepath = os.path.join(folder_path, filename)
                                    
                                    # --- NEW LOGIC HERE ---
                                    # Check if this file is the one currently being written to
                                    is_active = (filepath == camera_processor.current_recording_path)
                                    # --- END NEW LOGIC ---

                                    try:
                                        stat = os.stat(filepath)
                                        created_time = datetime.fromtimestamp(stat.st_mtime)
                                        size_bytes = stat.st_size
                                    except Exception:
                                        created_time = datetime.now()
                                        size_bytes = 0
                                    
                                    camera_list.append({
                                        'filename': f"{folder}/{filename}",
                                        'date_folder': folder,
                                        'camera': camera_name,
                                        'created': created_time.strftime('%Y-%m-%d %H:%M:%S'),
                                        'size': size_bytes,
                                        'is_active': is_active  # <-- ADDED THIS KEY
                                    })
                    camera_list.sort(key=lambda x: x['created'], reverse=True)
                    result[camera_name] = camera_list
                return jsonify({'by_camera': result})
            except Exception as e:
                logger.error(f"Error listing recordings by camera: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/recordings/<path:filename>')
        def serve_recording(filename):
            """Serve a recording file with proper headers for video streaming"""
            try:
                if '..' in filename or filename.startswith('/'):
                    return jsonify({'error': 'Invalid filename'}), 400
                
                parts = filename.split('/')
                if len(parts) == 2:
                    date_folder, actual_filename = parts
                else:
                    date_folder = None
                    actual_filename = filename
                
                for camera_processor in self.camera_processors:
                    if date_folder:
                        filepath = os.path.join(camera_processor.camera_recordings_dir, date_folder, actual_filename)
                    else:
                        daily_folders = [f for f in os.listdir(camera_processor.camera_recordings_dir) 
                                       if os.path.isdir(os.path.join(camera_processor.camera_recordings_dir, f))]
                        for folder in daily_folders:
                            test_path = os.path.join(camera_processor.camera_recordings_dir, folder, actual_filename)
                            if os.path.exists(test_path):
                                filepath = test_path
                                break
                        else:
                            continue
                    
                    if os.path.exists(filepath):
                        if filepath == camera_processor.current_recording_path:
                            return jsonify({'error': 'Cannot stream active recording'}), 400
                        
                        return self._stream_video_file(filepath)
                
                return jsonify({'error': 'File not found'}), 404
            except Exception as e:
                logger.error(f"Error serving recording {filename}: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/toggle_low_power', methods=['POST'])
        def toggle_low_power():
            """Toggle Low Power Mode (320x240 @ 5FPS vs 640x480 @ 10FPS)"""
            try:
                data = request.json
                enabled = data.get('enabled', False)
                
                with self.settings_lock:
                    if enabled:
                        self.settings['processing_width'] = 320
                        self.settings['processing_height'] = 240
                        self.settings['target_fps'] = 5
                        logger.warning("Low Power Mode ENABLED: Switching to 320x240 @ 5FPS")
                    else:
                        self.settings['processing_width'] = 640
                        self.settings['processing_height'] = 480
                        self.settings['target_fps'] = 10
                        logger.warning("Low Power Mode DISABLED: Switching to 640x480 @ 10FPS")
                    
                    # Re-initialize all cameras to apply new resolution
                    for cp in self.camera_processors:
                        cp.settings = self.settings.copy()
                        # Signal the capture thread to restart safely
                        cp.restart_requested = True
                
                return jsonify({'success': True, 'enabled': enabled})
                
            except Exception as e:
                logger.error(f"Error toggling low power mode: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/recordings/<path:filename>', methods=['DELETE'])
        def delete_recording(filename):
            """Delete a recording file"""
            try:
                if '..' in filename or filename.startswith('/'):
                    return jsonify({'error': 'Invalid filename'}), 400
                
                parts = filename.split('/')
                if len(parts) == 2:
                    date_folder, actual_filename = parts
                else:
                    date_folder = None
                    actual_filename = filename
                
                for camera_processor in self.camera_processors:
                    if date_folder:
                        filepath = os.path.join(camera_processor.camera_recordings_dir, date_folder, actual_filename)
                    else:
                        # Find the file in any daily folder for that camera
                        daily_folders = [f for f in os.listdir(camera_processor.camera_recordings_dir) 
                                       if os.path.isdir(os.path.join(camera_processor.camera_recordings_dir, f))]
                        for folder in daily_folders:
                            test_path = os.path.join(camera_processor.camera_recordings_dir, folder, actual_filename)
                            if os.path.exists(test_path):
                                filepath = test_path
                                break
                        else:
                            continue
                    
                    if os.path.exists(filepath):
                        if filepath == camera_processor.current_recording_path:
                            return jsonify({'error': 'Cannot delete active recording'}), 400
                        
                        os.remove(filepath)
                        logger.warning(f"Deleted recording: {filename}")
                        
                        # Clean up empty daily folder
                        folder_path = os.path.dirname(filepath)
                        if not os.listdir(folder_path):
                            os.rmdir(folder_path)
                            logger.warning(f"Removed empty folder: {folder_path}")
                        
                        return jsonify({'success': True})
                
                return jsonify({'error': 'File not found'}), 404
            except Exception as e:
                logger.error(f"Error deleting recording {filename}: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _get_video_duration(self, filepath):
        """Get video duration in seconds"""
        try:
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                return duration
        except Exception as e:
            logger.error(f"Error getting video duration for {filepath}: {e}")
        return 0
    
    def _stream_video_file(self, filepath):
        """Stream video file with range request support"""
        file_size = os.path.getsize(filepath)
        
        range_header = request.headers.get('Range', None)
        if range_header:
            byte1, byte2 = 0, None
            match = re.search(r'bytes=(\d+)-(\d*)', range_header)
            groups = match.groups()
            
            if groups[0]:
                byte1 = int(groups[0])
            if groups[1]:
                byte2 = int(groups[1])
            
            if byte2 is None:
                byte2 = file_size - 1
            
            length = byte2 - byte1 + 1
            
            with open(filepath, 'rb') as f:
                f.seek(byte1)
                data = f.read(length)
            
            mimetype = f'video/{filepath.split(".")[-1]}'
            resp = Response(
                data,
                206,
                mimetype=mimetype,
                direct_passthrough=True)
            resp.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
            resp.headers.add('Accept-Ranges', 'bytes')
            return resp
        else:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            mimetype = f'video/{filepath.split(".")[-1]}'
            return Response(
                data,
                200,
                mimetype=mimetype,
                direct_passthrough=True)
    
    def create_video_feed_generator(self, camera_processor):
        """Create a video feed generator for a specific camera"""
        def video_feed():
            return Response(camera_processor.generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        return video_feed
        
    def run(self):
        """Run the Flask application on port 8080."""
        port = 8080  # Set port directly
        
        logger.warning(f"Server starting on http://localhost:{port}")
        
        # Let Flask handle port errors.
        # The 'finally' block is removed.
        # Shutdown is now handled ONLY by the KeyboardInterrupt in __main__.
        self.app.run(host='0.0.0.0', port=port, threaded=True)
    
    def shutdown(self):
        """Gracefully shut down all components."""
        logger.warning("Shutting down all components...")
        # Clean up GPIO
        if hasattr(self, 'led_controller'):
            self.led_controller.cleanup()
        if hasattr(self, 'speaker_relay_controller'):
            self.speaker_relay_controller.cleanup()
        # Stop all camera processors
        if hasattr(self, 'camera_processors'):
            for camera_processor in self.camera_processors:
                camera_processor.stop()
        logger.warning("Cleanup complete. Exiting.")

if __name__ == '__main__':
    web_app = WebApp()
    try:
        web_app.run()
    except KeyboardInterrupt:
        logger.warning("Ctrl+C detected. Initiating graceful shutdown...")
        # We manually call shutdown to ensure everything stops
        # before the script exits.
        web_app.shutdown()
        sys.exit(0)