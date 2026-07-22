import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Audio parameters
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
CHANNELS = 1

# Musical scale (C Major diatonic) extended
base_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88] # C4 to B4
octave_names = ["C", "D", "E", "F", "G", "A", "B"]

SCALE_FREQS = []
NOTES_NAMES = []
# Create 3 octaves (C3 to B5)
for oct_offset in [-1, 0, 1]:
    for i, freq in enumerate(base_freqs):
        SCALE_FREQS.append(freq * (2 ** oct_offset))
        NOTES_NAMES.append(f"{octave_names[i]}{4 + oct_offset}")
# Add C6
SCALE_FREQS.append(base_freqs[0] * 4)
NOTES_NAMES.append("C6")

# Connections for drawing the hand
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        p1 = landmarks[start_idx]
        p2 = landmarks[end_idx]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # Draw points
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

class AudioSynth:
    def __init__(self):
        # 3 voices (2 hands, 1 face), 5 oscillators each (root, 5th, 3rd, 7th, 9th)
        self.voices = [
            {'freqs': [0.0]*5, 'target_amps': [0.0]*5, 'current_amps': [0.0]*5, 'phases': [0.0]*5} 
            for _ in range(3)
        ]
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.callback, blocksize=BLOCK_SIZE
        )
        self.stream.start()

    def callback(self, outdata, frames, time, status):
        t = np.arange(frames) / SAMPLE_RATE
        audio_chunk = np.zeros(frames)
        
        for voice in self.voices:
            for i in range(5):
                freq = voice['freqs'][i]
                target_amp = voice['target_amps'][i]
                current_amp = voice['current_amps'][i]
                
                if target_amp < 0.001 and current_amp < 0.001:
                    voice['current_amps'][i] = target_amp
                    continue
                    
                # Smoothly interpolate amplitude over the chunk to avoid clicks
                amp_envelope = np.linspace(current_amp, target_amp, frames)
                voice['current_amps'][i] = target_amp 
                
                phase = voice['phases'][i]
                wave = amp_envelope * np.sin(2 * np.pi * freq * t + phase)
                audio_chunk += wave
                
                # Update phase
                voice['phases'][i] = (phase + 2 * np.pi * freq * frames / SAMPLE_RATE) % (2 * np.pi)
                    
        # Soft clipping to prevent distortion
        audio_chunk = np.tanh(audio_chunk)
            
        outdata[:, 0] = audio_chunk

    def update_voice(self, voice_idx, active, base_freq=440.0, num_notes=1, volume=1.0):
        if not active:
            self.voices[voice_idx]['target_amps'] = [0.0] * 5
            return
            
        amp_per_osc = 0.08 * volume # Lower volume to prevent clipping with many oscillators
        
        # Intervals to add as fingers open up:
        # 1: Root (0)
        # 2: P5 (7)
        # 3: M3 (4) -> makes a Major Triad
        # 4: M7 (11) -> makes a Major 7th chord
        # 5: M9 (14) -> makes a Major 9th chord
        intervals_semitones = [0, 7, 4, 11, 14]
        
        notes_to_play = min(max(1, num_notes), 5)
        
        for i in range(5):
            if i < notes_to_play:
                semitone_offset = intervals_semitones[i]
                freq = base_freq * (2 ** (semitone_offset / 12.0))
                self.voices[voice_idx]['freqs'][i] = freq
                self.voices[voice_idx]['target_amps'][i] = amp_per_osc
            else:
                self.voices[voice_idx]['target_amps'][i] = 0.0

    def stop(self):
        self.stream.stop()
        self.stream.close()

def get_hand_state(hand_landmarks):
    wrist = hand_landmarks[0]
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    main_extended = 0
    for tip_idx, pip_idx in zip(tips, pips):
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        dist_tip = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
        dist_pip = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
        if dist_tip > dist_pip:
            main_extended += 1
            
    # Check thumb
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_extended = 1 if ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2) > ((thumb_ip.x - wrist.x)**2 + (thumb_ip.y - wrist.y)**2) else 0
        
    return main_extended, thumb_extended

def is_pinch(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    # Simple Euclidean distance
    dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2 + (thumb_tip.z - index_tip.z)**2)
    return dist < 0.05

def get_continuous_freq(c_idx):
    c_idx = max(0.0, min(len(SCALE_FREQS) - 1, c_idx))
    idx1 = int(math.floor(c_idx))
    idx2 = min(idx1 + 1, len(SCALE_FREQS) - 1)
    fraction = c_idx - idx1
    if idx1 == idx2:
        return SCALE_FREQS[idx1]
    # Logarithmic interpolation for pitch
    return SCALE_FREQS[idx1] * ((SCALE_FREQS[idx2] / SCALE_FREQS[idx1]) ** fraction)

def main():
    synth = AudioSynth()
    
    # Initialize HandLandmarker
    base_options_hand = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand,
                                           num_hands=2)
    detector_hand = vision.HandLandmarker.create_from_options(options_hand)
    
    # Initialize FaceLandmarker
    base_options_face = python.BaseOptions(model_asset_path='face_landmarker.task')
    options_face = vision.FaceLandmarkerOptions(base_options=base_options_face,
                                                num_faces=1)
    detector_face = vision.FaceLandmarker.create_from_options(options_face)
    
    cap = cv2.VideoCapture(0)
    print("Starting Music Hands & Face app...")
    print("Press 'q' in the video window to quit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        results_hand = detector_hand.detect(mp_image)
        results_face = detector_face.detect(mp_image)
        
        voice_active = [False, False, False]
        voice_params = [{'base_freq': 0.0, 'num_notes': 1, 'c_idx': 0.0, 'volume': 1.0} for _ in range(3)]
        
        # --- Process Hands ---
        if results_hand.hand_landmarks:
            for idx, hand_landmarks in enumerate(results_hand.hand_landmarks):
                if idx >= 2:
                    break
                    
                draw_landmarks(image, hand_landmarks)
                
                palm_y = hand_landmarks[9].y
                palm_y = max(0.1, min(0.9, palm_y))
                normalized_y = (palm_y - 0.1) / 0.8
                inverted_y = 1.0 - normalized_y
                
                # Continuous note index
                c_idx = inverted_y * (len(SCALE_FREQS) - 1)
                base_freq = get_continuous_freq(c_idx)
                
                pinch = is_pinch(hand_landmarks)
                main_extended, thumb_extended = get_hand_state(hand_landmarks)
                fingers = main_extended + thumb_extended
                
                # If the 4 main fingers are curled, it's a fist. Ignore thumb/pinch noise.
                if main_extended == 0:
                    voice_active[idx] = False
                    num_notes = 0
                else:
                    voice_active[idx] = True
                    if pinch:
                        num_notes = 1
                    else:
                        num_notes = max(1, fingers)
                
                voice_params[idx]['base_freq'] = base_freq
                voice_params[idx]['num_notes'] = num_notes
                voice_params[idx]['c_idx'] = c_idx
                voice_params[idx]['volume'] = 1.0
                
                # Draw text near the hand
                closest_idx = int(round(c_idx))
                closest_idx = max(0, min(len(SCALE_FREQS) - 1, closest_idx))
                note_name = NOTES_NAMES[closest_idx]
                
                if num_notes > 1:
                    note_name += f" ({num_notes} notes)"
                    
                cx, cy = int(hand_landmarks[9].x * w), int(hand_landmarks[9].y * h)
                cv2.putText(image, note_name, (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # --- Process Face (Mouth) ---
        if results_face.face_landmarks:
            face_landmarks = results_face.face_landmarks[0]
            
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            nose = face_landmarks[1]
            
            # Draw mouth markers
            cv2.circle(image, (int(upper_lip.x * w), int(upper_lip.y * h)), 4, (255, 0, 255), -1)
            cv2.circle(image, (int(lower_lip.x * w), int(lower_lip.y * h)), 4, (255, 0, 255), -1)
            
            # Compute openness of the mouth based on upper and lower lip distance
            openness = math.sqrt((upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2)
            
            # Clamp and normalize openness (0.015 threshold to ignore normal closed mouth)
            clamped_open = max(0.0, min(0.08, openness - 0.015))
            volume = clamped_open / 0.065 # Scales from 0.0 to 1.0
            
            if volume > 0.05:
                # Use nose Y-position for pitch
                face_y = nose.y
                face_y = max(0.1, min(0.9, face_y))
                normalized_y = (face_y - 0.1) / 0.8
                inverted_y = 1.0 - normalized_y
                
                c_idx = inverted_y * (len(SCALE_FREQS) - 1)
                base_freq = get_continuous_freq(c_idx)
                
                # As mouth opens wider, add more notes to the chord
                if volume > 0.8:
                    num_notes = 5
                elif volume > 0.5:
                    num_notes = 3
                else:
                    num_notes = 1
                
                voice_active[2] = True
                voice_params[2]['base_freq'] = base_freq
                voice_params[2]['num_notes'] = num_notes
                voice_params[2]['c_idx'] = c_idx
                voice_params[2]['volume'] = volume
                
                closest_idx = int(round(c_idx))
                closest_idx = max(0, min(len(SCALE_FREQS) - 1, closest_idx))
                note_name = NOTES_NAMES[closest_idx]
                
                cv2.putText(image, f"Mouth: {note_name} ({num_notes} notes)", (int(nose.x * w), int(nose.y * h) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2, cv2.LINE_AA)
        
        # --- Update Synthesizer ---
        for i in range(3):
            synth.update_voice(i, voice_active[i], voice_params[i]['base_freq'], voice_params[i]['num_notes'], voice_params[i]['volume'])
        
        # --- Tuner UI on the right side ---
        tuner_x = w - 80
        tuner_center_y = h // 2
        tuner_height = h // 2
        
        # Draw tuner background
        cv2.line(image, (tuner_x, tuner_center_y - tuner_height//2), (tuner_x, tuner_center_y + tuner_height//2), (200, 200, 200), 2)
        cv2.line(image, (tuner_x - 15, tuner_center_y), (tuner_x + 15, tuner_center_y), (0, 255, 0), 2)
        cv2.putText(image, "Tuner", (tuner_x - 20, tuner_center_y - tuner_height//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for idx, active in enumerate(voice_active):
            if active:
                c_idx = voice_params[idx]['c_idx']
                closest_idx = int(round(c_idx))
                closest_idx = max(0, min(len(SCALE_FREQS) - 1, closest_idx))
                
                offset = c_idx - closest_idx # -0.5 to 0.5
                marker_y = tuner_center_y - int(offset * tuner_height)
                
                if idx == 0: color = (0, 255, 255) # Yellow for Hand 1
                elif idx == 1: color = (255, 0, 255) # Purple for Hand 2
                else: color = (100, 100, 255) # Light Red for Face (BGR)
                
                cv2.circle(image, (tuner_x, marker_y), 8, color, -1)
                
                # Show note name next to the marker
                note_name = NOTES_NAMES[closest_idx]
                text_x_offset = 45 if idx < 2 else 90 # Stagger the face text so it doesn't overlap hand text
                cv2.putText(image, note_name, (tuner_x - text_x_offset, marker_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw instructions on screen
        cv2.putText(image, "Music Hands & Face Control", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "- Move hands/head to glide pitch", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, "- Pinch fingers: Single note", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, "- Open fingers: Build Major 9th chord", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, "- Open mouth: Play 3rd voice (wider = louder & chords)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
        cv2.putText(image, "Press 'q' to quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        cv2.imshow('Music Hands', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    synth.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
