# Airtone 🎶

**Play music with your hands and face — no instrument, no touch, just your webcam.**

Airtone turns your body into a real-time synthesizer. Using computer vision, it tracks your hand gestures and mouth movements through your webcam and maps them to sound — glide your hands through the air to change pitch, open your fingers to build chords, and open your mouth to add a third voice. Everything is synthesized live, with no pre-recorded samples.

![Airtone in action](docs/screenshot-main.png)
<!-- TODO: Replace with a screenshot of the app running — hands + face tracked, tuner visible on the right -->

---

## Demo

![Airtone demo GIF](docs/demo.gif)
<!-- TODO: Replace with a short GIF of you playing a chord progression with your hands + mouth -->

---

## Features

- **Three independent voices** — left hand, right hand, and mouth each control their own synth voice, so you can layer up to three parts at once.
- **Continuous pitch gliding** — vertical position maps to a C Major scale spanning ~3 octaves (C3–C6), with logarithmic interpolation for smooth, theremin-like slides between notes.
- **Gesture-based chords** — the number of extended fingers builds a chord one interval at a time: root → 5th → major 3rd → major 7th → major 9th.
- **Pinch to solo** — pinch thumb and index together to collapse back to a single clean note.
- **Mouth as an instrument** — mouth openness controls both volume and chord density; open wider to get louder and add more notes.
- **Real-time additive synthesis** — audio is generated sample-by-sample from summed sine oscillators, with amplitude smoothing to avoid clicks and soft-clipping (`tanh`) to prevent distortion.
- **On-screen tuner** — a live tuner on the right edge shows each active voice's pitch and how close it is to the nearest scale note.

---

## How It Works

Airtone runs a real-time loop that ties together computer vision and audio synthesis:

### 1. Vision — tracking the body
Each webcam frame is passed to two Google [MediaPipe](https://ai.google.dev/edge/mediapipe) models:

- **Hand Landmarker** (`hand_landmarker.task`) — detects up to 2 hands, 21 landmarks each.
- **Face Landmarker** (`face_landmarker.task`) — detects facial landmarks for a single face.

### 2. Gesture interpretation
Raw landmarks are turned into musical intent:

| Input | What's measured | Musical effect |
|-------|-----------------|----------------|
| Hand height | Y-position of the palm center | Pitch (higher hand = higher note) |
| Extended fingers | Fingertip vs. knuckle distance from wrist | Chord size (1–5 notes) |
| Pinch | Thumb-to-index distance | Force a single note |
| Fist | All fingers curled | Mute that voice |
| Mouth openness | Upper-lip to lower-lip distance | Volume + chord density |
| Head height | Y-position of the nose | Mouth voice pitch |

### 3. Chord construction
As fingers open, intervals stack on top of the root note to build progressively richer chords:

```
1 finger  → Root                (single note)
2 fingers → Root + P5           (power chord)
3 fingers → Root + P5 + M3      (major triad)
4 fingers → + M7                (major 7th)
5 fingers → + M9                (major 9th)
```

### 4. Audio synthesis
A `sounddevice` output stream runs an additive synth. Each of the 3 voices has 5 sine oscillators. On every audio block, amplitudes are smoothly interpolated toward their targets (no clicks), phases are tracked continuously (no discontinuities), all voices are summed, and the mix is soft-clipped through `tanh`.

![Architecture / controls diagram](docs/diagram.png)
<!-- TODO: Optional — replace with a diagram of the vision → gesture → synth pipeline, or a labeled screenshot of the controls -->

---

## Controls

| Action | Result |
|--------|--------|
| Move hands up/down | Glide pitch |
| Pinch thumb + index | Single note |
| Open fingers | Build up to a major 9th chord |
| Make a fist | Mute that hand |
| Open your mouth | Play the third voice (wider = louder + more notes) |
| Move your head up/down | Change mouth-voice pitch |
| Press `q` | Quit |

---

## Getting Started

### Prerequisites
- Python 3.9+
- A webcam
- macOS, Windows, or Linux

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/airtone.git
cd airtone

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe numpy sounddevice
```

The MediaPipe model files (`hand_landmarker.task` and `face_landmarker.task`) are included in the repo.

### Run

```bash
python main.py
```

A window opens showing your webcam feed with tracking overlays. Start moving your hands and mouth to make sound. Press `q` to quit.

> **Note:** Wear headphones if your mic-adjacent speakers cause feedback, and make sure your OS grants camera + audio permissions to your terminal.

---

## Tech Stack

- **[OpenCV](https://opencv.org/)** — webcam capture, image processing, and on-screen UI/overlays
- **[MediaPipe](https://ai.google.dev/edge/mediapipe)** — hand and face landmark detection
- **[NumPy](https://numpy.org/)** — vectorized audio buffer math
- **[sounddevice](https://python-sounddevice.readthedocs.io/)** — low-latency real-time audio output

---

## Screenshots

<!-- TODO: Add a few more shots showing different states -->

| Playing a chord with open fingers | Mouth-controlled voice | The tuner in action |
|-----------------------------------|------------------------|---------------------|
| ![](docs/screenshot-chord.png)    | ![](docs/screenshot-mouth.png) | ![](docs/screenshot-tuner.png) |
<!-- TODO: Replace the three images above -->

---

## Possible Improvements

- Selectable musical scales / keys (minor, pentatonic, chromatic)
- Different oscillator waveforms and simple effects (reverb, delay)
- MIDI output to drive external synths and DAWs
- Recording and looping

---

## License

MIT — feel free to use, modify, and build on it.
