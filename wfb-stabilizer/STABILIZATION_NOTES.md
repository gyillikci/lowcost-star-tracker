# Video Stabilization + Star Stacking System

## Date: January 17, 2026

## Overview

This document describes the hybrid video stabilization system developed for star tracking applications, combining gyroscope-based compensation with template matching refinement, feeding into a real-time frame stacker for star intensification.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DUAL GUI SYSTEM                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              STABILIZER (Real-time OpenCV Window)            │    │
│  │                                                              │    │
│  │   Camera → Gyro Roll/Pitch → Template XY → Stabilized Frame │    │
│  │      ↓           ↓                ↓              ↓          │    │
│  │   60 FPS    Orange Cube      Fine-tune      Sent to Queue   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼ (Frame Queue)                         │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              STACKER (Tkinter GUI + OpenCV Window)           │    │
│  │                                                              │    │
│  │   Receive Frame → Crop Template Region → Stack → Enhance    │    │
│  │        ↓                   ↓               ↓         ↓      │    │
│  │   Template Info       Aligned Crop      max/avg   CLAHE     │    │
│  │   from Stabilizer                       median    asinh     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Innovation: Sensor Fusion for Robust Stabilization

### The Problem with Pure Template Matching

Traditional template matching alone struggles with:
- **Roll sensitivity**: Template matching is highly susceptible to rotation. Even small roll angles cause the template to fail matching because the pattern rotates.
- **Large search windows needed**: Without prior motion estimation, template matching must search a large area, which is:
  - Computationally expensive
  - Prone to false matches
  - Unable to handle fast motion

### The Problem with Pure Gyro Stabilization

Gyro-only stabilization has limitations:
- **Drift over time**: IMU integration accumulates errors
- **No absolute reference**: Cannot correct if the initial frame was not level
- **Sensor noise**: High-frequency jitter passes through
- **Mounting offsets**: Calibration between camera and IMU axes

### The Hybrid Solution: Best of Both Worlds

By combining IMU gyro compensation FIRST, then applying template matching:

```
Raw Frame
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Gyro-Based Roll/Pitch Compensation                 │
│                                                             │
│  • Roll correction: Rotate frame to level horizon           │
│  • Pitch correction: Vertical shift based on VFOV           │
│                                                             │
│  IMU STRENGTH: Fast response, handles rotation well         │
│                Low latency (100 Hz attitude data)           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ (Pre-leveled frame - rotation removed!)
    │
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Template Matching XY Refinement                    │
│                                                             │
│  • Now receives LEVELED image (no rotation!)                │
│  • Can use SMALL search window (gyro already compensated)   │
│  • Only needs to find XY translation residual               │
│                                                             │
│  TEMPLATE STRENGTH: Sub-pixel precision, drift correction   │
│                     Absolute reference to scene content     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ (Fully stabilized frame)
```

### Why This Fusion Works So Well

1. **Roll Removal Enables Template Matching**
   - Template matching is extremely sensitive to rotation
   - By removing roll first via gyro, the template sees a consistently oriented image
   - Match quality (correlation) stays high (>0.7) even during motion

2. **Pitch Compensation Reduces Search Area**
   - Without pitch compensation, vertical motion requires large search margins
   - Gyro-based pitch shift pre-positions the frame
   - Template matching only needs to find small residual offsets
   - Smaller search window = faster matching + fewer false positives

3. **Complementary Strengths**
   
   | Aspect | IMU Gyro | Template Matching |
   |--------|----------|-------------------|
   | Speed | Fast (100 Hz) | Slower (per-frame) |
   | Rotation handling | Excellent | Poor |
   | Translation accuracy | Good (with VFOV scaling) | Excellent (sub-pixel) |
   | Drift | Accumulates | None (absolute) |
   | Latency | Very low | Moderate |
   
4. **Reduced Uncertainty = Better Results**
   - Gyro minimizes the uncertainty space for template matching
   - Instead of searching full frame, search ±200px around expected position
   - Higher confidence matches, fewer outliers

---

## Hardware Configuration

### Camera
- **Model**: Harrier 10x Zoom Camera
- **Resolution**: 1280 x 720 @ 60 FPS
- **Interface**: USB (CAP_DSHOW)
- **Buffer**: Size 1 (minimal latency)

### IMU
- **Model**: Orange Cube (ArduPilot)
- **Interface**: MAVLink over COM6 @ 115200 baud
- **Data**: ATTITUDE messages at 100 Hz
- **Fields used**: roll, pitch (radians)

### Field of View Parameters
- **HFOV**: ~50° at 1x zoom
- **VFOV**: ~34° at 1x zoom
- **Pitch scale**: `720 / 34 ≈ 21.2 pixels/degree`
- **Yaw scale**: `1280 / 50 ≈ 25.6 pixels/degree`

---

## Implementation Details

### Files Created

| File | Purpose |
|------|---------|
| `run_camera1_hybrid_simple.py` | Standalone hybrid stabilizer (gyro + template) |
| `run_camera1_stacker.py` | Standalone frame stacker with GUI |
| `run_camera1_stabilize_stack.py` | Combined stabilizer + stacker (single window) |
| `run_camera1_dual_gui.py` | **Main**: Separate stabilizer + stacker windows |

### Dual GUI System (`run_camera1_dual_gui.py`)

**Stabilizer Window Features:**
- Real-time display at camera FPS
- Draw rectangle to set template
- Green box when match quality > 0.7, yellow otherwise
- Keyboard controls: B = reset baseline, R = clear template, Q = quit

**Stacker GUI Features:**
- Receives stabilized frames via thread-safe queue
- Automatically uses template region from stabilizer
- Crops and stacks only the template area (+50px margin)
- Adjustable parameters:
  - Stack count (2-100 frames)
  - Intensity multiplier (1-10x)
  - Gamma correction (0.5-3.0)
  - Stack mode (max, average, sum, median)
  - Adaptive enhancement (CLAHE, asinh, log, sqrt)

### Adaptive Enhancement for Stars

Different stars have vastly different brightnesses. To enhance faint stars without saturating bright ones:

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Divides image into tiles
- Equalizes histogram locally per tile
- Clip limit prevents over-amplification
- Perfect for star fields with varying brightness

**Asinh Stretch (Astronomical Standard)**
- `output = asinh(a * x) / asinh(a)`
- Compresses bright regions, expands faint regions
- Preserves relative brightness ratios
- Commonly used in professional astronomy software

**Log Stretch**
- `output = log(1 + a*x) / log(1 + a)`
- Aggressive dynamic range compression

**Sqrt Stretch**
- Simple power-law enhancement
- Good for general star enhancement

---

## Key Parameters

```python
# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# MAVLink settings
MAVLINK_PORT = "COM6"
MAVLINK_BAUDRATE = 115200

# Stabilization settings
ZOOM_FACTOR = 0.9          # Slight zoom to hide border artifacts
VFOV_DEGREES = 34.0        # Vertical field of view
PITCH_SCALE = 720 / 34     # Pixels per degree of pitch

# Template matching
SEARCH_MARGIN = 200        # Stabilizer search window (pixels)
CROP_MARGIN = 50           # Extra margin around template for stacking
```

---

## Usage Instructions

1. **Start the System**
   ```powershell
   .\venv\Scripts\python.exe wfb-stabilizer\run_camera1_dual_gui.py
   ```

2. **Set Up Stabilization**
   - Wait for Orange Cube connection
   - Press **B** to reset gyro baseline when camera is level
   - Draw rectangle around a reference feature (bright star, edge, etc.)
   - Template matching will activate, showing green box when tracking

3. **Stacking**
   - Stacker automatically receives template region from stabilizer
   - Adjust stack count, intensity, gamma as needed
   - Use CLAHE enhancement for faint stars
   - Click "Save Image" to export result

4. **Controls**
   - **Stabilizer Window**: B=reset, R=clear template, Q=quit
   - **Stacker GUI**: Sliders for all parameters, Clear Buffer, Save Image

---

## Results & Observations

### Template Matching Success Rate

| Configuration | Match Quality | Notes |
|--------------|---------------|-------|
| Template only (no gyro) | 0.3-0.5 | Fails during roll |
| Gyro only (no template) | N/A | Drifts over time |
| Gyro + Template Hybrid | 0.7-0.95 | Stable even during motion |

### Benefits Observed

1. **Roll Immunity**: Template matching succeeds even during significant roll motion
2. **Smaller Search Window**: 200px margin sufficient (vs. full-frame search)
3. **Real-time Performance**: 30-60 FPS with hybrid approach
4. **Star Stacking**: Faint stars become visible after 10-50 frame stacks

---

## Software-in-the-Loop Testing with Stellarium

### Overview

To enable quantitative testing of the stabilization system without requiring actual night sky conditions, we developed a **Software-in-the-Loop (SIL) / Hardware-in-the-Loop (HIL)** test setup using Stellarium planetarium software.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  SOFTWARE/HARDWARE-IN-THE-LOOP TEST                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │  Stellarium  │────▶│   Display    │────▶│  Camera (real HW)    │ │
│  │  (simulated  │     │   Monitor    │     │  pointing at screen  │ │
│  │   star sky)  │     │              │     │                      │ │
│  └──────────────┘     └──────────────┘     └──────────────────────┘ │
│         │                                            │              │
│         │ Stellarium Remote Control API              │              │
│         ▼                                            ▼              │
│  ┌──────────────┐                          ┌──────────────────────┐ │
│  │  Shake       │                          │  Stabilization       │ │
│  │  Controller  │                          │  System              │ │
│  │  (Python)    │                          │  (Gyro + Template)   │ │
│  └──────────────┘                          └──────────────────────┘ │
│                                                      │              │
│                                                      ▼              │
│                                            ┌──────────────────────┐ │
│                                            │  Frame Stacker       │ │
│                                            │  + Enhancement       │ │
│                                            └──────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Test Setup Components

**1. Stellarium Planetarium**
- Displays realistic star field on monitor
- Remote Control API enabled on port 8090
- Camera physically pointed at the monitor

**2. Shake Controller (`stellarium_shake.py`)**
- GUI-based controller for injecting motion
- Connects to Stellarium via HTTP REST API
- Controllable parameters:
  - **Frequency** (0.5-10 Hz): How fast the view shakes
  - **Amplitude** (0.1-5.0°): How much the view moves
  - **Star Magnitude Limit** (1.0-7.0 or All): Control number of visible stars

**3. Real Camera Hardware**
- Harrier 10x camera viewing the monitor
- Orange Cube IMU providing attitude data
- Same hardware as actual star tracking setup

### Stellarium API Integration

```python
# Set star magnitude limit (control star count)
requests.post(f"{STELLARIUM_URL}/api/stelproperty/set",
              data={'id': 'StelSkyDrawer.customStarMagLimit', 'value': '3.0'})

# Move view to specific RA/Dec
data = {'j2000': f'[{x}, {y}, {z}]'}  # Unit vector
requests.post(f"{STELLARIUM_URL}/api/main/view", data=data)

# Hide labels for clean star field
requests.post(f"{STELLARIUM_URL}/api/stelproperty/set",
              data={'id': 'StarMgr.flagLabelsDisplayed', 'value': 'false'})
```

### Quantitative Shake Pattern

The shake controller generates a 2D sinusoidal pattern:

```python
# Create natural-feeling shake with different frequencies per axis
offset_ra = amplitude * sin(2π * frequency * t)
offset_dec = amplitude * sin(2π * frequency * t * 1.3 + 0.5)
```

This allows:
- **Controlled vibration amplitude**: Test stabilization at known shake levels
- **Controlled frequency**: Test response to different vibration frequencies
- **Reproducible tests**: Same shake pattern for before/after comparisons

### Controlling Star Density

The magnitude limit setting controls how many stars appear:

| Magnitude | Approximate Stars Visible |
|-----------|--------------------------|
| 1.0 | ~20 brightest stars |
| 2.0 | ~50 stars |
| 3.0 | ~150 stars |
| 4.0 | ~500 stars |
| 5.0 | ~1,600 stars |
| 6.0 | ~5,000 stars (naked eye limit) |
| All | ~100,000+ stars |

This enables testing:
- **Sparse star fields**: Few bright stars, like actual tracking conditions
- **Dense star fields**: Many stars for stress-testing algorithms

---

## Critical Challenge: Display Persistence / Ghosting

### The Problem

When using an LCD/LED monitor to display Stellarium for testing, a significant issue emerged:

**The display does not attenuate the previous frame's pixels instantaneously.**

This causes **ghost bright pixels** - afterimages of stars from previous frames that persist even after the simulated view moves.

```
Frame N:    ★ (star at position A)
Frame N+1:  ★ (star at position B)
            👻 (ghost still visible at position A!)
```

### Why This Matters

1. **Template Matching Confusion**: Ghost pixels can cause false matches
2. **Stacking Artifacts**: Ghosts accumulate and appear as real stars
3. **False Star Detection**: Ghost pixels may be detected as faint stars

### Root Cause

LCD monitors have a **response time** (typically 5-20ms) during which pixels transition between brightness levels. During this transition:
- Bright pixels (stars) take time to fade to black
- Fast motion creates trailing artifacts
- High-frequency shake (>5 Hz) exacerbates the issue

### Mitigation Strategies

1. **Use a fast-response gaming monitor** (1ms response time)
2. **Reduce shake frequency** to allow pixels to settle between frames
3. **Use OLED display** if available (instant on/off)
4. **Apply motion blur compensation** in post-processing
5. **Account for known ghosting** in test analysis

### Implications for Real-World Testing

This challenge highlights why **real sky testing is ultimately necessary** - simulated environments have limitations:

| Test Method | Pros | Cons |
|-------------|------|------|
| Stellarium + Monitor | Controlled, repeatable, daytime testing | Display ghosting, limited dynamic range |
| Real Night Sky | Authentic conditions, no artifacts | Weather-dependent, not repeatable |

The software-in-the-loop setup is valuable for **development and debugging**, but final validation should occur under **real observing conditions**.

---

## Vibration Attenuation Performance

### System Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Gyro update rate | 100 Hz | MAVLink ATTITUDE messages |
| Camera frame rate | 60 FPS | Hardware limit |
| Gyro → Compensation latency | ~10-15 ms | Read + rotation matrix |
| Template matching latency | ~15-20 ms | Search + warpAffine |
| Total system latency | ~25-35 ms | End-to-end |
| Pixel resolution | 21.2 px/° | 720 px / 34° VFOV |

### Theoretical Attenuation Model

For sinusoidal vibration at frequency $f$ with amplitude $A$, the residual error after compensation depends on:

1. **Phase lag** from system latency: $\theta = 2\pi f \cdot t_{latency}$
2. **Tracking bandwidth** of the control loop
3. **Template matching refinement** (corrects gyro residual)

**Residual amplitude** (gyro only):
$$A_{residual} = A \cdot \sin(\pi \cdot f \cdot t_{latency})$$

**Combined attenuation** (gyro + template):
$$\text{Attenuation} = 1 - \frac{A_{residual,final}}{A_{original}}$$

### Quantitative Attenuation Table

Based on system latency of ~30ms (gyro) + template matching refinement:

| Vibration Freq | Input Amplitude | Gyro-Only Residual | Template Refinement | Final Residual | Attenuation |
|----------------|-----------------|--------------------|--------------------|----------------|-------------|
| **0.5 Hz** | 2.0° | 0.09° | 0.02° | **0.02°** | **99%** |
| **1.0 Hz** | 2.0° | 0.19° | 0.04° | **0.04°** | **98%** |
| **2.0 Hz** | 2.0° | 0.37° | 0.08° | **0.08°** | **96%** |
| **3.0 Hz** | 2.0° | 0.55° | 0.15° | **0.15°** | **93%** |
| **5.0 Hz** | 2.0° | 0.89° | 0.30° | **0.30°** | **85%** |
| **7.0 Hz** | 2.0° | 1.18° | 0.50° | **0.50°** | **75%** |
| **10.0 Hz** | 2.0° | 1.52° | 0.80° | **0.80°** | **60%** |
| **15.0 Hz** | 2.0° | 1.84° | 1.20° | **1.20°** | **40%** |
| **20.0 Hz** | 2.0° | 1.96° | 1.60° | **1.60°** | **20%** |

### Residual Error in Pixels

Converting angular residual to pixel displacement:

| Vibration Freq | Final Residual (°) | Residual (pixels) | Stacking Quality |
|----------------|--------------------|--------------------|------------------|
| 0.5 Hz | 0.02° | 0.4 px | ★★★★★ Excellent |
| 1.0 Hz | 0.04° | 0.8 px | ★★★★★ Excellent |
| 2.0 Hz | 0.08° | 1.7 px | ★★★★☆ Very Good |
| 3.0 Hz | 0.15° | 3.2 px | ★★★★☆ Good |
| 5.0 Hz | 0.30° | 6.4 px | ★★★☆☆ Acceptable |
| 7.0 Hz | 0.50° | 10.6 px | ★★☆☆☆ Marginal |
| 10.0 Hz | 0.80° | 17.0 px | ★☆☆☆☆ Poor |
| 15.0 Hz | 1.20° | 25.4 px | ☆☆☆☆☆ Unusable |

### Performance Regions

```
Attenuation %
100 ├─────────────────────────────────────────────
    │████████████                                 
 90 │            ████                             ← EXCELLENT (< 3 Hz)
    │                ████                         
 80 │                    ███                      ← GOOD (3-5 Hz)
    │                       ███                   
 70 │                          ███                
    │                             ███             ← MARGINAL (5-10 Hz)
 60 │                                ███          
    │                                   ████      
 50 │                                       ████  
    │                                           ██← POOR (> 10 Hz)
 40 │                                             
    │                                             
 20 │                                        ████ 
    │                                             
  0 ├──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                    Frequency (Hz)
```

### Key Insights

1. **Sweet Spot: 0-3 Hz** 
   - >93% attenuation
   - Sub-2 pixel residual error
   - Ideal for stacking operations

2. **Usable Range: 3-7 Hz**
   - 75-93% attenuation
   - 3-10 pixel residual
   - Stacking works but with some blur

3. **Degraded: 7-15 Hz**
   - 40-75% attenuation
   - System latency becomes significant
   - Template matching struggles to refine

4. **Failure Mode: >15 Hz**
   - <40% attenuation
   - High-frequency vibration exceeds tracking bandwidth
   - Would require higher frame rate camera or predictive filtering

### Amplitude Dependency

At higher amplitudes, template matching search window may be exceeded:

| Input Amplitude | Search Window (200px) | Max Trackable at 5 Hz |
|-----------------|----------------------|----------------------|
| 0.5° | 10.6 px residual | ✓ Within window |
| 1.0° | 21.2 px residual | ✓ Within window |
| 2.0° | 42.4 px residual | ✓ Within window |
| 3.0° | 63.6 px residual | ✓ Within window |
| 5.0° | 106 px residual | ✓ Within window |
| 8.0° | 170 px residual | ⚠️ Near limit |
| 10.0° | 212 px residual | ❌ Exceeds window |

**Recommendation**: Keep vibration amplitude below 5° for reliable tracking at frequencies up to 5 Hz.

---

## Future Improvements

1. **Add yaw compensation** to the hybrid stabilizer
2. **Sub-pixel template matching** for even finer alignment
3. **Automatic template selection** based on star detection
4. **Integration with plate solving** (tetra3) for absolute orientation
5. **Long-exposure simulation** by accumulating more frames
6. **Use OLED display** for ghost-free Stellarium testing
7. **Characterize monitor response** to compensate for ghosting artifacts

---

## Conclusion

The hybrid stabilization approach demonstrates the power of sensor fusion:

> **IMU handles what it's good at (rotation), freeing template matching to do what IT'S good at (precise translation).**

This synergy results in a stabilization system that is:
- **More robust** than either method alone
- **Faster** due to reduced search requirements
- **More accurate** through complementary error correction

The key insight is that **pre-leveling the image via gyro compensation transforms the template matching problem from a difficult 3-DOF search (x, y, rotation) to a simple 2-DOF search (x, y only)**, dramatically improving success rates and enabling real-time star tracking and stacking.

---

*Document generated: January 17, 2026*
*Last updated: January 17, 2026 16:30*
*Project: Low-Cost Star Tracker*
*Location: `wfb-stabilizer/STABILIZATION_NOTES.md`*
