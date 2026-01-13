# Low-Cost Star Tracker Development Roadmap

## Overview

This roadmap outlines the development priorities for the Low-Cost Star Tracker project based on the technical paper evaluation. Items are organized by priority and estimated complexity.

---

## Phase 1: Validation & Documentation (High Priority)

### 1.1 Experimental Validation
**Goal:** Provide quantitative evidence that the system works as designed.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| Positional Accuracy Test | Compare detected star positions against Hipparcos/Gaia catalog | Medium | Pending |
| SNR Scaling Verification | Measure SNR improvement for 1, 4, 9, 16, 25 stacked frames | Low | Pending |
| Processing Benchmarks | Time measurements on Raspberry Pi 4, laptop, desktop | Low | Pending |
| Motion Compensation Demo | Side-by-side: raw vs. gyro-stabilized frames | Medium | Pending |
| Centroid Accuracy | Sub-pixel accuracy measurements vs. ground truth | Medium | Pending |

**Deliverables:**
- `results/accuracy_validation.md` - Accuracy test results
- `results/snr_curves.png` - SNR vs. frame count plots
- `results/benchmark_table.md` - Processing time comparisons

### 1.2 Sensor Fusion Specification
**Goal:** Fully document the IMU fusion algorithm for reproducibility.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| Algorithm Selection | Document: Complementary vs. EKF vs. VQF | Low | Pending |
| Filter Parameters | Publish gain values, noise covariances | Low | Pending |
| Quaternion Handling | Normalization frequency, unwinding prevention | Low | Pending |
| Magnetometer Integration | Document role (or exclusion) in fusion | Low | Pending |
| Numerical Stability | RK4 step size selection, error bounds | Medium | Pending |

**Deliverables:**
- Updated Section 8: Software Algorithms with explicit filter equations
- `src/sensor_fusion/README.md` - Implementation guide

### 1.3 Environmental Characterization
**Goal:** Document real-world performance limitations.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| Temperature Testing | Gyro drift vs. temperature (-10°C to +40°C) | High | Pending |
| Light Pollution Test | SNR degradation in Bortle 5-8 skies | Medium | Pending |
| Long Exposure Analysis | Hot pixel density at 10s, 20s, 30s exposures | Low | Pending |
| Vibration Tolerance | Image quality on tripod vs. handheld vs. vehicle | Medium | Pending |

**Deliverables:**
- `results/environmental_testing.md` - Test results
- Updated limitations section in technical paper

---

## Phase 2: Algorithm Enhancements (Medium Priority)

### 2.1 Gyroscope Drift Compensation
**Goal:** Extend usable observation time beyond current limits.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| Star-Aided Correction | Use detected stars to correct accumulated drift | High | Pending |
| Adaptive Bias Estimation | Online gyro bias tracking during observation | High | Pending |
| Temperature Compensation | Model and correct thermal drift | Medium | Pending |

### 2.2 Optical Calibration Improvements
**Goal:** Better handle wide-angle lens distortion.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| Division Model | Implement alternative to Brown-Conrady for fisheye | Medium | Pending |
| Star-Based Calibration | Use known star positions instead of checkerboard | High | Pending |
| Vignetting Correction | Flat-field calibration workflow | Low | Pending |
| Dark Frame Subtraction | Hot pixel removal for long exposures | Low | Pending |

### 2.3 Triangle Matching Robustness
**Goal:** Improve star identification reliability.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| False Star Rejection | Filter cosmic rays, hot pixels, satellites | Medium | Pending |
| Sparse Field Handling | Minimum 5-star matching capability | Medium | Pending |
| Confidence Metrics | Probabilistic match quality scoring | Medium | Pending |

---

## Phase 3: New Features (Lower Priority)

### 3.1 Plate Solving Integration
**Goal:** Enable absolute celestial coordinate output.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| Astrometry.net API | Cloud-based plate solving integration | Low | Pending |
| Local Solver | Offline astrometry.net on Raspberry Pi | High | Pending |
| WCS Output | World Coordinate System headers in output | Medium | Pending |

### 3.2 Machine Learning Enhancements
**Goal:** Improve detection and quality assessment.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| CNN Star Detection | Train neural network for star detection | High | Pending |
| Quality Classifier | ML-based frame quality scoring | Medium | Pending |
| Satellite/Meteor Detection | Distinguish trails from stars | High | Pending |

### 3.3 All-Sky Network Integration
**Goal:** Enable citizen science participation.

| Task | Description | Complexity | Status |
|------|-------------|------------|--------|
| CAMS Compatibility | Output format compatible with meteor networks | Medium | Pending |
| Multi-Camera Triangulation | 3D trajectory from multiple observers | High | Pending |
| Real-Time Alerts | Automated transient detection and reporting | High | Pending |

---

## Phase 4: Hardware Expansion (Future)

### 4.1 CubeSat Integration
- Radiation tolerance testing
- Power consumption optimization
- Space-grade component alternatives

### 4.2 Alternative Camera Support
- Raspberry Pi HQ Camera integration
- DSLR/mirrorless tethering
- Industrial camera support

### 4.3 Motorized Mount Control
- GoTo mount integration
- Autoguiding output
- Drift alignment assistance

---

## Implementation Timeline

```
Phase 1 (Validation)     ████████████████████  [Current Focus]
Phase 2 (Enhancements)   ░░░░░░░░████████████
Phase 3 (New Features)   ░░░░░░░░░░░░░░██████
Phase 4 (Hardware)       ░░░░░░░░░░░░░░░░░░██
```

---

## Contributing

Contributions welcome in all phases. Priority areas:
1. **Validation testing** - Real-world accuracy measurements
2. **Documentation** - Algorithm details and tutorials
3. **Code review** - Sensor fusion implementation
4. **Field testing** - Environmental characterization

See `CONTRIBUTING.md` for guidelines.

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Positional Accuracy | <1 arcminute | TBD |
| SNR Improvement (25 frames) | >4x (√25) | TBD |
| Processing Speed (1080p) | <5 seconds | TBD |
| Star Detection Rate | >95% (mag <6) | TBD |
| False Positive Rate | <5% | TBD |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial roadmap based on paper evaluation |
