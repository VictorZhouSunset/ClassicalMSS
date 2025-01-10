*This repo is currently under a big structural change inspired by the META's Demucs project, so it may have extensive bugs and API errors, I am still working on it. And I am trying to upload the datasets I currently have to an external platform (like Kaggle) and create a wrapper for easier access and referencing.*

---

# Music Source Separation (MSS) for Classical Ensemble Music Recordings

## 🎵 Project Overview
An AI-powered approach to separating individual instruments from classical ensemble recordings, pushing the boundaries of Music Source Separation (MSS) beyond its typical application in pop music.

## 🎯 Motivation
As both a conservatory graduate and machine learning enthusiast, I've always been fascinated by the intersection of classical music and AI. While MSS has made remarkable strides in separating pop band instruments, classical ensemble recordings present unique challenges that remain largely unexplored.

Also, after finishing the Machine Learning and NLP specialization on Coursera, I wanted to apply my knowledge to a real-world problem and learn more about machine learning in a practical way and also from a project management perspective.

## 🔬 Research Stages

### 1. Synthetic Data Exploration
- Created basic synthesized tones and layered melodies
- Implemented Wave-U-Net and Conv-Tasnet models
- Achieved impressive ~40dB SDR (Signal-to-Distortion Ratio)

### 2. Semi-Realistic Synthesis
- Utilized FluidSynth and soundfonts for MIDI rendering
- Experimented with various regularization schemes
- Achieved 20-30dB SDR through multi-stage training
- Tested against real piano-violin recordings (obviously, the results are not good)

### 3. State-of-the-Art Implementation
- Adapt the HT-Demucs and other cutting-edge models

### 4. More Realistic Training Data
- Incorporating different audio effects in training data
  - Different reverbs
  - Different EQs
  - Different compressions
- Improving out-of-group (real world) separation performance and finding good metrics to evaluate the performance on live recordings that doesn't have a pre-defined ground truth
- Investigating model behavior and details

### 5. Future Direction
- Fine-tuning with live recordings
- Exploring model explainability

## 🎯 Long-Term Goals
- Using the MSS as a boost to convert audio recordings directly to symbolic representations
- Create a robust dataset (of the original mix and the separated instruments recordings) from existing chamber music recordings
- Study musician collaboration patterns using AI

## 🤔 Interesting Findings
- Weight initialization significantly impacts training results
- Normalization techniques (batch, group, layer) show unexpected negative effects
- Trying to investigate these phenomena through model explainability research


