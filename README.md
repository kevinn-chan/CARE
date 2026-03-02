# Context Aware Risk Engine (CARE)
**🚦 Autonomous Traffic Incident Engine for Context-Aware Public Safety using Bayesian-Simulated Value-at-Risk**

### **1. The Vision**
Surveillance cameras and traditional safety AI detects objects but ignores context. This creates an "Interpretation Gap" where a rainy night triggers the same alerts as a clear afternoon, leading to fatal alarm fatigue. 

Our project closes this gap by porting **Mathematical and Value-at-Risk (VaR) frameworks** into public safety. We move from *detecting objects* to *quantifying environmental chaos*.

### **2. Key Innovations**
* **The Stochastic Noise Floor (Monte Carlo)**: We establish a dynamic **VaR** threshold using 10,000 Monte Carlo simulations based on road architecture, traffic flow, and weather. This represents the environmental noise floor.
* **Semantic Intent Engine (Zero-Shot CLIP)**: Unlike standard object detection, we use a **CLIP Transformer** to understand the *intent* of a scene using several engineered semantic anchors (e.g., "crumpled metal" vs. "normal traffic").
* **Actionable Certainty**: Our **Persistence Gate** requires consecutive breaches of the VaR threshold before confirming an incident, ensuring emergency responders only act on verified data.

### **3. System Architecture**
* `math_engine.py`: The analytical core. Runs the Monte Carlo simulations and Odds-form Bayesian fusion.
* `computer_vision.py`: The semantic sensor. Performs zero-shot inference using OpenAI's CLIP, featuring Out-of-Domain (OOD) suppression.
* `app.py`: The orchestration layer. A real-time Streamlit dashboard for telemetry and environmental toggle testing.
* `testbench/`: Automated headless scripts to verify mathematical stability across extreme weather and traffic scenarios.

---

### **4. Setup Instructions**

**Step 1: Clone the Repository**
```bash
git clone https://github.com/kevinn-chan/DLW_kuteng.git
cd DLW_kuteng
```

**Step 2: Install Dependencies**
Ensure you have Python 3.9+ installed, then run:
```bash
py -m pip install -r requirements.txt
```
*(Note: Core dependencies include `torch`, `transformers`, `streamlit`, `opencv-python`, `numpy`, `pandas`, and `pillow`.)*

---

### **5. How to Run the Project**

**Launch the Live Dashboard (UI)**
To view the real-time Bayesian fusion and interact with the environmental toggles:
```bash
py -m streamlit run app.py
```
This will open the application in your default web browser. You can use the sidebar to change the "Infrastructure Profile" and watch the Blue VaR Line dynamically adjust to the new environment.

**Run the Automated Testbench (Headless)**
To verify the system's operational reliability and mathematical stability without the UI:
```bash
py testbench/workbench.py
```
This script will output the results of three critical stress tests:
1. **Nominal Flow**: Verifies baseline stability in clear conditions.
2. **False Positive Prevention**: Tests the system's ability to raise the VaR floor during extreme weather (Night/Rain) to prevent false positives.
3. **Actual Crash**: Proves the system is not blind in severe conditions. 

---

**Developed by Team Kuteng**
*Quantifying the chaos of the smart city to support informed decision making.*
