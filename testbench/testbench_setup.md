# 🛠️ B-SVaR Testbench: Setup & Execution Guide

This testbench allows judges and reviewers to strictly verify the mathematical stability and Bayesian logic of the B-SVaR system without requiring a live video feed or GPU acceleration. It executes four targeted stress tests to prove the system's "Operational Reliability."

### **1. Prerequisites**
This testbench isolates the stochastic math engine and relies on numerical simulations. Ensure you have installed the core mathematical dependencies in your environment. If you have already installed the root `requirements.txt`, you are good to go. 

Otherwise, you can install the required packages directly:
```bash
pip install numpy
```

### **2. Execution Instructions**
The workbench script is designed to be run headless from the terminal. 

Ensure you are in the root directory of the project (`DLW_kuteng`), and execute the following command:
```bash
python testbench/workbench.py
```

### **3. Expected Output & Scenario Breakdown**
The script will output the exact risk spreads (Delta), VaR Thresholds, and Bayesian Posterior logic for the following four scenarios:

* **Scenario 1: Normal Day (Baseline)**
  * **Goal:** Verify baseline sensitivity.
  * **Expected Result:** A moderate vision score (0.65) triggers an alert because the environmental baseline is stable and clear, keeping the VaR threshold low.

* **Scenario 2: False Positive Prevention**
  * **Goal:** Test the Stochastic Prior.
  * **Expected Result:** The Monte Carlo engine correctly raises the VaR "Noise Floor" due to extreme weather (Night/Rain), proving the system defends against visual noise before it even occurs.

* **Scenario 3: False Positive Check**
  * **Goal:** Stress-test the Bayesian Filter.
  * **Expected Result:** The system successfully suppresses a high visual "false positive" (0.70) because the mathematical logic determines the anomaly is absorbed by the chaotic environment of a stormy rush hour.

* **Scenario 4: Actual Crash**
  * **Goal:** Prove the system is not blind in severe conditions.
  * **Expected Result:** Despite a massively elevated noise floor from rain and gridlock, a genuine, high-certainty visual threat (0.95) successfully breaches the VaR threshold, triggering a confirmed alert.
