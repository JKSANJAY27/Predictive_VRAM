# 🚀 Quick Presentation & Demo Field Guide

> **Master Academic Report:** For the full academic discussion, literature review, mathematical proofs, and Q&A defense, see **[ACADEMIC_PROJECT_REPORT_FIRST_DRAFT.md](file:///c:/Users/reape/OneDrive/Desktop/Predictive_VRAM/ACADEMIC_PROJECT_REPORT_FIRST_DRAFT.md)**.

---

## ⚡ The 3-Step Demo Setup (For You & Friends)

### Step 1: Run Setup (Installs everything in `.venv`)
```powershell
.\setup.ps1
```
*(or double-click `setup.bat`, or `./setup` on Mac/Linux)*

### Step 2: Start the Interactive Web Dashboard
```powershell
.\start.ps1
```
*(or double-click `start.bat`, or `./start` on Mac/Linux)*  
Your browser will automatically open at: **`http://localhost:8000`**

### Step 3: Present Live to Professor / Committee
1. **Show Hardware Provenance (Top Pill):**
   - *"Host RAM is genuinely measured with `psutil`, and GPU VRAM is marked UNAVAILABLE because no physical CUDA GPU is attached. Our system strictly follows scientific provenance."*
2. **Show the 3-Tier Topology:**
   - Point out `User Device` (Client / Layers 0–1), `Edge Node A` (Near-Edge / Layers 2–3), and `Edge Node B` (Standby).
3. **Click "🚀 Execute Live Migration Demo":**
   - Watch `Layer 1` physically slide over the wire into `Edge Node A`!
   - Show the 6 migration phases completing in **2.08 milliseconds**.
   - Show the **Token Parity: 100% BIT-IDENTICAL [PASS]** card!
4. **Click "🔮 Run Predictive Forecast (+2.0s)":**
   - Show the network card turning red as bandwidth collapses to 0.00 Mbps, warning the controller before a crash happens.

### Step 4: Stop When Done
```powershell
.\end.ps1
```
*(or double-click `end.bat`, or `./end` on Mac/Linux)*

---

## 💻 Terminal CLI Commands (If Evaluator Prefers Terminal)
1. **Hardware Inspection:** `python scripts/inspect_hardware.py --provider auto`
2. **Split Catalog (78 plans):** `python scripts/run_candidate_demo.py`
3. **Predictive Engine:** `python scripts/run_prediction_demo.py`
4. **Adaptive Controller:** `python scripts/run_controller_demo.py`
5. **Live Migration & Parity:** `python scripts/run_migration_demo.py`
6. **Automated Test Suite (567 Tests):** `pytest tests/test_gpu_validation.py`
