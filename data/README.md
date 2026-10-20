# Synthetic Data Documentation & Provenance

## VitalTwin AI — Happiest Health Digital Twin Challenge 2026

### 1. Synthetic Data Overview
All patient profiles, Electronic Health Record (EHR) baselines, continuous glucose monitor (CGM) readings, and wearable sensor streams in **VitalTwin AI** are **100% synthetically generated in silico**. No real human patient data, Protected Health Information (PHI), or electronic medical records were collected, ingested, or stored.

### 2. Clinical Phenotypes Simulated
The system includes three distinct clinical archetypes representing common adult Type 2 Diabetes phenotypes:

1. **PT-1001 — Eleanor Vance (Moderate / Well-Managed)**:
   - **Age:** 58 | **Sex:** Female | **BMI:** 27.4 kg/m²
   - **T2D Duration:** 6.0 years | **HbA1c:** 7.4%
   - **Medication:** Metformin 1000mg BID
   - **Characteristics:** Demonstrates predictable postprandial excursions after carbohydrate meals, with moderate clearance facilitated by regular daily walking.

2. **PT-1002 — Marcus Reed (Brittle / High Glycemic Variability)**:
   - **Age:** 64 | **Sex:** Male | **BMI:** 32.8 kg/m²
   - **T2D Duration:** 14.0 years | **HbA1c:** 8.9%
   - **Medication:** Basal Insulin Glargine 28 units + Empagliflozin 10mg
   - **Characteristics:** Pronounced dawn phenomenon (early morning hepatic glucose release), blunted physical activity clearance, frequent postprandial excursions exceeding 200 mg/dL, and higher glycemic standard deviation.

3. **PT-1003 — Priya Sharma (Early-Onset / Lifestyle Responsive)**:
   - **Age:** 42 | **Sex:** Female | **BMI:** 24.6 kg/m²
   - **T2D Duration:** 1.5 years | **HbA1c:** 6.7%
   - **Medication:** Metformin 500mg daily + Lifestyle Modification
   - **Characteristics:** Preserved endogenous beta-cell sensitivity, rapid post-exercise glucose uptake, excursions driven primarily by irregular high-carbohydrate meals.

### 3. Mathematical & Physiological Simulation Formulation
- **Sampling Interval:** 5 minutes (standard commercial CGM sampling frequency, e.g. Dexcom G7, Abbott FreeStyle Libre 3).
- **Postprandial Absorption Dynamics:**
  $$\Delta G_{meal}(t) = \text{Carbs} \times S_{carb} \times \left(\frac{t - t_{meal}}{\tau}\right) \exp\left(1 - \frac{t - t_{meal}}{\tau}\right)$$
  where $\tau \approx 45 \text{ min}$ (time-to-peak) and $S_{carb}$ is the patient-specific carbohydrate excursion coefficient.
- **Diurnal & Dawn Phenomenon:**
  Modeled using a composite sinusoidal circadian curve combined with a Gaussian morning dawn phenomenon spike centered at 06:30 AM:
  $$G_{dawn}(h) = A_{dawn} \exp\left(-0.5 \left(\frac{h - 6.5}{1.5}\right)^2\right)$$
- **Physical Activity Glucose Clearance:**
  Muscular glucose uptake is simulated as a linear clearance function scaled by active steps and the patient's insulin sensitivity / clearance factor:
  $$\Delta G_{clearance}(t) = -\text{Steps}_t \times K_{clearance}$$
- **Sensor Autocorrelation & Noise:**
  Sensor measurement noise is modeled as an autoregressive process $\text{AR}(1)$ with coefficient $\phi = 0.85$ and $\sigma = 1.2\text{ mg/dL}$, mirroring authentic interstitial fluid lag and probe variance.

### 4. Privacy, Compliance, and Ethical Declaration
- **Compliance:** Inherently compliant with HIPAA Safe Harbor, European Union General Data Protection Regulation (GDPR), and the Indian Digital Personal Data Protection (DPDP) Act 2023 because no identifiable natural person's data is present.
- **Reproducibility:** Seeded pseudo-random number generation (`random_state=42`) enables full determinism and independent auditability.
