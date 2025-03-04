# ML Anomaly Detection

As the SDLC becomes faster and more sophisticated, the relevance of DevOps practices that could mitigate potential issues by anticipating them has grown significantly. The fusion of AI with DevOps
practices is revolutionizing the way systems are built, deployed, and maintained. AI driven DevOps enables teams to proactively predict failures, optimize performance, and maintain system reliability. This project explores
the application of AI in DevOps, focusing on predictive analytics and anomaly. See the [Implementation Guide](./docs/IMPLEMENTATION.md) with all the steps and architecture decisions explained.


## Executive Summary

This solution monitors system metrics in real time, detects anomalies using an autoencoder trained on the normal behavior of the machine, and triggers alerts via Prometheus and Grafana. This proactive approach minimizes downtime and supports business continuity by enhancing the system reliability.

The anomaly detection system goes beyond the traditional DevOps approach of setting fixed metric thresholds. **Instead of waiting for a single metric, for example CPU, to exceed a known limit, the pipeline continuously monitors subtle multivariate correlations between the different
metrics**, such as CPU and memory usage, disk I/O and many more. Whenever the reconstruction error surpasses the pipeline threshold, an anomaly is flagged, and it could reveal problems before any single metric reach their own individual threshold.

This proactive detection of hidden correlations and unusual resource usage patterns significantly reduces false negatives and helps DevOps teams by improving the reliability of the system, and leading to faster detection.

Moreover, the solution is designed for scalability and integration with modern DevOps with the option of adding automatic remediation actions such as restarting failing pods or scaling resources when anomalies are detected.

## Architecture Overview

![Flowchart (1)](https://github.com/user-attachments/assets/c1b2e5cf-7fce-4b13-93cd-73687f4d4290)

**The system is composed of the following main components:**

1. **Ubuntu VM (Metrics Source)**
   - Hosts the target environment where system metrics originate.

2. **Prometheus & Grafana**
   - **Prometheus**: Scrapes system metrics from the Ubuntu VM at regular intervals and stores them in a time series database.
   - **Grafana**: Visualizes the collected metrics in real time, providing dashboards and alerting capabilities.

3. **Feature Engineering & Model Training**
   - A Python script `process_prometheus_data.py` processes raw metrics into a comprehensive dataset, applying:
     - **Data cleaning:** All the possible NaN values are handled either removed or filled.
     - **Min-Max Scaling** to normalize values.
     - **Rolling averages**, **change rates**, and **ratios** to provide meaningful features.
   - A **TensorFlow** autoencoder model is trained on this processed dataset to learn the system’s normal behavior.

4. **Flask Model Deployment**
   - The trained autoencoder is converted to **TensorFlow Lite** for inference in Ubuntu.
   - A **Flask API** (`flask_inference.py`) hosts the TFLite model and exposes endpoints to:
     - Compute the **reconstruction error** on the latest metrics for the anomaly detection.
     - Serve the current anomaly score via the `/metrics` endpoint.

5. **Prometheus Scraping (Anomaly Score)**
   - **Prometheus** is configured to scrape the Flask API’s `/metrics` endpoint every 15s.
   - This collects the **anomaly_score** (reconstruction error) and stores it alongside system metrics.

6. **Thresholds & Alerting**
   - **Grafana** reads both system metrics and the anomaly score from Prometheus.
   - A **threshold**  0.1998 < MSE flags anomalies when exceeded.
   - Alerts, for example via **Telegram bot**  are sent by grafana to notify DevOps when anomalies occur.

7. **Incident Response**
   - DevOps investigates alerts, diagnoses issues, and remediates any detected anomalies.
   - The process closes the loop, reducing downtime and improving system reliability.


## Workflow: From Data Collection to Alerts

### Data Collection
- **Prometheus Node Exporter** collects real time system metrics.
- Metrics are scraped and stored in Prometheus’ time series database.

### Data Preprocessing and Feature Engineering
Raw data is processed and enriched to improve model performance:

- Data Cleaning
  - Handles **missing values** and inconsistencies in the dataset.

- Feature Scaling
  - **Min-Max Normalization** ensures all values are in the same range [0,1] despite being in different measures, such as Mb/s and percentages.

- Feature Engineering
  - **Rolling Averages** → Smoothens noisy fluctuations.
  - **Change Rates** → Measures sudden spikes that could be strong indicators of anomalies.
  - **Ratios** → Helps identify inefficient resource utilization like the CPU and memory ratio.

- Data Splitting
  - **Training Data** → Used to train the Autoencoder model.
  - **Validation Data** → Used to evaluate the model’s performance.
---

### Training the Autoencoder Model
- The autoencoder is trained to learn the **normal behavior** of system metrics.
- hyperparameter tuning is later made to get the best hyperparameters.
- Once trained, the model is converted to TensorFlow Lite for efficient real time inference.

---

### Deploying the Model with Flask API
- The trained model is **served via a Flask API** in `flask_inference.py`.
- The API:
  - Reads the **latest system metrics** from the processed data.
  - Runs **inference** by feeding data into the Autoencoder.
  - Computes **reconstruction error** to determine if an anomaly is present.

---

### Prometheus Scrapes Anomaly Scores
- Flask exposes a **Prometheus metric (`anomaly_score`)** via the `/metrics` endpoint.
- **Prometheus is configured to scrape this metric every 15 seconds**.
- The anomaly scores are stored in Prometheus and can be queried.

---

### Visualization in Grafana
- **Grafana queries Prometheus** and visualizes the anomaly scores over time.
- A fixed threshold line is used to identify anomalies.
- If the **anomaly score crosses the threshold**, it is marked on the dashboard.

---

### Alerting via Grafana
- **Grafana triggers an alert** when an anomaly is detected, in this case a simple telegram bot was configured to send alerts triggered by Grafana.
- Notifications can be sent via:
  - **Email**
  - **Slack**
  - **Telegram**
  - **Other integrations**

---

## Benefits of AI driven failure prediction

- **Proactive Monitoring:** failures are predicted before they impact the system.
- **Automated responses:** Self healing mechanisms are ready to be set before a major incident.
- **Reduced downtime:** Downtime is minimized improving user satisfaction.

## Conclusion

AI powered DevOps can be a game changer for modern practices. Implementing this technologies requires careful planning, but the benefits are worth the effort. 
By starting to adopt AI on DevOps, the teams can put a major focus on innovation rather than firefighting, allowing to expand the full potential of their infrastructure and applications. This pipeline succesfully integrates data collection, machine learning, real time inference, and alerting.
By combining Prometheus, Grafana, and an autoencoder served via Flask, the system automatically flags anomalous behavior and ensures proactive incident response

---

<i>Note: The autoencoder was trained with over two thousand samples that can be found in the `processed_prometheus_data_scaled.py` capturing a variety of system behaviors and workload conditions, however, the model is designed to continuously evolve as new data is collected. Future iterations can incorporate additional training data to improve accuracy and adapt to changing system patterns to ensure long term reliability.</i>

