# Machine Learning-Based Post-Harvest Environmental Risk Demonstrator

A runnable academic prototype containing a Random Forest classifier, a Streamlit dashboard, simulated-live readings, optional MQTT integration, and ESP8266 NodeMCU firmware.

## 1. Important academic scope and limitations

- The included CSV is **synthetic demonstration data**, generated reproducibly by `train_model.py`. It is not a record of actual farms, crops, storage rooms, or losses.
- The included model predicts an environmental risk category (`NORMAL`, `WARNING`, `HIGH RISK`) from temperature, relative humidity and soil moisture. It does **not** predict actual loss percentage, shelf life, or confirmed spoilage.
- Soil moisture is mainly a field/environment context variable. For a stronger post-harvest study, add crop-specific storage temperature, storage-air humidity, produce moisture/water activity, handling/storage duration, crop type and observed spoilage/loss labels.
- Accuracy, precision, recall and F1-score calculated on synthetic data only verify the demonstration workflow. They are **not evidence of real-world prediction performance**. Real claims require representative, independently labelled data and validation.
- The firmware thresholds are starter demonstration settings, not universal agricultural recommendations.

## 2. Project contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard, demo mode, optional MQTT mode |
| `train_model.py` | Reproducible data generation, training and evaluation |
| `postharvest_risk_model.joblib` | Trained demonstration model |
| `demo_training_dataset.csv` | Synthetic labelled dataset |
| `model_metrics.json` | Holdout metrics and confusion matrix |
| `requirements.txt` | Python dependencies |
| `esp8266_postharvest_monitor.ino` | NodeMCU firmware |
| `.gitignore` | Excludes local secrets and Python cache |

## 3. Run on Windows (no VS Code required)

1. Install Python 3.10 or 3.11 from https://www.python.org/downloads/ . During setup, tick **Add Python to PATH**.
2. Extract the ZIP file. Open the extracted `postharvest_loss_prediction_system` folder in File Explorer.
3. Click the folder address bar, type `cmd`, and press Enter. A Command Prompt opens in the project folder.
4. Create a virtual environment:

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install packages:

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. (Optional) Rebuild the synthetic dataset, model and metrics:

   ```bat
   python train_model.py
   ```

7. Start the dashboard:

   ```bat
   streamlit run app.py
   ```

8. A browser should open at `http://localhost:8501`. Choose **Simulated-live demo** in the sidebar. This mode requires no ESP8266, MQTT credentials, Firebase or cloud database.

To stop Streamlit, return to Command Prompt and press `Ctrl+C`.

## 4. Dashboard demonstration

The dashboard contains:
1. Executive Overview
2. Live Sensor Monitoring
3. Machine Learning Predictions
4. Environmental Risk Assessment
5. Historical Analytics
6. Alerts and Recommendations
7. Model Performance
8. Remote Device Control
9. System Information

The simulated-live mode creates moving demonstration readings. The interface labels simulated data so it is not mistaken for actual sensor readings.

## 5. Optional MQTT integration

The dashboard and firmware use these exact topics:

- Telemetry (ESP8266 publishes; dashboard subscribes): `postharvest/device01/telemetry`
- Control (dashboard publishes; ESP8266 subscribes): `postharvest/device01/control`

The firmware sends JSON similar to:

```json
{"temperature_c":29.4,"relative_humidity_pct":71.2,"soil_moisture_pct":42.0,"device_id":"device01","uptime_ms":123456}
```

The dashboard publishes control JSON:

```json
{"fan":true,"aux":false,"issued_at":"2026-11-04T08:00:00+00:00"}
```

### Configure a broker

You need an MQTT broker reachable by both the ESP8266 and the Streamlit host. For a local classroom test, a broker can run on a laptop on the same Wi-Fi network. For online deployment, use a reputable hosted broker with authentication and TLS if supported by your hosting environment. Do not expose an unauthenticated broker to the public internet.

Dashboard environment variables (or enter these in the sidebar):

- `MQTT_HOST` — broker hostname
- `MQTT_PORT` — broker port (often 1883 locally; TLS brokers often use 8883)
- `MQTT_USERNAME` — optional username
- `MQTT_PASSWORD` — optional password
- `MQTT_TELEMETRY_TOPIC` — optional; default shown above
- `MQTT_CONTROL_TOPIC` — optional; default shown above

For Streamlit Community Cloud, add credentials under **App → Settings → Secrets** and adapt the app/environment configuration if you use Streamlit secrets. Do not commit passwords or broker credentials to GitHub. The sample app intentionally does not assume credentials. Some hosted environments restrict outbound MQTT ports; test connectivity before relying on remote MQTT. If MQTT cannot connect, the app reports offline/fallback simulation.

The current MQTT dashboard reads the latest message on each rerun and keeps a simulated chart background plus the latest live point. It does not persist a full live time-series database. For production history, add a time-series database or durable message consumer.

## 6. ESP8266 hardware and Arduino IDE

### Parts

- ESP8266 NodeMCU development board
- DHT11 module
- YL-69 soil moisture probe/module
- Buzzer module suitable for the board's GPIO current limits
- Relay modules with compatible 3.3 V logic input, or suitable transistor/driver interfaces
- Low-voltage DC fan and correctly rated external DC supply
- Jumper wires and breadboard for low-voltage signals

### Pin assignment

| Module signal | NodeMCU pin | Notes |
|---|---|---|
| DHT11 DATA | D4 (GPIO2) | Add pull-up if your module does not include one |
| YL-69 AO | A0 | Check your board's A0 maximum input voltage; use a divider if required |
| Buzzer signal | D5 (GPIO14) | Use a suitable driver if the buzzer current exceeds GPIO capability |
| Fan relay IN | D1 (GPIO5) | Relay module/driver input only |
| Auxiliary relay IN | D2 (GPIO4) | Relay module/driver input only |
| Sensor VCC/GND | suitable supply/GND | Check each module voltage specification |

**Power and electrical safety:** Never power a fan or other high-current load from an ESP8266 GPIO. GPIO pins only provide control signals. Use an appropriately rated relay/driver and separate supply for the DC load. Confirm relay input compatibility with 3.3 V logic, current ratings, flyback protection where applicable, and common ground requirements for the low-voltage control circuit. Do not put mains voltage on a breadboard or build exposed mains wiring. Mains-powered loads require a certified enclosed switching device and a qualified adult/electrician. For a student demonstration, use a low-voltage DC fan.

### Install Arduino IDE libraries

1. Install Arduino IDE from https://www.arduino.cc/en/software .
2. In Preferences, add the ESP8266 board package URL from the official ESP8266 Arduino core documentation, then install **ESP8266** in Boards Manager.
3. Select **NodeMCU 1.0 (ESP-12E Module)** under Tools → Board.
4. In Library Manager install:
   - `DHT sensor library` (Adafruit)
   - `Adafruit Unified Sensor`
   - `PubSubClient` (Nick O'Leary)
   - `ArduinoJson` (Benoit Blanchon)
5. Open `esp8266_postharvest_monitor.ino`.
6. Replace `YOUR_WIFI_NAME`, `YOUR_WIFI_PASSWORD`, and `YOUR_MQTT_BROKER_HOST` in the sketch. Add broker credentials only if needed. These are hardware/network-specific settings.
7. Check the relay active level (`RELAY_ON` / `RELAY_OFF`) and calibrate `SOIL_DRY_RAW` and `SOIL_WET_RAW` from your own sensor.
8. Connect the NodeMCU by USB, select the correct COM port, and click Upload. Open Serial Monitor at 115200 baud.

**Important:** The YL-69 is a resistive probe and can corrode; readings vary with soil type, supply and probe condition. Calibrate locally and avoid leaving it powered continuously where corrosion is a concern. A0 voltage limits vary across ESP8266 boards—check the exact board before connecting the analog output.

## 7. GitHub upload and Streamlit Community Cloud

1. Create a GitHub account at https://github.com/ if you do not have one.
2. Create a new public repository, for example `postharvest-loss-demo`.
3. Upload the project files (GitHub → Add file → Upload files). Do **not** upload `.venv`, Wi-Fi passwords, broker passwords, or other secrets.
4. Visit https://share.streamlit.io/ and sign in with GitHub.
5. Select **Create app**, choose the repository, branch (usually `main`) and main file path `app.py`.
6. Deploy. The dependency file `requirements.txt` is included.
7. Open the app URL. It runs in simulated-live mode by default. Configure optional MQTT only after you have a reachable broker and secure credentials.

The ZIP is designed to run locally without any online service. Streamlit Community Cloud is an optional deployment route and depends on current platform/network limits.

## 8. Testing checklist

- [ ] `python train_model.py` completes and regenerates the CSV, `.joblib` model and metrics JSON.
- [ ] `streamlit run app.py` opens the dashboard in simulated-live mode.
- [ ] The overview shows three sensor values, risk class and confidence.
- [ ] The Model Performance tab shows metrics and a confusion matrix.
- [ ] MQTT mode without a broker reports offline/fallback rather than pretending to be live.
- [ ] With a configured broker, the ESP8266 publishes telemetry and receives fan/aux relay commands on the documented topics.
- [ ] Calibrate sensors and verify relay logic using safe low-voltage loads before demonstration.

## 9. Suggested next research stage

Collect crop-specific readings and observed post-harvest outcomes over time. Document sensor calibration, crop variety, storage method, sampling frequency, handling events and label definitions. Split evaluation by farm, storage location or time period (not only random rows) to reduce data leakage. Compare a simple baseline with Random Forest, report class-wise metrics, and validate on a held-out real dataset before making performance claims.
