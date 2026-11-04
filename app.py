"""Streamlit demonstration dashboard for post-harvest environmental risk."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json, os, time
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "postharvest_risk_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
FEATURES = ["temperature_c", "relative_humidity_pct", "soil_moisture_pct"]
DISPLAY = {"temperature_c": "Temperature (°C)", "relative_humidity_pct": "Relative humidity (%)",
           "soil_moisture_pct": "Soil moisture (%)"}
TELEMETRY_TOPIC = os.getenv("MQTT_TELEMETRY_TOPIC", "postharvest/device01/telemetry")
CONTROL_TOPIC = os.getenv("MQTT_CONTROL_TOPIC", "postharvest/device01/control")

st.set_page_config(page_title="Post-Harvest Loss Risk Monitor", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file is missing. Run: python train_model.py")
    bundle = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return bundle, metrics

def demo_readings():
    """Repeatable moving demonstration series, clearly marked as simulated."""
    now = datetime.now()
    idx = int(time.time() // 5)
    rng = np.random.default_rng(1000 + idx)
    t = 29 + 4*np.sin(idx/7) + rng.normal(0, .5)
    h = 68 + 12*np.sin(idx/9 + 1) + rng.normal(0, 1.5)
    s = 54 + 18*np.sin(idx/11 + 2) + rng.normal(0, 2)
    current = {"temperature_c": float(np.clip(t, 10, 42)),
               "relative_humidity_pct": float(np.clip(h, 20, 100)),
               "soil_moisture_pct": float(np.clip(s, 0, 100))}
    times = pd.date_range(end=now, periods=48, freq="15min")
    rows = []
    for i, stamp in enumerate(times):
        phase = (idx - (47-i))/7
        rows.append({"timestamp": stamp, "temperature_c": float(np.clip(29+4*np.sin(phase)+rng.normal(0,.8),10,42)),
                     "relative_humidity_pct": float(np.clip(68+12*np.sin(phase/1.3+1)+rng.normal(0,2),20,100)),
                     "soil_moisture_pct": float(np.clip(54+18*np.sin(phase/1.6+2)+rng.normal(0,2.5),0,100))})
    return current, pd.DataFrame(rows)

def recommendations(risk, readings):
    tips = {
      "NORMAL": ["Continue routine monitoring and keep harvested produce clean, shaded and ventilated.",
                 "Use clean, dry containers and avoid placing produce directly on the soil."],
      "WARNING": ["Increase monitoring frequency and inspect produce for condensation, damage or spoilage.",
                  "Improve ventilation and shade; separate damaged produce from healthy produce.",
                  "Check that storage conditions match the crop's recommended temperature and humidity."],
      "HIGH RISK": ["Inspect stored produce promptly and move it to a suitable, well-ventilated storage area if safe.",
                    "Reduce excess humidity/condensation and remove damaged or spoiled items using hygienic handling.",
                    "Check sensor placement and calibration before making storage decisions."]
    }
    return tips.get(risk, tips["WARNING"])

def mqtt_read_latest(host, port, username, password, topic, timeout=2.0):
    """Read one retained/latest MQTT message. Requires optional paho-mqtt package."""
    try:
        import paho.mqtt.client as mqtt
        received = []
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="postharvest_dashboard")
        if username:
            client.username_pw_set(username, password or None)
        client.on_message = lambda c, u, msg: received.append(msg.payload.decode("utf-8", errors="replace"))
        client.connect(host, int(port), keepalive=10)
        client.subscribe(topic, qos=0)
        client.loop_start()
        deadline = time.time() + timeout
        while time.time() < deadline and not received:
            time.sleep(.1)
        client.loop_stop()
        client.disconnect()
        if received:
            data = json.loads(received[-1])
            return {k: float(data[k]) for k in FEATURES}, data
    except Exception as exc:
        return None, {"error": str(exc)}
    return None, {"error": "No MQTT telemetry received before timeout."}

def send_mqtt_command(host, port, username, password, topic, command):
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="postharvest_dashboard_control")
    if username:
        client.username_pw_set(username, password or None)
    client.connect(host, int(port), keepalive=10)
    info = client.publish(topic, json.dumps(command), qos=0, retain=False)
    info.wait_for_publish(timeout=3)
    client.disconnect()

try:
    bundle, metrics = load_assets()
    model = bundle["model"]
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.info("From the project folder, run `pip install -r requirements.txt` then `python train_model.py`.")
    st.stop()

st.title("🌾 Post-Harvest Loss Risk Monitor")
st.caption("Environmental monitoring and machine-learning demonstration • Academic prototype")
with st.sidebar:
    st.header("Device connection")
    mode = st.radio("Data source", ["Simulated-live demo", "MQTT device"], index=0)
    st.caption("Simulated-live mode works without hardware, cloud database or MQTT credentials.")
    mqtt_host = st.text_input("MQTT broker host", value=os.getenv("MQTT_HOST", ""))
    mqtt_port = st.number_input("MQTT port", min_value=1, max_value=65535, value=int(os.getenv("MQTT_PORT", "1883")))
    mqtt_user = st.text_input("MQTT username (optional)", value=os.getenv("MQTT_USERNAME", ""))
    mqtt_pass = st.text_input("MQTT password (optional)", value=os.getenv("MQTT_PASSWORD", ""), type="password")
    st.caption(f"Telemetry: `{TELEMETRY_TOPIC}`")
    st.caption(f"Control: `{CONTROL_TOPIC}`")
    st.divider()
    st.write("**Model:** Random Forest • v1.0-demo")
    st.write("**Data:** Synthetic demonstration data")

sim_current, sim_history = demo_readings()
device_meta = {}
is_online = False
if mode == "MQTT device":
    if not mqtt_host.strip():
        st.warning("Enter your MQTT broker host. Until then, no physical-device readings are available.")
        current, history = sim_current, sim_history
        data_label = "SIMULATED FALLBACK — MQTT NOT CONFIGURED"
    else:
        current, device_meta = mqtt_read_latest(mqtt_host, mqtt_port, mqtt_user, mqtt_pass, TELEMETRY_TOPIC)
        if current:
            is_online = True
            current = {k: float(np.clip(current[k], 0 if k != "temperature_c" else -10, 100 if k != "temperature_c" else 60)) for k in FEATURES}
            history = sim_history.copy()
            history.loc[len(history)] = [datetime.now(), *[current[k] for k in FEATURES]]
            data_label = "LIVE MQTT TELEMETRY (latest message)"
        else:
            current, history = sim_current, sim_history
            data_label = "SIMULATED FALLBACK — DEVICE OFFLINE"
else:
    current, history = sim_current, sim_history
    data_label = "SIMULATED-LIVE DEMONSTRATION"

pred = model.predict(pd.DataFrame([current], columns=FEATURES))[0]
probs = model.predict_proba(pd.DataFrame([current], columns=FEATURES))[0]
classes = list(model.classes_)
confidence = float(max(probs))
risk_colors = {"NORMAL": "green", "WARNING": "orange", "HIGH RISK": "red"}

st.info(f"**Data mode:** {data_label}  |  **Device status:** {'ONLINE' if is_online else 'OFFLINE / SIMULATED'}")
if mode == "MQTT device" and not is_online:
    st.caption("The dashboard remains demonstrable in fallback simulation. A simulated reading is never presented as physical sensor data.")

tabs = st.tabs(["Executive Overview", "Live Monitoring", "ML Predictions", "Risk Assessment",
                "Historical Analytics", "Alerts & Recommendations", "Model Performance",
                "Remote Device Control", "System Information"])

with tabs[0]:
    st.subheader("Executive Overview")
    a,b,c,d = st.columns(4)
    a.metric("Temperature", f"{current['temperature_c']:.1f} °C")
    b.metric("Relative humidity", f"{current['relative_humidity_pct']:.1f} %")
    c.metric("Soil moisture", f"{current['soil_moisture_pct']:.1f} %")
    d.metric("Predicted risk", pred)
    st.metric("Prediction confidence", f"{confidence*100:.1f}%")
    st.write("This prototype estimates environmental risk categories from sensor inputs. It does not directly measure crop quality or quantify actual post-harvest loss.")
    st.subheader("Latest readings")
    st.dataframe(pd.DataFrame([{"Measurement": DISPLAY[k], "Value": round(current[k],2)} for k in FEATURES]), hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("Live Sensor Monitoring")
    cols = st.columns(3)
    for col, key in zip(cols, FEATURES):
        col.metric(DISPLAY[key], f"{current[key]:.1f}")
    st.line_chart(history.set_index("timestamp")[FEATURES], use_container_width=True)
    st.caption("In MQTT mode, this demonstration keeps a simulated historical background and appends the latest received reading. A production deployment should persist every telemetry message to durable storage.")

with tabs[2]:
    st.subheader("Machine Learning Predictions")
    st.markdown(f"### Current classification: :{risk_colors.get(pred,'orange')}[{pred}]")
    st.progress(confidence, text=f"Highest class probability: {confidence:.1%}")
    prob_df = pd.DataFrame({"Risk category": classes, "Probability": probs}).set_index("Risk category")
    st.bar_chart(prob_df)
    st.write("Probabilities are model outputs, not calibrated guarantees of real-world risk.")

with tabs[3]:
    st.subheader("Environmental Risk Assessment")
    st.write(f"**Assessment:** {pred}")
    st.write(f"**Inputs:** {current['temperature_c']:.1f} °C, {current['relative_humidity_pct']:.1f}% RH, {current['soil_moisture_pct']:.1f}% soil moisture.")
    st.write("Soil moisture is included as contextual field/environment information. For post-harvest storage, measuring the produce or storage-air conditions directly is more relevant; soil moisture alone is not a post-harvest spoilage indicator.")

with tabs[4]:
    st.subheader("Historical Analytics")
    st.line_chart(history.set_index("timestamp")[FEATURES], use_container_width=True)
    st.subheader("Risk distribution in demonstration history")
    hist_predictions = model.predict(history[FEATURES])
    dist = pd.Series(hist_predictions).value_counts().reindex(classes, fill_value=0)
    st.bar_chart(dist)
    st.caption("Historical series is simulated except for the latest MQTT point when connected.")

with tabs[5]:
    st.subheader("Alerts and Recommendations")
    if pred == "NORMAL":
        st.success("No elevated environmental risk predicted by the demonstration model.")
    elif pred == "WARNING":
        st.warning("Environmental conditions require attention.")
    else:
        st.error("High environmental risk predicted. Inspect storage conditions and produce.")
    for item in recommendations(pred, current):
        st.markdown(f"- {item}")
    st.caption("Recommendations are general educational guidance; crop-specific thresholds should be validated with agricultural experts.")

with tabs[6]:
    st.subheader("Model Performance")
    st.warning(metrics.get("data_notice", "Synthetic demonstration metrics only."))
    m1,m2,m3,m4 = st.columns(4)
    for col, key, label in [(m1,"accuracy","Accuracy"),(m2,"precision_macro","Macro precision"),(m3,"recall_macro","Macro recall"),(m4,"f1_macro","Macro F1")]:
        col.metric(label, f"{metrics.get(key,0):.3f}")
    cm = np.array(metrics.get("confusion_matrix", []))
    labels = metrics.get("class_labels", classes)
    if cm.size:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.imshow(cm)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title("Confusion matrix — synthetic holdout set")
        for (i,j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("A random split from synthetic, rule-labelled data evaluates code behaviour only. It does not establish accuracy on real farms, crops or storage facilities.")

with tabs[7]:
    st.subheader("Remote Device Control")
    st.write("Control messages are sent only in MQTT mode with a configured broker. Use a low-voltage DC fan/load for the classroom prototype.")
    fan = st.toggle("Fan relay ON", value=False, key="fan_control")
    aux = st.toggle("Auxiliary relay ON", value=False, key="aux_control")
    if st.button("Send control command", type="primary"):
        if mode != "MQTT device" or not mqtt_host.strip():
            st.error("Remote control is unavailable in simulated mode. Configure an MQTT broker first.")
        else:
            try:
                send_mqtt_command(mqtt_host, mqtt_port, mqtt_user, mqtt_pass, CONTROL_TOPIC,
                                  {"fan": bool(fan), "aux": bool(aux), "issued_at": datetime.now(timezone.utc).isoformat()})
                st.success("Command published to MQTT broker. Device acknowledgement is not implemented; verify the hardware state separately.")
            except Exception as exc:
                st.error(f"Could not publish command: {exc}")
    st.caption("Do not switch mains-powered equipment with a student-built prototype. Use a certified, enclosed interface and qualified supervision for any mains installation.")

with tabs[8]:
    st.subheader("System Information")
    st.markdown("""
    **Architecture:** DHT11 + YL-69 → ESP8266 NodeMCU → MQTT broker → Streamlit dashboard → Random Forest prediction.

    **MQTT topics (must match firmware):**
    - Telemetry: `postharvest/device01/telemetry`
    - Control: `postharvest/device01/control`

    **Synthetic-data notice:** the included CSV and model are demonstration artifacts generated by `train_model.py`. Replace them with ethically collected, crop-specific and properly labelled field/storage data before making research or deployment claims.

    **Important scope limitation:** the model predicts a demonstration environmental risk class, not measured post-harvest loss percentage, shelf life, or confirmed spoilage.
    """)
    st.json({"model_version": bundle.get("model_version"), "features": FEATURES,
             "classes": classes, "data_mode": data_label, "mqtt_telemetry_topic": TELEMETRY_TOPIC,
             "mqtt_control_topic": CONTROL_TOPIC})

st.divider()
st.caption("Academic prototype • Not a food-safety certification or substitute for crop-specific storage guidance.")
