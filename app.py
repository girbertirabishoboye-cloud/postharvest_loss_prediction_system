
import json
import time
import uuid
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Post-Harvest Loss Prediction System",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "postharvest_risk_model.joblib"
DATASET_PATH = BASE_DIR / "demo_training_dataset.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"

BROKER = "broker.emqx.io"
PORT = 8883

# =====================================================
# 2. MQTT SECRETS
# =====================================================

try:
    TOPIC_KEY = str(
        st.secrets["mqtt"]["topic_key"]
    ).strip()
except Exception:
    TOPIC_KEY = ""

MQTT_READY = len(TOPIC_KEY) >= 16

if MQTT_READY:
    TELEMETRY_TOPIC = f"phms/{TOPIC_KEY}/telemetry"
    COMMAND_TOPIC = f"phms/{TOPIC_KEY}/commands"
    STATUS_TOPIC = f"phms/{TOPIC_KEY}/status"
else:
    TELEMETRY_TOPIC = None
    COMMAND_TOPIC = None
    STATUS_TOPIC = None

# =====================================================
# 3. LOAD OPTIONAL ML MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    import joblib

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)


@st.cache_data
def load_dataset():

    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    return pd.DataFrame()


@st.cache_data
def load_metrics():

    if METRICS_PATH.exists():

        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)

    return {}


dataset = load_dataset()

metrics = load_metrics()

# =====================================================
# 4. MQTT CONNECTION
# =====================================================

@st.cache_resource
def create_mqtt_connection(topic_key):

    state = {
        "connected": False,
        "telemetry": None,
        "status": None,
        "error": None,
        "history": [],
        "last_received": None,
        "lock": threading.Lock()
    }

    if mqtt is None:
        state["error"] = "Install paho-mqtt>=2.0"
        return None, state

    telemetry_topic = f"phms/{topic_key}/telemetry"
    status_topic = f"phms/{topic_key}/status"

    def on_connect(
        client, userdata, flags,
        reason_code, properties
    ):

        if reason_code == 0:

            with state["lock"]:
                state["connected"] = True
                state["error"] = None

            client.subscribe(telemetry_topic)
            client.subscribe(status_topic)

        else:

            with state["lock"]:
                state["connected"] = False
                state["error"] = (
                    f"MQTT connection rejected: {reason_code}"
                )

    def on_disconnect(
        client, userdata, disconnect_flags,
        reason_code, properties
    ):

        with state["lock"]:
            state["connected"] = False

            if reason_code != 0:
                state["error"] = (
                    f"MQTT disconnected: {reason_code}"
                )

    def on_message(client, userdata, message):

        try:

            payload = json.loads(
                message.payload.decode("utf-8")
            )

            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")

            with state["lock"]:

                if message.topic == telemetry_topic:

                    payload["temperature"] = payload.get(
                        "temperature",
                        payload.get("temp")
                    )

                    payload["humidity"] = payload.get(
                        "humidity"
                    )

                    payload["soil_moisture_percent"] = (
                        payload.get(
                            "soil_moisture_percent",
                            payload.get(
                                "soil_moisture",
                                payload.get("moisture")
                            )
                        )
                    )

                    state["telemetry"] = payload
                    state["last_received"] = time.time()

                    record = dict(payload)

                    record["received_at"] = (
                        time.strftime("%Y-%m-%d %H:%M:%S")
                    )

                    state["history"].append(record)

                    state["history"] = (
                        state["history"][-100:]
                    )

                elif message.topic == status_topic:

                    state["status"] = payload

        except Exception as error:

            with state["lock"]:
                state["error"] = str(error)

    try:

        client = mqtt.Client(
            callback_api_version=(
                mqtt.CallbackAPIVersion.VERSION2
            ),
            client_id=(
                "PHMS_DASH_" +
                uuid.uuid4().hex[:10]
            )
        )

        client.tls_set()

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message

        client.reconnect_delay_set(
            min_delay=1,
            max_delay=30
        )

        client.connect(
            BROKER,
            PORT,
            keepalive=60
        )

        client.loop_start()

        return client, state

    except Exception as error:

        state["error"] = str(error)

        return None, state


if MQTT_READY:

    mqtt_client, mqtt_data = create_mqtt_connection(
        TOPIC_KEY
    )

else:

    mqtt_client = None

    mqtt_data = {
        "connected": False,
        "telemetry": None,
        "status": None,
        "error": "MQTT topic key is missing.",
        "history": [],
        "last_received": None,
        "lock": threading.Lock()
    }

# =====================================================
# 5. RISK ASSESSMENT
# =====================================================

def assess_risk(temperature, humidity):

    if temperature >= 40 or humidity >= 80:

        return (
            "HIGH RISK",
            "Check the storage environment immediately."
        )

    elif temperature >= 35 or humidity >= 70:

        return (
            "WARNING",
            "Increase monitoring of storage conditions."
        )

    return (
        "NORMAL",
        "Conditions are within demonstration thresholds."
    )


# =====================================================
# 6. VALIDATE SENSOR DATA
# =====================================================

def valid_readings(data):

    try:

        t = float(data["temperature"])
        h = float(data["humidity"])
        s = float(data["soil_moisture_percent"])

        return (
            np.isfinite(t)
            and np.isfinite(h)
            and np.isfinite(s)
            and 0 <= t <= 60
            and 0 <= h <= 100
            and 0 <= s <= 100
        )

    except (TypeError, ValueError, KeyError):

        return False


# =====================================================
# 7. DISPLAY FUNCTIONS
# =====================================================

def display_readings(data):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🌡️ Temperature",
            f'{float(data["temperature"]):.1f} °C'
        )

    with col2:

        st.metric(
            "💧 Humidity",
            f'{float(data["humidity"]):.1f} %'
        )

    with col3:

        st.metric(
            "🌱 Soil Moisture",
            f'{float(data["soil_moisture_percent"]):.1f} %'
        )


def display_risk(risk, explanation):

    if risk == "HIGH RISK":

        st.error(f"**{risk}**: {explanation}")

    elif risk == "WARNING":

        st.warning(f"**{risk}**: {explanation}")

    else:

        st.success(f"**{risk}**: {explanation}")


# =====================================================
# 8. MACHINE LEARNING PREDICTION
# =====================================================

def predict_risk(temperature, humidity, soil):

    if model is None:
        return None, "Trained model is unavailable."

    features = getattr(
        model,
        "feature_names_in_",
        None
    )

    if features is None:

        return None, "Model feature names are unavailable."

    values = {}

    for feature in features:

        name = str(feature).lower().replace(" ", "_")

        if "temperature" in name or name in ["temp", "temp_c"]:
            values[feature] = temperature

        elif "humidity" in name:
            values[feature] = humidity

        elif "soil" in name or "moisture" in name:
            values[feature] = soil

        else:

            return None, (
                f"Unsupported model feature: {feature}"
            )

    try:

        prediction = model.predict(
            pd.DataFrame([values])
        )[0]

        return prediction, "ML prediction generated."

    except Exception as error:

        return None, str(error)


# =====================================================
# 9. SIDEBAR
# =====================================================

st.sidebar.title("🌾 System Navigation")

page = st.sidebar.radio(
    "Choose page",
    [
        "Dashboard",
        "Risk Prediction",
        "Dataset",
        "Model Performance",
        "MQTT Monitoring",
        "Device Control",
        "About System"
    ]
)

st.sidebar.divider()

st.sidebar.write("MQTT Broker:", BROKER)
st.sidebar.write("MQTT Port:", PORT)

if MQTT_READY:
    st.sidebar.success("MQTT key configured")
else:
    st.sidebar.error("MQTT key missing")

if model is not None:
    st.sidebar.success("ML model loaded")
else:
    st.sidebar.warning("ML model unavailable")

# =====================================================
# 10. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("System Dashboard")

    st.write(
        "Monitor live environmental conditions "
        "and post-harvest storage risks."
    )

    @st.fragment(run_every="3s")
    def live_dashboard():

        with mqtt_data["lock"]:

            connected = mqtt_data["connected"]

            telemetry = (
                dict(mqtt_data["telemetry"])
                if mqtt_data["telemetry"]
                else None
            )

            history = list(mqtt_data["history"])

            last_received = mqtt_data["last_received"]

            error = mqtt_data["error"]

        if not MQTT_READY:

            st.error("Configure MQTT topic key in Secrets.")

            return

        if not connected:

            st.warning("Connecting to MQTT broker...")

            if error:
                st.caption(error)

            return

        if telemetry is None:

            st.info(
                "Connected to MQTT. Waiting for "
                "sensor readings from ESP8266."
            )

            st.caption(
                f"Subscribed topic: {TELEMETRY_TOPIC}"
            )

            return

        if not valid_readings(telemetry):

            st.warning("Invalid or incomplete sensor readings.")

            st.json(telemetry)

            return

        st.success("Live sensor connection active.")

        display_readings(telemetry)

        risk, explanation = assess_risk(
            float(telemetry["temperature"]),
            float(telemetry["humidity"])
        )

        display_risk(risk, explanation)

        if last_received:

            st.caption(
                "Last received: " +
                time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(last_received)
                )
            )

        if history:

            st.subheader("Sensor History")

            history_df = pd.DataFrame(history)

            chart_columns = [
                "temperature",
                "humidity",
                "soil_moisture_percent"
            ]

            st.line_chart(
                history_df[chart_columns]
            )

            st.dataframe(
                history_df.tail(10),
                use_container_width=True,
                hide_index=True
            )

    live_dashboard()

# =====================================================
# 11. RISK PREDICTION
# =====================================================

elif page == "Risk Prediction":

    st.header("Machine Learning Risk Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:

        temperature = st.number_input(
            "Temperature °C",
            0.0, 60.0, 28.0
        )

    with col2:

        humidity = st.number_input(
            "Humidity %",
            0.0, 100.0, 60.0
        )

    with col3:

        soil = st.number_input(
            "Soil Moisture %",
            0.0, 100.0, 45.0
        )

    if st.button("Predict Risk", type="primary"):

        prediction, message = predict_risk(
            temperature,
            humidity,
            soil
        )

        if prediction is not None:

            st.success(f"ML Prediction: {prediction}")

        else:

            st.warning(message)

            risk, explanation = assess_risk(
                temperature,
                humidity
            )

            st.caption("Demonstration threshold assessment:")
            display_risk(risk, explanation)

# =====================================================
# 12. DATASET
# =====================================================

elif page == "Dataset":

    st.header("Training Dataset")

    if not dataset.empty:

        st.write("Records:", len(dataset))

        st.write("Columns:", len(dataset.columns))

        st.dataframe(
            dataset,
            use_container_width=True
        )

        st.download_button(
            "Download Dataset",
            dataset.to_csv(index=False),
            file_name="postharvest_dataset.csv",
            mime="text/csv"
        )

    else:

        st.warning("Dataset file not found.")

# =====================================================
# 13. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.header("Model Performance")

    if model is not None:

        st.success("ML model loaded successfully.")

        st.write("Model type:", type(model).__name__)

        features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if features is not None:
            st.write("Model features:", list(features))

    else:

        st.warning("ML model unavailable.")

        if model_error:
            st.code(model_error)

    if metrics:

        st.subheader("Evaluation Metrics")

        for name, value in metrics.items():

            st.write(
                name.replace("_", " ").title(),
                value
            )

    else:

        st.info("No model metrics file found.")

# =====================================================
# 14. MQTT MONITORING
# =====================================================

elif page == "MQTT Monitoring":

    st.header("MQTT Monitoring")

    if not MQTT_READY:

        st.error("MQTT topic key is missing.")

    else:

        with mqtt_data["lock"]:

            connected = mqtt_data["connected"]

            telemetry = mqtt_data["telemetry"]

            status = mqtt_data["status"]

            error = mqtt_data["error"]

        if connected:

            st.success("MQTT connection active.")

        else:

            st.error("MQTT connection inactive.")

        st.write("Broker:", BROKER)
        st.write("Port:", PORT)

        st.write("Telemetry topic:", TELEMETRY_TOPIC)
        st.write("Commands topic:", COMMAND_TOPIC)
        st.write("Status topic:", STATUS_TOPIC)

        if error:
            st.error(error)

        st.subheader("Latest Sensor Readings")

        if telemetry:
            st.json(telemetry)
        else:
            st.info("No sensor readings received.")

        st.subheader("Device Status")

        if status:
            st.json(status)
        else:
            st.info("No device status received.")

# =====================================================
# 15. DEVICE CONTROL
# =====================================================

elif page == "Device Control":

    st.header("Remote Device Control")

    if not MQTT_READY or mqtt_client is None:

        st.warning("MQTT is not configured.")

    elif not mqtt_data["connected"]:

        st.warning("MQTT connection is inactive.")

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            buzzer = st.selectbox(
                "Buzzer",
                ["AUTO", "ON", "OFF"]
            )

        with col2:

            fan = st.selectbox(
                "Fan",
                ["AUTO", "ON", "OFF"]
            )

        with col3:

            relay = st.selectbox(
                "Extra Relay",
                ["AUTO", "ON", "OFF"]
            )

        if st.button("Send Commands", type="primary"):

            command = {
                "buzzer": buzzer,
                "fan": fan,
                "relay": relay
            }

            result = mqtt_client.publish(
                COMMAND_TOPIC,
                json.dumps(command),
                qos=0
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:

                st.success(
                    "Command submitted to MQTT."
                )

                st.caption(
                    "This does not confirm that a physical device "
                    "has applied the command."
                )

            else:

                st.error("Command publishing failed.")

# =====================================================
# 16. ABOUT SYSTEM
# =====================================================

elif page == "About System":

    st.header("About the Project")

    st.markdown("""
**Project:** Machine Learning-Based System for Predicting
and Reducing Post-Harvest Losses Among Farmers.

**Technologies:**
- ESP8266 NodeMCU
- DHT11 sensor
- Capacitive soil moisture sensor
- MQTT communication
- Python and Streamlit
- Machine Learning

**Objectives:**
1. Monitor environmental conditions.
2. Predict storage risks.
3. Support post-harvest loss reduction.
4. Provide remote environmental monitoring.
""")

# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.caption(
    "Post-Harvest Loss Prediction System | Academic Project"
)
