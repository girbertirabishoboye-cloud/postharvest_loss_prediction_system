
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Post-Harvest Loss Prediction System",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Post-Harvest Loss Prediction and Monitoring System")
st.caption("Machine learning, environmental monitoring and MQTT-based control")


# =====================================================
# MQTT CONFIGURATION
# =====================================================

BROKER = "broker.emqx.io"
PORT = 8883

try:
    TOPIC_KEY = str(st.secrets["mqtt"]["topic_key"]).strip()
except Exception:
    TOPIC_KEY = ""

if len(TOPIC_KEY) < 16:
    st.error(
        "MQTT topic key is missing or too short. "
        "Configure it in Streamlit Secrets."
    )
    st.code(
        '[mqtt]\n'
        'topic_key = "REPLACE_WITH_A_LONG_RANDOM_KEY"',
        language="toml"
    )
    st.info("Add this under your app's Settings → Secrets.")
    st.stop()

TELEMETRY_TOPIC = f"phms/{TOPIC_KEY}/telemetry"
COMMAND_TOPIC = f"phms/{TOPIC_KEY}/commands"
STATUS_TOPIC = f"phms/{TOPIC_KEY}/status"


# =====================================================
# SHARED MQTT DATA
# =====================================================

@st.cache_resource
def create_mqtt_connection():
    if mqtt is None:
        return None, {"telemetry": None, "status": None, "error": None}

    shared_data = {
        "telemetry": None,
        "status": None,
        "error": None
    }

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            shared_data["error"] = None
            client.subscribe(TELEMETRY_TOPIC)
            client.subscribe(STATUS_TOPIC)
        else:
            shared_data["error"] = f"MQTT connection error: {reason_code}"

    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))

            if message.topic == TELEMETRY_TOPIC:
                shared_data["telemetry"] = payload

            elif message.topic == STATUS_TOPIC:
                shared_data["status"] = payload

        except Exception as error:
            shared_data["error"] = str(error)

    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"PHMS_DASHBOARD_{int(time.time())}"
        )

        client.tls_set()
        client.on_connect = on_connect
        client.on_message = on_message

        client.connect(BROKER, PORT, keepalive=60)
        client.loop_start()

        return client, shared_data

    except Exception as error:
        shared_data["error"] = str(error)
        return None, shared_data


mqtt_client, mqtt_data = create_mqtt_connection()


# =====================================================
# LOAD MACHINE LEARNING MODEL
# =====================================================

MODEL_PATH = Path("postharvest_risk_model.joblib")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


try:
    model = load_model()
except Exception as error:
    model = None
    st.warning(f"Model could not be loaded: {error}")


# =====================================================
# RISK PREDICTION
# =====================================================

def predict_risk(temperature, humidity, soil_moisture):
    """
    Uses the saved model if its input features are available.
    Otherwise, returns a simple threshold-based demonstration.
    """

    if model is not None:
        feature_names = getattr(model, "feature_names_in_", None)

        if feature_names is not None:
            values = {}

            for feature in feature_names:
                name = str(feature).lower()

                if "temperature" in name or name in ["temp", "temp_c"]:
                    values[feature] = temperature

                elif "humidity" in name:
                    values[feature] = humidity

                elif "soil" in name or "moisture" in name:
                    values[feature] = soil_moisture

                else:
                    # Do not silently invent unknown model inputs.
                    return (
                        "MODEL INPUTS NEED CHECKING",
                        "Your trained model has additional input features. "
                        "Check its training dataset before using this prediction."
                    )

            try:
                prediction = model.predict(pd.DataFrame([values]))[0]
                return str(prediction), "Prediction produced by the trained model."
            except Exception as error:
                return (
                    "PREDICTION ERROR",
                    f"Check model features and data types: {error}"
                )

        return (
            "MODEL INPUTS NEED CHECKING",
            "The saved model does not expose feature names. "
            "Check the model's training script."
        )

    # Demonstration fallback only; this is not machine learning.
    if temperature >= 40 or humidity >= 80:
        return "HIGH_RISK", "Environmental conditions are very high."

    if temperature >= 35 or humidity >= 70:
        return "WARNING", "Environmental conditions need attention."

    return "NORMAL", "Environmental conditions are within the demonstration thresholds."


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("System Settings")

mode = st.sidebar.radio(
    "Monitoring mode",
    ["Simulated / Manual", "Live MQTT"]
)

st.sidebar.write("MQTT broker:", BROKER)
st.sidebar.write("MQTT port:", PORT)


# =====================================================
# SIMULATED / MANUAL MODE
# =====================================================

if mode == "Simulated / Manual":

    st.subheader("Manual demonstration readings")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.slider(
            "Temperature (°C)", 0.0, 60.0, 28.0
        )

    with col2:
        humidity = st.slider(
            "Humidity (%)", 0.0, 100.0, 60.0
        )

    with col3:
        soil_moisture = st.slider(
            "Soil moisture (%)", 0, 100, 45
        )

    risk, explanation = predict_risk(
        temperature, humidity, soil_moisture
    )

    st.info("You are using demonstration readings, not live sensor data.")


# =====================================================
# LIVE MQTT MODE
# =====================================================

else:

    st.subheader("Live ESP8266 sensor readings")

    telemetry = mqtt_data["telemetry"]

    if mqtt_client is None:
        st.error("Could not connect to the MQTT broker.")
        if mqtt_data["error"]:
            st.code(mqtt_data["error"])

        st.warning("Check your internet connection and MQTT broker availability.")

        temperature = humidity = None
        soil_moisture = None
        risk = "NOT CONNECTED"
        explanation = "No live data is available."

    elif telemetry is None:
        st.warning(
            "Waiting for ESP8266 sensor data. "
            "Check the Wi-Fi connection, MQTT connection and Arduino topics."
        )

        temperature = humidity = None
        soil_moisture = None
        risk = "WAITING"
        explanation = "No sensor message has arrived yet."

    else:
        temperature = telemetry.get("temperature")
        humidity = telemetry.get("humidity")
        soil_moisture = telemetry.get("soil_moisture_percent")

        risk = telemetry.get("environmental_risk", "UNKNOWN")
        explanation = telemetry.get("recommendation", "No recommendation received.")

        if temperature is None or humidity is None or soil_moisture is None:
            st.warning("Some sensor readings are missing.")
            risk = "SENSOR ERROR"
            explanation = "Check the sensors and their connections."

    if mqtt_data["error"]:
        st.warning(mqtt_data["error"])


# =====================================================
# DISPLAY READINGS
# =====================================================

st.divider()
st.subheader("Environmental Monitoring")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Temperature",
    f"{temperature:.1f} °C" if temperature is not None else "N/A"
)

col2.metric(
    "Humidity",
    f"{humidity:.1f} %" if humidity is not None else "N/A"
)

col3.metric(
    "Soil Moisture",
    f"{soil_moisture:.0f} %" if soil_moisture is not None else "N/A"
)


# =====================================================
# RISK PREDICTION AND RECOMMENDATIONS
# =====================================================

st.divider()
st.subheader("Post-Harvest Risk Assessment")

if mode == "Simulated / Manual" or (
    mode == "Live MQTT" and temperature is not None
):
    if mode == "Simulated / Manual":
        risk, explanation = predict_risk(
            temperature, humidity, soil_moisture
        )

    st.write("**Risk classification:**", risk)
    st.write("**Explanation:**", explanation)

else:
    st.write("Risk classification:", risk)
    st.write(explanation)

if model is None:
    st.warning(
        "The trained model file was not loaded. "
        "The demonstration mode uses simple thresholds instead."
    )


# =====================================================
# REMOTE DEVICE CONTROL
# =====================================================

st.divider()
st.subheader("Remote Device Control")

st.caption(
    "These commands are sent to the ESP8266 through MQTT. "
    "AUTO allows the device to use its programmed control rules."
)

col1, col2, col3 = st.columns(3)

with col1:
    buzzer_command = st.selectbox(
        "Buzzer",
        ["AUTO", "ON", "OFF"],
        key="buzzer_command"
    )

with col2:
    fan_command = st.selectbox(
        "Fan",
        ["AUTO", "ON", "OFF"],
        key="fan_command"
    )

with col3:
    relay_command = st.selectbox(
        "Extra relay",
        ["AUTO", "ON", "OFF"],
        key="relay_command"
    )

if st.button("Send commands to ESP8266", type="primary"):

    if mqtt_client is None:
        st.error("MQTT is not connected. Commands were not sent.")

    else:
        command = {
            "buzzer": buzzer_command,
            "fan": fan_command,
            "relay": relay_command
        }

        try:
            result = mqtt_client.publish(
                COMMAND_TOPIC,
                json.dumps(command),
                qos=0
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                st.success("Command sent to MQTT broker.")
            else:
                st.error("The MQTT command could not be sent.")

        except Exception as error:
            st.error(f"Command error: {error}")


# =====================================================
# DEVICE STATUS
# =====================================================

st.divider()
st.subheader("ESP8266 Device Status")

device_status = mqtt_data["status"]

if device_status:
    st.write(device_status)
else:
    st.write("No device status received yet.")


# =====================================================
# AUTOMATIC REFRESH
# =====================================================

if mode == "Live MQTT":
    st.caption("The dashboard refreshes every 5 seconds in live mode.")
    time.sleep(5)
    st.rerun()
