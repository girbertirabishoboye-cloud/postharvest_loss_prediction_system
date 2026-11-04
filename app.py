
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


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Post-Harvest Loss Prediction System",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Post-Harvest Loss Prediction and Monitoring System")

st.markdown(
    """
    This system monitors environmental conditions, assesses
    post-harvest risks, and supports farmers in improving
    agricultural storage conditions.
    """
)


# =========================================================
# 2. FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "postharvest_risk_model.joblib"
DATASET_PATH = BASE_DIR / "demo_training_dataset.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"


# =========================================================
# 3. MQTT CONFIGURATION
# =========================================================

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 8883

TOPIC_KEY = ""

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


# =========================================================
# 4. LOAD MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


try:
    model = load_model()
    model_error = None

except Exception as error:
    model = None
    model_error = str(error)


# =========================================================
# 5. LOAD MODEL METRICS
# =========================================================

@st.cache_data
def load_metrics():

    if METRICS_PATH.exists():

        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {}


try:
    model_metrics = load_metrics()

except Exception:
    model_metrics = {}


# =========================================================
# 6. LOAD DEMONSTRATION DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    return pd.DataFrame()


dataset = load_dataset()


# =========================================================
# 7. MACHINE LEARNING PREDICTION
# =========================================================

def predict_risk(temperature, humidity, soil_moisture):

    if model is None:

        return (
            "MODEL NOT AVAILABLE",
            "The trained model file could not be loaded."
        )

    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:

        return (
            "MODEL INPUT ERROR",
            "The model does not provide its expected feature names. "
            "Check the training script."
        )

    values = {}

    for feature in feature_names:

        name = str(feature).lower().replace(" ", "_")

        if "temperature" in name or name in ["temp", "temp_c"]:
            values[feature] = temperature

        elif "humidity" in name:
            values[feature] = humidity

        elif "soil" in name or "moisture" in name:
            values[feature] = soil_moisture

        else:

            return (
                "MODEL INPUT ERROR",
                f"The model requires an additional feature: {feature}. "
                "The dashboard needs to collect this feature before "
                "making a valid prediction."
            )

    try:

        input_data = pd.DataFrame([values])

        prediction = model.predict(input_data)[0]

        return (
            str(prediction),
            "Prediction generated using the trained machine learning model."
        )

    except Exception as error:

        return (
            "PREDICTION ERROR",
            str(error)
        )


# =========================================================
# 8. DEMONSTRATION RISK ASSESSMENT
# =========================================================

def demonstration_risk(temperature, humidity):

    if temperature >= 40 or humidity >= 80:

        return (
            "HIGH_RISK",
            "Environmental conditions are very high. "
            "Check the storage conditions."
        )

    elif temperature >= 35 or humidity >= 70:

        return (
            "WARNING",
            "Environmental conditions need attention. "
            "Increase monitoring."
        )

    else:

        return (
            "NORMAL",
            "Environmental conditions are within the "
            "demonstration thresholds."
        )


# =========================================================
# 9. MQTT CONNECTION
# =========================================================

@st.cache_resource
def create_mqtt_connection(topic_key):

    if mqtt is None:

        return None, {
            "telemetry": None,
            "status": None,
            "error": "Paho MQTT is not installed."
        }

    topics = {
        "telemetry": f"phms/{topic_key}/telemetry",
        "commands": f"phms/{topic_key}/commands",
        "status": f"phms/{topic_key}/status"
    }

    shared_data = {
        "telemetry": None,
        "status": None,
        "error": None
    }

    def on_connect(client, userdata, flags, reason_code, properties):

        if reason_code == 0:

            shared_data["error"] = None

            client.subscribe(topics["telemetry"])
            client.subscribe(topics["status"])

        else:

            shared_data["error"] = (
                f"MQTT connection failed: {reason_code}"
            )

    def on_message(client, userdata, message):

        try:

            payload = json.loads(
                message.payload.decode("utf-8")
            )

            if message.topic == topics["telemetry"]:

                shared_data["telemetry"] = payload

            elif message.topic == topics["status"]:

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

        client.connect(
            MQTT_BROKER,
            MQTT_PORT,
            keepalive=60
        )

        client.loop_start()

        return client, shared_data

    except Exception as error:

        shared_data["error"] = str(error)

        return None, shared_data


mqtt_client = None

mqtt_data = {
    "telemetry": None,
    "status": None,
    "error": None
}

if MQTT_READY:

    mqtt_client, mqtt_data = create_mqtt_connection(TOPIC_KEY)


# =========================================================
# 10. SIDEBAR
# =========================================================

st.sidebar.title("System Navigation")

page = st.sidebar.radio(
    "Select a page",
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

st.sidebar.write("MQTT Broker:", MQTT_BROKER)
st.sidebar.write("MQTT Port:", MQTT_PORT)

if MQTT_READY:

    st.sidebar.success("MQTT key configured")

else:

    st.sidebar.warning("MQTT key not configured")

    st.sidebar.caption(
        "Simulated mode remains available. "
        "Configure Secrets to enable live MQTT."
    )


# =========================================================
# 11. DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.header("System Dashboard")

    st.write(
        "Monitor temperature, humidity, soil moisture "
        "and post-harvest environmental risk."
    )

    monitoring_mode = st.radio(
        "Monitoring mode",
        ["Simulated Mode", "Live MQTT Mode"],
        horizontal=True
    )

    temperature = None
    humidity = None
    soil_moisture = None

    if monitoring_mode == "Simulated Mode":

        st.info("You are using demonstration readings.")

        col1, col2, col3 = st.columns(3)

        with col1:

            temperature = st.slider(
                "Temperature (°C)",
                0.0,
                60.0,
                28.0
            )

        with col2:

            humidity = st.slider(
                "Humidity (%)",
                0.0,
                100.0,
                60.0
            )

        with col3:

            soil_moisture = st.slider(
                "Soil moisture (%)",
                0,
                100,
                45
            )

    else:

        if not MQTT_READY:

            st.error(
                "MQTT key is missing or too short. "
                "Configure it in Streamlit Secrets."
            )

            st.code(
                '[mqtt]\n'
                'topic_key = "YOUR_LONG_RANDOM_SHARED_TOPIC_KEY"',
                language="toml"
            )

        elif mqtt_client is None:

            st.error("Could not connect to the MQTT broker.")

            if mqtt_data["error"]:
                st.code(mqtt_data["error"])

        elif mqtt_data["telemetry"] is None:

            st.warning(
                "Waiting for sensor readings from ESP8266. "
                "Check your hardware and MQTT connection."
            )

        else:

            telemetry = mqtt_data["telemetry"]

            temperature = telemetry.get("temperature")
            humidity = telemetry.get("humidity")
            soil_moisture = telemetry.get("soil_moisture_percent")

    if (
        temperature is not None
        and humidity is not None
        and soil_moisture is not None
    ):

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Temperature",
            f"{float(temperature):.1f} °C"
        )

        col2.metric(
            "Humidity",
            f"{float(humidity):.1f} %"
        )

        col3.metric(
            "Soil Moisture",
            f"{float(soil_moisture):.1f} %"
        )

        if monitoring_mode == "Simulated Mode":

            risk, explanation = demonstration_risk(
                float(temperature),
                float(humidity)
            )

            st.caption(
                "This is a threshold-based demonstration, "
                "not a machine-learning prediction."
            )

        else:

            risk = mqtt_data["telemetry"].get(
                "environmental_risk",
                "UNKNOWN"
            )

            explanation = mqtt_data["telemetry"].get(
                "recommendation",
                "No recommendation received."
            )

        st.subheader("Environmental Risk Assessment")

        st.write("Risk classification:", risk)

        st.info(explanation)

    if monitoring_mode == "Live MQTT Mode":

        st.caption("Live readings are refreshed every 5 seconds.")

        time.sleep(5)
        st.rerun()


# =========================================================
# 12. RISK PREDICTION PAGE
# =========================================================

elif page == "Risk Prediction":

    st.header("Machine Learning Risk Prediction")

    st.write(
        "Enter environmental measurements to assess "
        "post-harvest risk using the trained model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=28.0
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

    with col3:

        soil_moisture = st.number_input(
            "Soil moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=45.0
        )

    if st.button("Predict Post-Harvest Risk", type="primary"):

        prediction, explanation = predict_risk(
            temperature,
            humidity,
            soil_moisture
        )

        st.subheader("Prediction Result")

        st.write("Predicted risk:", prediction)

        st.info(explanation)


# =========================================================
# 13. DATASET PAGE
# =========================================================

elif page == "Dataset":

    st.header("Training Dataset")

    if not dataset.empty:

        st.write(
            "Demonstration dataset used for model training."
        )

        st.dataframe(
            dataset,
            use_container_width=True
        )

        st.write("Number of records:", len(dataset))

        st.write("Number of columns:", len(dataset.columns))

        st.write("Dataset columns:")

        st.write(list(dataset.columns))

        csv_data = dataset.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Dataset CSV",
            csv_data,
            file_name="postharvest_dataset.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Dataset not found. Place demo_training_dataset.csv "
            "in the project folder."
        )


# =========================================================
# 14. MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    if model is None:

        st.warning("The trained model is not available.")

        if model_error:
            st.code(model_error)

    else:

        st.success("Trained model loaded successfully.")

        st.write("Model type:", type(model).__name__)

        feature_names = getattr(
            model,
            "feature_names_in_",
            None
        )

        if feature_names is not None:

            st.write("Model input features:")

            st.write(list(feature_names))

    if model_metrics:

        st.subheader("Evaluation Metrics")

        for metric_name, metric_value in model_metrics.items():

            if isinstance(metric_value, (int, float)):

                st.metric(
                    metric_name.replace("_", " ").title(),
                    f"{metric_value:.4f}"
                )

            else:

                st.write(
                    metric_name.replace("_", " ").title(),
                    metric_value
                )

    else:

        st.warning("Model metrics JSON file was not found.")


# =========================================================
# 15. MQTT MONITORING PAGE
# =========================================================

elif page == "MQTT Monitoring":

    st.header("MQTT Device Monitoring")

    if not MQTT_READY:

        st.error(
            "MQTT topic key is missing or too short. "
            "Configure it in Streamlit Secrets."
        )

        st.code(
            '[mqtt]\n'
            'topic_key = "YOUR_LONG_RANDOM_SHARED_TOPIC_KEY"',
            language="toml"
        )

    elif mqtt_client is None:

        st.error("MQTT connection is not available.")

        if mqtt_data["error"]:
            st.code(mqtt_data["error"])

    else:

        st.success("MQTT client started.")

        st.write("Broker:", MQTT_BROKER)

        st.write("Telemetry topic:", TELEMETRY_TOPIC)

        st.write("Commands topic:", COMMAND_TOPIC)

        st.write("Status topic:", STATUS_TOPIC)

        st.subheader("Latest Sensor Message")

        if mqtt_data["telemetry"]:

            st.json(mqtt_data["telemetry"])

        else:

            st.warning("Waiting for sensor telemetry.")

        st.subheader("Latest Device Status")

        if mqtt_data["status"]:

            st.json(mqtt_data["status"])

        else:

            st.warning("No device status received yet.")

        if mqtt_data["error"]:

            st.error(mqtt_data["error"])


# =========================================================
# 16. DEVICE CONTROL PAGE
# =========================================================

elif page == "Device Control":

    st.header("Remote Device Control")

    st.write(
        "Send commands to the ESP8266 through MQTT."
    )

    if not MQTT_READY:

        st.error(
            "Configure MQTT Secrets before sending commands."
        )

    elif mqtt_client is None:

        st.error("MQTT is not connected.")

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            buzzer_mode = st.selectbox(
                "Buzzer",
                ["AUTO", "ON", "OFF"]
            )

        with col2:

            fan_mode = st.selectbox(
                "Fan",
                ["AUTO", "ON", "OFF"]
            )

        with col3:

            relay_mode = st.selectbox(
                "Extra relay",
                ["AUTO", "ON", "OFF"]
            )

        if st.button("Send Device Commands", type="primary"):

            command = {
                "buzzer": buzzer_mode,
                "fan": fan_mode,
                "relay": relay_mode
            }

            try:

                result = mqtt_client.publish(
                    COMMAND_TOPIC,
                    json.dumps(command),
                    qos=0
                )

                if result.rc == mqtt.MQTT_ERR_SUCCESS:

                    st.success("Commands sent to MQTT broker.")

                else:

                    st.error("Failed to send MQTT commands.")

            except Exception as error:

                st.error(f"Command error: {error}")


# =========================================================
# 17. ABOUT SYSTEM PAGE
# =========================================================

elif page == "About System":

    st.header("About the Project")

    st.markdown(
        """
        **Project title:** Machine Learning-Based System for
        Predicting and Reducing Post-Harvest Losses Among Farmers.

        **Main objectives:**

        1. Monitor environmental conditions.
        2. Assess post-harvest storage risks.
        3. Apply machine learning for risk prediction.
        4. Provide recommendations to improve storage.
        5. Support remote monitoring using ESP8266 and MQTT.

        **Technologies:**

        - Python
        - Streamlit
        - Machine Learning
        - Random Forest (if used by the trained model)
        - ESP8266 NodeMCU
        - DHT11 sensor
        - Soil moisture sensor
        - MQTT communication
        """
    )

    st.info(
        "This is an academic demonstration system. "
        "Synthetic training data and demonstration thresholds "
        "should not be treated as validated agricultural advice."
    )


# =========================================================
# 18. FOOTER
# =========================================================

st.divider()

st.caption(
    "Post-Harvest Loss Prediction and Monitoring System | "
    "Academic Project"
)
