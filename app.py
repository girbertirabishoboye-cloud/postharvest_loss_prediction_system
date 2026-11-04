import json
import time
import uuid
import threading
from pathlib import Path

import joblib
import numpy as np
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
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "postharvest_risk_model.joblib"
DATASET_PATH = BASE_DIR / "demo_training_dataset.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 8883


# =========================================================
# 2. CUSTOM DASHBOARD DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 30px;
        font-weight: 800;
        color: #16834a;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #777;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .sensor-card {
        background: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
        min-height: 115px;
    }

    .sensor-label {
        font-size: 14px;
        color: var(--text-color);
        opacity: 0.75;
    }

    .sensor-value {
        font-size: 27px;
        font-weight: 750;
        margin-top: 8px;
        color: var(--text-color);
    }

    .risk-card {
        padding: 18px;
        border-radius: 12px;
        margin-top: 15px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
    }

    .small-note {
        color: #888;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🌾 Post-Harvest Loss Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Environmental monitoring, machine learning risk prediction and remote device control</div>',
    unsafe_allow_html=True
)


# =========================================================
# 3. LOAD MACHINE LEARNING MODEL
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
# 4. LOAD DATASET AND MODEL METRICS
# =========================================================

@st.cache_data
def load_dataset():
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


try:
    dataset = load_dataset()
except Exception:
    dataset = pd.DataFrame()

try:
    model_metrics = load_metrics()
except Exception:
    model_metrics = {}


# =========================================================
# 5. MQTT CONFIGURATION
# =========================================================

TOPIC_KEY = ""

try:
    TOPIC_KEY = str(st.secrets["mqtt"]["topic_key"]).strip()
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
# 6. MQTT CONNECTION
# =========================================================

@st.cache_resource
def create_mqtt_connection(topic_key):

    shared_data = {
        "telemetry": None,
        "status": None,
        "error": None,
        "connected": False,
        "history": [],
        "lock": threading.Lock()
    }

    if mqtt is None:
        shared_data["error"] = "Paho MQTT is not installed."
        return None, shared_data

    telemetry_topic = f"phms/{topic_key}/telemetry"
    status_topic = f"phms/{topic_key}/status"

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            shared_data["connected"] = True
            shared_data["error"] = None

            client.subscribe(telemetry_topic)
            client.subscribe(status_topic)
        else:
            shared_data["connected"] = False
            shared_data["error"] = f"MQTT connection failed: {reason_code}"

    def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
        shared_data["connected"] = False

    def on_message(client, userdata, message):

        try:
            payload = json.loads(message.payload.decode("utf-8"))

            with shared_data["lock"]:

                if message.topic == telemetry_topic:
                    shared_data["telemetry"] = payload

                    record = dict(payload)
                    record["received_at"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    shared_data["history"].append(record)
                    shared_data["history"] = shared_data["history"][-100:]

                elif message.topic == status_topic:
                    shared_data["status"] = payload

        except Exception as error:
            shared_data["error"] = str(error)

    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"PHMS_DASHBOARD_{uuid.uuid4().hex[:10]}"
        )

        client.tls_set()

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
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
    "error": None,
    "connected": False,
    "history": [],
    "lock": threading.Lock()
}

if MQTT_READY:
    mqtt_client, mqtt_data = create_mqtt_connection(TOPIC_KEY)


# =========================================================
# 7. RISK CLASSIFICATION
# =========================================================

def demonstration_risk(temperature, humidity):

    if temperature >= 40 or humidity >= 80:
        return (
            "HIGH RISK",
            "Environmental conditions are very high. "
            "Check the storage environment."
        )

    elif temperature >= 35 or humidity >= 70:
        return (
            "WARNING",
            "Environmental conditions need attention. "
            "Increase monitoring."
        )

    return (
        "NORMAL",
        "Conditions are within the demonstration thresholds."
    )


# =========================================================
# 8. MACHINE LEARNING PREDICTION
# =========================================================

def predict_risk(temperature, humidity, soil_moisture):

    if model is None:
        return (
            "MODEL NOT AVAILABLE",
            "The trained model could not be loaded."
        )

    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        return (
            "MODEL INPUT ERROR",
            "The model does not provide feature names. "
            "Check the model training process."
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
                f"The model requires another feature: {feature}."
            )

    try:
        input_data = pd.DataFrame([values])
        prediction = model.predict(input_data)[0]

        return (
            str(prediction),
            "Prediction generated using the trained machine learning model."
        )

    except Exception as error:
        return "PREDICTION ERROR", str(error)


# =========================================================
# 9. REUSABLE DISPLAY FUNCTIONS
# =========================================================

def sensor_card(label, value, unit):

    st.markdown(
        f"""
        <div class="sensor-card">
            <div class="sensor-label">{label}</div>
            <div class="sensor-value">{value} {unit}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_sensor_values(temperature, humidity, soil_moisture):

    col1, col2, col3 = st.columns(3)

    with col1:
        sensor_card(
            "Temperature",
            f"{float(temperature):.1f}",
            "°C"
        )

    with col2:
        sensor_card(
            "Humidity",
            f"{float(humidity):.1f}",
            "%"
        )

    with col3:
        sensor_card(
            "Soil Moisture",
            f"{float(soil_moisture):.1f}",
            "%"
        )


def display_risk(risk, explanation):

    st.markdown(
        f"""
        <div class="risk-card">
            <h3>Environmental Risk Assessment</h3>
            <h2>{risk}</h2>
            <p>{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def valid_readings(temperature, humidity, soil_moisture):

    try:
        values = [
            float(temperature),
            float(humidity),
            float(soil_moisture)
        ]

        return (
            all(np.isfinite(value) for value in values)
            and 0 <= values[0] <= 60
            and 0 <= values[1] <= 100
            and 0 <= values[2] <= 100
        )

    except (TypeError, ValueError):
        return False


# =========================================================
# 10. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🌾 System Navigation")

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
        "Simulated mode is available. "
        "Configure Streamlit Secrets for live monitoring."
    )

if model is not None:
    st.sidebar.success("Machine learning model loaded")
else:
    st.sidebar.warning("Machine learning model unavailable")


# =========================================================
# 11. DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.header("System Dashboard")

    st.write(
        "Monitor environmental conditions and assess "
        "post-harvest storage risks."
    )

    monitoring_mode = st.radio(
        "Monitoring mode",
        ["Simulated Mode", "Live MQTT Mode"],
        horizontal=True
    )

    if monitoring_mode == "Simulated Mode":

        st.info("Demonstration readings are being used.")

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
                "Soil Moisture (%)", 0, 100, 45
            )

        display_sensor_values(
            temperature,
            humidity,
            soil_moisture
        )

        risk, explanation = demonstration_risk(
            temperature,
            humidity
        )

        display_risk(risk, explanation)

        st.caption(
            "Simulated dashboard risk uses demonstration thresholds. "
            "It is not an ML prediction."
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
            st.error("Could not create the MQTT connection.")

            if mqtt_data["error"]:
                st.code(mqtt_data["error"])

        else:

            @st.fragment(run_every="5s")
            def live_dashboard():

                if not mqtt_data["connected"]:
                    st.warning(
                        "Connecting to MQTT or waiting for the broker..."
                    )

                    if mqtt_data["error"]:
                        st.caption(mqtt_data["error"])

                    return

                telemetry = mqtt_data["telemetry"]

                if not telemetry:
                    st.info(
                        "Connected to MQTT. Waiting for ESP8266 readings."
                    )
                    return

                temperature = telemetry.get("temperature")
                humidity = telemetry.get("humidity")
                soil_moisture = telemetry.get("soil_moisture_percent")

                if not valid_readings(
                    temperature,
                    humidity,
                    soil_moisture
                ):
                    st.warning(
                        "The received sensor data is incomplete "
                        "or outside the expected range."
                    )
                    st.json(telemetry)
                    return

                st.success("Live MQTT connection is active.")

                display_sensor_values(
                    temperature,
                    humidity,
                    soil_moisture
                )

                risk = telemetry.get(
                    "environmental_risk",
                    "UNKNOWN"
                )

                explanation = telemetry.get(
                    "recommendation",
                    "No recommendation received from the device."
                )

                display_risk(risk, explanation)

                with mqtt_data["lock"]:
                    history = list(mqtt_data["history"])

                if history:
                    history_df = pd.DataFrame(history)

                    st.subheader("Recent Sensor Readings")

                    chart_columns = [
                        column for column in [
                            "temperature",
                            "humidity",
                            "soil_moisture_percent"
                        ]
                        if column in history_df.columns
                    ]

                    if chart_columns:
                        st.line_chart(
                            history_df[chart_columns],
                            use_container_width=True
                        )

                    st.caption(
                        f"Showing up to {len(history_df)} recent readings."
                    )

            live_dashboard()


# =========================================================
# 12. RISK PREDICTION PAGE
# =========================================================

elif page == "Risk Prediction":

    st.header("Machine Learning Risk Prediction")

    st.write(
        "Enter environmental measurements to generate "
        "a prediction using the trained model."
    )

    if model is None:
        st.warning(
            "The trained model is unavailable. "
            "Check that postharvest_risk_model.joblib "
            "is in your project folder."
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
            "Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=45.0
        )

    if st.button("Predict Post-Harvest Risk", type="primary"):

        if model is None:
            st.error("Prediction cannot run because the model is unavailable.")

        else:
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

        col1, col2 = st.columns(2)

        with col1:
            st.write("Number of records:", len(dataset))

        with col2:
            st.write("Number of columns:", len(dataset.columns))

        st.dataframe(
            dataset,
            use_container_width=True
        )

        st.subheader("Dataset Information")

        information = pd.DataFrame({
            "Column": dataset.columns,
            "Data Type": [
                str(dataset[column].dtype)
                for column in dataset.columns
            ],
            "Missing Values": [
                int(dataset[column].isna().sum())
                for column in dataset.columns
            ]
        })

        st.dataframe(
            information,
            use_container_width=True,
            hide_index=True
        )

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

        metric_items = [
            (name, value)
            for name, value in model_metrics.items()
            if isinstance(value, (int, float))
        ]

        if metric_items:

            columns = st.columns(min(4, len(metric_items)))

            for index, (name, value) in enumerate(metric_items):
                with columns[index % len(columns)]:
                    sensor_card(
                        name.replace("_", " ").title(),
                        f"{value:.4f}",
                        ""
                    )

        for name, value in model_metrics.items():
            if not isinstance(value, (int, float)):
                st.write(
                    name.replace("_", " ").title(),
                    value
                )

    else:
        st.warning("Model metrics JSON file was not found.")

    st.caption(
        "Evaluation metrics describe the model's performance "
        "on its evaluation dataset. They do not guarantee "
        "the same performance in real storage environments."
    )


# =========================================================
# 15. MQTT MONITORING PAGE
# =========================================================

elif page == "MQTT Monitoring":

    st.header("MQTT Device Monitoring")

    if not MQTT_READY:

        st.error("MQTT topic key is not configured.")

        st.code(
            '[mqtt]\n'
            'topic_key = "YOUR_LONG_RANDOM_SHARED_TOPIC_KEY"',
            language="toml"
        )

    elif mqtt_client is None:

        st.error("MQTT connection is unavailable.")

        if mqtt_data["error"]:
            st.code(mqtt_data["error"])

    else:

        if mqtt_data["connected"]:
            st.success("MQTT connection is active.")
        else:
            st.warning("Waiting for MQTT connection.")

        st.write("Broker:", MQTT_BROKER)
        st.write("Telemetry topic:", TELEMETRY_TOPIC)
        st.write("Commands topic:", COMMAND_TOPIC)
        st.write("Status topic:", STATUS_TOPIC)

        st.subheader("Latest Sensor Message")

        if mqtt_data["telemetry"]:
            st.json(mqtt_data["telemetry"])
        else:
            st.info("Waiting for sensor telemetry.")

        st.subheader("Latest Device Status")

        if mqtt_data["status"]:
            st.json(mqtt_data["status"])
        else:
            st.info("No device status received yet.")

        if mqtt_data["error"]:
            st.error(mqtt_data["error"])

        st.caption(
            "This page refreshes when Streamlit reruns. "
            "The live dashboard refreshes automatically."
        )


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

    elif mqtt_client is None or not mqtt_data["connected"]:

        st.warning(
            "The MQTT device connection is not active. "
            "Check the broker and ESP8266."
        )

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
                "Extra Relay",
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
                    st.success("Command submitted to the MQTT client.")
                else:
                    st.error("Failed to submit the MQTT command.")

            except Exception as error:
                st.error(f"Command error: {error}")


# =========================================================
# 17. ABOUT SYSTEM PAGE
# =========================================================

elif page == "About System":

    st.header("About the Project")

    st.markdown(
        """
        **Project title:**

        Machine Learning-Based System for Predicting and
        Reducing Post-Harvest Losses Among Farmers.

        **Main objectives:**

        1. Monitor environmental conditions.
        2. Assess post-harvest storage risks.
        3. Apply machine learning for risk prediction.
        4. Provide recommendations to improve storage.
        5. Support remote monitoring using ESP8266 and MQTT.

        **Technologies used:**

        - Python
        - Streamlit
        - Machine Learning
        - Random Forest, if used by the trained model
        - ESP8266 NodeMCU
        - DHT11 sensor
        - Soil moisture sensor
        - MQTT communication
        """
    )

    st.subheader("System Components")

    components = pd.DataFrame({
        "Component": [
            "ESP8266 NodeMCU",
            "DHT11 Sensor",
            "Soil Moisture Sensor",
            "MQTT Broker",
            "Machine Learning Model",
            "Streamlit Dashboard"
        ],
        "Purpose": [
            "Collects and transmits sensor readings",
            "Measures temperature and humidity",
            "Measures soil moisture",
            "Transfers data between device and dashboard",
            "Predicts post-harvest risk",
            "Displays monitoring and prediction results"
        ]
    })

    st.dataframe(
        components,
        use_container_width=True,
        hide_index=True
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
