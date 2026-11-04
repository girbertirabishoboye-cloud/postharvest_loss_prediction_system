/*
  Post-harvest monitoring node — ESP8266 NodeMCU
  Sensors: DHT11 (temperature/RH), YL-69 analog output (soil moisture)
  Outputs: buzzer, relay-controlled low-voltage DC fan, auxiliary relay.
  MQTT telemetry: postharvest/device01/telemetry
  MQTT control:   postharvest/device01/control

  SAFETY: Never connect a fan, pump, or other load directly to a GPIO.
  Use a suitable relay module/driver and separate correctly-rated supply.
  This classroom sketch is intended for low-voltage DC loads. Do not wire
  mains voltage on a breadboard; mains work requires qualified supervision.
*/
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_HOST = "YOUR_MQTT_BROKER_HOST";
const uint16_t MQTT_PORT = 1883; // Use TLS-capable configuration for public networks.
const char* MQTT_USER = "";      // Optional; set broker credentials if required.
const char* MQTT_PASSWORD = "";

const char* TELEMETRY_TOPIC = "postharvest/device01/telemetry";
const char* CONTROL_TOPIC = "postharvest/device01/control";

#define DHT_PIN D4
#define DHT_TYPE DHT11
#define SOIL_PIN A0
#define BUZZER_PIN D5
#define FAN_RELAY_PIN D1
#define AUX_RELAY_PIN D2

// Many relay boards are active LOW. Change these if your board is active HIGH.
const uint8_t RELAY_ON = LOW;
const uint8_t RELAY_OFF = HIGH;

// Calibrate these two raw ADC values for your actual YL-69 module and soil.
// Read dry-air/dry-soil and wet-soil values; never assume these defaults are calibrated.
const int SOIL_DRY_RAW = 850;
const int SOIL_WET_RAW = 350;

const unsigned long SENSOR_INTERVAL_MS = 10000;
const unsigned long WIFI_RETRY_MS = 10000;
const unsigned long MQTT_RETRY_MS = 5000;

DHT dht(DHT_PIN, DHT_TYPE);
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
unsigned long lastSensorRead = 0;
unsigned long lastWifiAttempt = 0;
unsigned long lastMqttAttempt = 0;

float latestTemp = NAN, latestHumidity = NAN, latestSoil = NAN;

float soilPercent(int raw) {
  // Most capacitive/resistive module outputs decrease as wetness increases.
  // Clamp and map the calibrated dry/wet endpoints to 0..100%.
  float pct = 100.0f * (SOIL_DRY_RAW - raw) / (float)(SOIL_DRY_RAW - SOIL_WET_RAW);
  return constrain(pct, 0.0f, 100.0f);
}

void setRelay(uint8_t pin, bool on) {
  digitalWrite(pin, on ? RELAY_ON : RELAY_OFF);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) != CONTROL_TOPIC) return;
  StaticJsonDocument<192> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) return;
  if (doc.containsKey("fan")) setRelay(FAN_RELAY_PIN, doc["fan"].as<bool>());
  if (doc.containsKey("aux")) setRelay(AUX_RELAY_PIN, doc["aux"].as<bool>());
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  if (millis() - lastWifiAttempt < WIFI_RETRY_MS) return;
  lastWifiAttempt = millis();
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void connectMQTT() {
  if (mqtt.connected() || WiFi.status() != WL_CONNECTED) return;
  if (millis() - lastMqttAttempt < MQTT_RETRY_MS) return;
  lastMqttAttempt = millis();
  String clientId = "postharvest-esp8266-" + String(ESP.getChipId(), HEX);
  bool connected;
  if (strlen(MQTT_USER) > 0) connected = mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD);
  else connected = mqtt.connect(clientId.c_str());
  if (connected) mqtt.subscribe(CONTROL_TOPIC);
}

void publishTelemetry() {
  if (!mqtt.connected() || isnan(latestTemp) || isnan(latestHumidity)) return;
  StaticJsonDocument<192> doc;
  doc["temperature_c"] = latestTemp;
  doc["relative_humidity_pct"] = latestHumidity;
  doc["soil_moisture_pct"] = latestSoil;
  doc["device_id"] = "device01";
  doc["uptime_ms"] = millis();
  char payload[192];
  size_t n = serializeJson(doc, payload, sizeof(payload));
  mqtt.publish(TELEMETRY_TOPIC, payload, n);
}

void readSensors() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  int raw = analogRead(SOIL_PIN);
  if (!isnan(t) && !isnan(h)) {
    latestTemp = t;
    latestHumidity = h;
    latestSoil = soilPercent(raw);
    publishTelemetry();

    // Simple local warning logic for demonstration. Validate thresholds per crop.
    bool alert = (latestTemp >= 36.0f) || (latestHumidity >= 90.0f) ||
                 (latestTemp >= 32.0f && latestHumidity >= 75.0f);
    digitalWrite(BUZZER_PIN, alert ? HIGH : LOW);
    Serial.printf("T=%.1f C RH=%.1f%% Soil=%.1f%% raw=%d alert=%d\n",
                  latestTemp, latestHumidity, latestSoil, raw, alert);
  } else {
    Serial.println("DHT read failed; check sensor wiring.");
    digitalWrite(BUZZER_PIN, LOW);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(FAN_RELAY_PIN, OUTPUT);
  pinMode(AUX_RELAY_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  setRelay(FAN_RELAY_PIN, false);
  setRelay(AUX_RELAY_PIN, false);
  dht.begin();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void loop() {
  connectWiFi();
  connectMQTT();
  if (mqtt.connected()) mqtt.loop();
  if (millis() - lastSensorRead >= SENSOR_INTERVAL_MS) {
    lastSensorRead = millis();
    readSensors();
  }
  delay(5);
}
