#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SPI.h>
#include <MFRC522.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Arduino.h>
#include <time.h>  // Added for time functions


// ---------- Pins ----------
#define SS_PIN     15
#define RST_PIN    16
#define SCK_PIN    17
#define MISO_PIN   21
#define MOSI_PIN   18
#define LCD_SDA    47
#define LCD_SCL    46
#define BUZZER_PIN 41  // Passive buzzer pin (PWM capable recommended)
#define LED_PIN 35
// ---------- WiFi ----------
const char* ssid     = "white_house_wifi";
const char* password = "radiation-waves:)$153$";


// ---------- Web App ----------
String Web_App_URL = "https://script.google.com/macros/s/AKfycbxrgvL0cvscMXECukZcuiTpO2j05yhu1w2jBNPCPJIJjkL7X2TMQKBeEIqcXBRR7Z5S/exec";


// ---------- Time Settings ----------
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = -21600;  // Chicago is UTC-6 (Central Time)
const int daylightOffset_sec = 3600; // Daylight saving time offset


// ---------- Globals ----------
LiquidCrystal_I2C lcd(0x27, 20, 4);
MFRC522 mfrc522(SS_PIN, RST_PIN);
char strbuf[32] = "";


// ---------- Get Current Date and Time ----------
String getCurrentDateTime() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return "Time not available";
  }
 
  char dateTimeStr[32];
  strftime(dateTimeStr, sizeof(dateTimeStr), "%m/%d/%Y %H:%M:%S", &timeinfo);
  return String(dateTimeStr);
}


String getCurrentDate() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return "Date not available";
  }
 
  char dateStr[16];
  strftime(dateStr, sizeof(dateStr), "%m/%d/%Y", &timeinfo);
  return String(dateStr);
}


String getCurrentTime() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return "Time not available";
  }
 
  char timeStr[16];
  strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &timeinfo);
  return String(timeStr);
}


// ---------- Helpers ----------
String getValue(String data, char sep, int index) {
  int found = 0, strIndex[] = { 0, -1 }, maxIndex = data.length() - 1;
  for (int i = 0; i <= maxIndex && found <= index; i++) {
    if (data.charAt(i) == sep || i == maxIndex) {
      found++; strIndex[0] = strIndex[1] + 1; strIndex[1] = (i == maxIndex) ? i + 1 : i;
    }
  }
  return (found > index) ? data.substring(strIndex[0], strIndex[1]) : "";
}


void byteArray_to_string(byte array[], unsigned int len, char buffer[]) {
  for (unsigned int i = 0; i < len; i++) {
    byte hi = (array[i] >> 4) & 0x0F, lo = array[i] & 0x0F;
    buffer[i * 2 + 0] = hi < 0xA ? '0' + hi : 'A' + hi - 0xA;
    buffer[i * 2 + 1] = lo < 0xA ? '0' + lo : 'A' + lo - 0xA;
  }
  buffer[len * 2] = '\0';
}


bool readUID(String &uid) {
  if (!mfrc522.PICC_IsNewCardPresent()) return false;
  if (!mfrc522.PICC_ReadCardSerial())   return false;
  byteArray_to_string(mfrc522.uid.uidByte, mfrc522.uid.size, strbuf);
  uid = strbuf;
  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
  return true;
}


// ---------- Passive Buzzer Tone ----------
void buzzTone(uint16_t freq = 2000, uint16_t duration = 200) {
  ledcAttach(BUZZER_PIN, freq, 8);  // Updated function name, 8-bit resolution
  ledcWrite(BUZZER_PIN, 128);       // 50% duty cycle (128 out of 255)
  delay(duration);
  ledcWrite(BUZZER_PIN, 0);         // stop tone
  ledcDetach(BUZZER_PIN);           // Updated function name
}


// ---------- HTTP Attendance ----------
void http_Attendance(const String& uid) {
  if (WiFi.status() != WL_CONNECTED) {
    lcd.clear(); lcd.setCursor(0, 0); lcd.print("WiFi disconnected");
    delay(1500); return;
  }


  String url = Web_App_URL + "?sts=atc&uid=" + uid;
  HTTPClient http;
  http.begin(url.c_str());
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  int code = http.GET();
  String payload = (code > 0) ? http.getString() : "";
  http.end();


  String sts = getValue(payload, ',', 0);
  if (sts != "OK") return;


  String info = getValue(payload, ',', 1);


  if (info == "CI_Successful") {
    String name = getValue(payload, ',', 2);
    String date = getValue(payload, ',', 3);
    String timeIn = getValue(payload, ',', 4);


    int pos = 0, L = name.length();
    if (L > 20) name = name.substring(0, 20);
    if (L > 0 && L <= 20) {
      pos = ((20 / 2) - 1) - map(L, 1, 20, 0, (20 / 2) - 1);
    }


    lcd.clear(); delay(100);
    lcd.setCursor(pos, 0); lcd.print(name);
    lcd.setCursor(0, 1); lcd.print("CHECKED IN");
    lcd.setCursor(0, 2); lcd.print("Date: "); lcd.print(date);
    lcd.setCursor(0, 3); lcd.print("Time: "); lcd.print(timeIn);
    delay(4000); lcd.clear();
  }
  else if (info == "CO_Successful") {
    String name = getValue(payload, ',', 2);
    String date = getValue(payload, ',', 3);
    String timeIn = getValue(payload, ',', 4);
    String timeOut = getValue(payload, ',', 5);


    int pos = 0, L = name.length();
    if (L > 20) name = name.substring(0, 20);
    if (L > 0 && L <= 20) {
      pos = ((20 / 2) - 1) - map(L, 1, 20, 0, (20 / 2) - 1);
    }


    lcd.clear(); delay(100);
    lcd.setCursor(pos, 0); lcd.print(name);
    lcd.setCursor(0, 1); lcd.print("CHECKED OUT");
    lcd.setCursor(0, 2); lcd.print("In: "); lcd.print(timeIn);
    lcd.setCursor(0, 3); lcd.print("Out: "); lcd.print(timeOut);
    delay(4000); lcd.clear();
  }
  else if (info == "CO_Updated") {
    String name = getValue(payload, ',', 2);
    String date = getValue(payload, ',', 3);
    String timeIn = getValue(payload, ',', 4);
    String timeOut = getValue(payload, ',', 5);


    int pos = 0, L = name.length();
    if (L > 20) name = name.substring(0, 20);
    if (L > 0 && L <= 20) {
      pos = ((20 / 2) - 1) - map(L, 1, 20, 0, (20 / 2) - 1);
    }


    lcd.clear(); delay(100);
    lcd.setCursor(pos, 0); lcd.print(name);
    lcd.setCursor(0, 1); lcd.print("CHECKOUT UPDATED");
    lcd.setCursor(0, 2); lcd.print("In: "); lcd.print(timeIn);
    lcd.setCursor(0, 3); lcd.print("Out: "); lcd.print(timeOut);
    delay(4000); lcd.clear();
  }
  else if (info == "atcErr01") {
    lcd.clear();
    lcd.setCursor(2, 0); lcd.print("UID not registered");
    delay(2000); lcd.clear();
  }
}


// ---------- Setup ----------
void setup() {
  pinMode(LED_PIN, OUTPUT);
digitalWrite(LED_PIN, LOW); // start with LED off
  Serial.begin(115200); delay(200);


  pinMode(BUZZER_PIN, OUTPUT);  // needed for passive buzzers
  digitalWrite(BUZZER_PIN, LOW);


  Wire.begin(LCD_SDA, LCD_SCL);
  lcd.init(); lcd.backlight(); lcd.clear();


  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN);
  mfrc522.PCD_Init();


  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(250);
  }


  // Initialize and configure time
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
 
  lcd.setCursor(0, 0); lcd.print("ATTENDANCE MODE");
  delay(1000); lcd.clear();
}


// ---------- Loop ----------
void loop() {
 
  lcd.setCursor(3, 0); lcd.print("ATTENDANCE");
  lcd.setCursor(0, 2); lcd.print("Tap your card...");


  String uid;
  if (!readUID(uid)) {
    delay(50);
    return;
  }


  buzzTone();  // beep on tag scan
  digitalWrite(LED_PIN, HIGH); // turn LED on


  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(getCurrentDate());
  lcd.setCursor(0, 1); lcd.print(getCurrentTime());
  //lcd.setCursor(0, 2); lcd.print(getCurrentTime());
  delay(300);


  http_Attendance(uid);
  // lcd.clear();
  digitalWrite(LED_PIN, LOW);
}
