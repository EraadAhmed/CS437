#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SPI.h>
#include <MFRC522.h>
#include <WiFi.h>
#include <HTTPClient.h>


// ---------- CORRECT PINS for Waveshare ESP32-S3-ETH ----------
// Based on your board image - these pins are confirmed available!


// MFRC522 RFID Module connections
#define SS_PIN    15   // CS/SDA  → GPIO15 (Pin 29 on left side)
#define RST_PIN   16   // RST     → GPIO16 (Pin 32 on left side)
#define SCK_PIN   17   // SCK     → GPIO17 (Pin 34 on left side)
#define MISO_PIN  21   // MISO    → GPIO21 (Pin 35 on left side)
#define MOSI_PIN  18   // MOSI    → GPIO18 (Pin 31 on left side)


// I2C LCD connections (using GPIO19/20 as requested)
#define LCD_SDA   47   // SDA     → GPIO20 (Pin 1 on right side)
#define LCD_SCL   46   // SCL     → GPIO19 (Pin 2 on right side)


// ---------- WiFi Credentials ----------
const char* ssid     = "white_house_wifi";
const char* password = "radiation-waves:)$153$";


// ---------- Google Apps Script URL ----------
String Web_App_URL = "https://script.google.com/macros/s/AKfycbxrgvL0cvscMXECukZcuiTpO2j05yhu1w2jBNPCPJIJjkL7X2TMQKBeEIqcXBRR7Z5S/exec";


// ---------- Global Objects ----------
LiquidCrystal_I2C lcd(0x27, 20, 4);  // 20x4 LCD at I2C address 0x27
MFRC522 mfrc522(SS_PIN, RST_PIN);    // MFRC522 instance
char strbuf[32] = "";                 // Buffer for UID string


// ---------- Helper Functions ----------
String getValue(String data, char sep, int index) {
  int found = 0, strIndex[] = { 0, -1 }, maxIndex = data.length() - 1;
  for (int i = 0; i <= maxIndex && found <= index; i++) {
    if (data.charAt(i) == sep || i == maxIndex) {
      found++;
      strIndex[0] = strIndex[1] + 1;
      strIndex[1] = (i == maxIndex) ? i + 1 : i;
    }
  }
  return (found > index) ? data.substring(strIndex[0], strIndex[1]) : "";
}


void byteArray_to_string(byte array[], unsigned int len, char buffer[]) {
  for (unsigned int i = 0; i < len; i++) {
    byte hi = (array[i] >> 4) & 0x0F;
    byte lo = array[i] & 0x0F;
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


// ---------- HTTP Registration Function ----------
void http_Register(const String& uid) {
  if (WiFi.status() != WL_CONNECTED) {
    lcd.clear();
    lcd.setCursor(0,0); lcd.print("WiFi disconnected");
    Serial.println("ERROR: WiFi disconnected");
    delay(1500);
    return;
  }


  Serial.print("Sending registration for UID: ");
  Serial.println(uid);


  String url = Web_App_URL + "?sts=reg&uid=" + uid;
  HTTPClient http;
  http.begin(url.c_str());
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
 
  int code = http.GET();
  String payload = (code > 0) ? http.getString() : "";
  http.end();


  Serial.print("HTTP Response Code: ");
  Serial.println(code);
  Serial.print("HTTP Payload: ");
  Serial.println(payload);


  String sts = getValue(payload, ',', 0);
  String info = getValue(payload, ',', 1);


  lcd.clear();
  if (sts == "OK" && info == "R_Successful") {
    lcd.setCursor(0,0); lcd.print("SUCCESS!");
    lcd.setCursor(0,1); lcd.print("UID registered");
    Serial.println("SUCCESS: UID registered successfully");
  } else if (sts == "OK" && info == "regErr01") {
    lcd.setCursor(0,0); lcd.print("ALREADY EXISTS");
    lcd.setCursor(0,1); lcd.print("UID already in DB");
    Serial.println("INFO: UID already exists in database");
  } else {
    lcd.setCursor(0,0); lcd.print("REGISTRATION ERROR");
    lcd.setCursor(0,1); lcd.print("Code: " + String(code));
    Serial.println("ERROR: Registration failed");
  }
 
  delay(3000);
  lcd.clear();
}


// ---------- Setup Function ----------
void setup() {
  Serial.begin(115200);
  delay(1000);
 
  Serial.println("=== ESP32-S3-ETH RFID Registration System ===");
  Serial.println("Initializing components...");
 
  // Initialize I2C LCD
  Serial.println("Initializing LCD...");
  Wire.begin(LCD_SDA, LCD_SCL);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0); lcd.print("Initializing...");
  lcd.setCursor(0,1); lcd.print("LCD: OK");
  Serial.println("✓ LCD initialized successfully");
 
  // Initialize SPI for MFRC522
  Serial.println("Initializing SPI and MFRC522...");
  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN);
  mfrc522.PCD_Init();
  delay(100);
 
  // Test MFRC522
  bool rfidOK = mfrc522.PCD_PerformSelfTest();
  Serial.print("MFRC522 Self-Test: ");
  Serial.println(rfidOK ? "PASSED ✓" : "FAILED ✗");
 
  if (!rfidOK) {
    lcd.setCursor(0,2); lcd.print("RFID: FAILED!");
    lcd.setCursor(0,3); lcd.print("Check wiring");
    Serial.println("ERROR: MFRC522 failed self-test!");
    Serial.println("Check your wiring connections:");
    Serial.println("- VCC → 3.3V (Pin 36)");
    Serial.println("- GND → GND (Pin 33/38)");
    Serial.println("- RST → GPIO16 (Pin 32)");
    Serial.println("- CS  → GPIO15 (Pin 29)");
    Serial.println("- SCK → GPIO17 (Pin 34)");
    Serial.println("- MOSI→ GPIO18 (Pin 31)");
    Serial.println("- MISO→ GPIO21 (Pin 35)");
    // Continue anyway for debugging
  } else {
    lcd.setCursor(0,2); lcd.print("RFID: OK");
    Serial.println("✓ MFRC522 initialized successfully");
  }
 
  // Re-initialize MFRC522 after self-test
  mfrc522.PCD_Init();
 
  // Initialize WiFi
  Serial.println("Connecting to WiFi...");
  lcd.setCursor(0,3); lcd.print("WiFi: Connecting");
 
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
 
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
 
  if (WiFi.status() == WL_CONNECTED) {
    lcd.setCursor(0,3); lcd.print("WiFi: Connected ");
    Serial.println("✓ WiFi connected successfully");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    lcd.setCursor(0,3); lcd.print("WiFi: FAILED    ");
    Serial.println("✗ WiFi connection failed");
  }
 
  delay(3000);
  lcd.clear();
  lcd.setCursor(0,0); lcd.print("REGISTRATION MODE");
  lcd.setCursor(0,1); lcd.print("System Ready");
  Serial.println("=== System Ready - Registration Mode Active ===");
  delay(2000);
  lcd.clear();
}


// ---------- Main Loop ----------
void loop() {
  // Display ready message
  lcd.setCursor(2,0); lcd.print("REGISTRATION");
  lcd.setCursor(6,1); lcd.print("SYSTEM");
  lcd.setCursor(0,2); lcd.print("Tap your card...");
  lcd.setCursor(0,3); lcd.print("Waiting for card...");


  String uid;
  if (!readUID(uid)) {
    delay(100);
    return;
  }


  // Card detected!
  Serial.println("=== CARD DETECTED ===");
  lcd.clear();
  lcd.setCursor(0,0); lcd.print("CARD DETECTED!");
  lcd.setCursor(0,1); lcd.print("Reading UID...");
 
  // Display UID
  lcd.setCursor(0,2);
  if (uid.length() <= 20) {
    lcd.print(uid);
  } else {
    lcd.print(uid.substring(0, 20));
    lcd.setCursor(0,3);
    lcd.print(uid.substring(20));
  }
 
  Serial.print("UID: ");
  Serial.println(uid);
  delay(1500);


  // Send to Google Sheets
  lcd.clear();
  lcd.setCursor(0,0); lcd.print("REGISTERING...");
  lcd.setCursor(0,1); lcd.print("Please wait...");
 
  http_Register(uid);
 
  // Prevent immediate re-reading
  delay(2000);
}
