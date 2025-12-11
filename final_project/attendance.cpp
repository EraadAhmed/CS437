#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SPI.h>
#include <esp_task_wdt.h>
#include <MFRC522.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Arduino.h>
#include <time.h> 
#include "time.h"

// ---------- Pins ----------
#define SS_PIN     15
#define RST_PIN    16
#define SCK_PIN    17
#define MISO_PIN   21
#define MOSI_PIN   18
#define LCD_SDA    47
#define LCD_SCL    46
#define BUZZER_PIN 41 
#define LED_PIN    35

// ---------- WiFi ----------
const char* ssid     = "Unit #301";
const char* password = "exact change";

// ---------- Web App ----------
String Web_App_URL = "https://script.google.com/macros/s/AKfycbxrgvL0cvscMXECukZcuiTpO2j05yhu1w2jBNPCPJIJjkL7X2TMQKBeEIqcXBRR7Z5S/exec";

// ---------- Time Settings ----------
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = -21600;  
const int daylightOffset_sec = 3600; 

// Timer variables
unsigned long lastActivityTime = 0;
const unsigned long INACTIVITY_TIMEOUT = 10 * 60 * 1000; 
bool isLowPowerMode = false;

// ---------- Globals ----------
LiquidCrystal_I2C lcd(0x27, 16, 2); 
MFRC522 mfrc522(SS_PIN, RST_PIN);
char strbuf[32] = "";
bool screenNeedsUpdate = true; 


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

void buzzTone(uint16_t freq = 2000, uint16_t duration = 200) {
  ledcAttach(BUZZER_PIN, freq, 8);  
  ledcWrite(BUZZER_PIN, 128);       
  delay(duration);
  ledcWrite(BUZZER_PIN, 0);         
  ledcDetach(BUZZER_PIN);           
}

// ---------- HTTP Attendance ----------
void http_Attendance(const String& uid) {
  if (WiFi.status() != WL_CONNECTED) {
    lcd.clear(); lcd.setCursor(0, 0); lcd.print("WiFi Disconn.");
    delay(2000); return;
  }

  // Prevent Watchdog Crash
  esp_task_wdt_reset();

  String url = Web_App_URL + "?sts=atc&uid=" + uid;
  HTTPClient http;
  
  http.setTimeout(25000); 
  http.begin(url.c_str());
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  
  int code = http.GET();
  String payload = "";

  if (code > 0) {
    payload = http.getString();
  } else {
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("HTTP Err: "); lcd.print(code);
    delay(3000);
    http.end();
    return;
  }
  http.end();

  Serial.println("Payload: " + payload);
  esp_task_wdt_reset(); // Feed again after HTTP return

  String sts = getValue(payload, ',', 0);
  sts.trim();

  if (sts != "OK") {
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("Server Err");
    lcd.setCursor(0, 1); lcd.print(sts);
    delay(2000);
    return;
  }

  String info = getValue(payload, ',', 1);
  info.trim();

  if (info == "CI_Successful" || info == "CO_Successful" || info == "CO_Updated") {
    
    String name = getValue(payload, ',', 2);
    String date = getValue(payload, ',', 3);
    String timeVal = "";
    String line1Prefix = "";

    if (info == "CI_Successful") {
      line1Prefix = "CHKIN ";      
      timeVal = getValue(payload, ',', 4); 
    } else {
      line1Prefix = "CHKOUT ";     
      timeVal = getValue(payload, ',', 5); 
    }

    if (timeVal.length() == 0) timeVal = getValue(payload, ',', 4);

    name.trim();
    timeVal.trim();
    date.trim();

    int spaceLeft = 16 - line1Prefix.length(); 
    if (name.length() > spaceLeft) {
      name = name.substring(0, spaceLeft); 
    }
    String line1 = line1Prefix + name;
    
    String shortDate = date;
    if (date.length() >= 5) {
      shortDate = date.substring(0, 5);
    }
    
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print(line1);
    lcd.setCursor(0, 1); lcd.print(shortDate + " " + timeVal);
    delay(3000); 
    lcd.clear();
  }
  else if (info == "atcErr01") {
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("ID Not Reg.");
    delay(2000); lcd.clear();
  }
  else {
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("Unknown Resp");
    delay(2000);
  }
  
  esp_task_wdt_reset();
}

void checkInactivity() {
  if (millis() - lastActivityTime > INACTIVITY_TIMEOUT) {
    if (!isLowPowerMode) {
      Serial.println("Entering Low Power Mode.");
      isLowPowerMode = true;
      lcd.noBacklight(); 
      digitalWrite(LED_PIN, LOW); 
    }
    esp_sleep_enable_timer_wakeup(1000 * 1000ULL); 
    esp_light_sleep_start(); 
  } else {
    if (isLowPowerMode) {
      Serial.println("Waking up.");
      isLowPowerMode = false;
      lcd.backlight(); 
    }
  }
}

// ---------- SETUP ----------
void setup() {
  // 1. Configure Watchdog properly (Don't re-init, just add)
  // This increases timeout to 40 seconds to prevent crash
  esp_task_wdt_config_t twdt_config = {
    .timeout_ms = 40000,
    .idle_core_mask = (1 << portNUM_PROCESSORS) - 1,
    .trigger_panic = true
  };
  esp_task_wdt_init(&twdt_config); // Reconfigure existing WDT
  esp_task_wdt_add(NULL);          // Add current thread (loop) to WDT

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 
  
  Serial.begin(115200); 
  delay(1000); 

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Wire.begin(LCD_SDA, LCD_SCL);
  lcd.init(); 
  lcd.backlight(); 
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Booting...");

  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN);
  mfrc522.PCD_Init();
  mfrc522.PCD_SetAntennaGain(mfrc522.RxGain_max); 

  WiFi.mode(WIFI_STA);
  WiFi.setTxPower(WIFI_POWER_8_5dBm); // Low power to prevent brownout
  WiFi.setSleep(false); 
  WiFi.begin(ssid, password);
  
  lcd.setCursor(0, 1); lcd.print("Connecting...");
  
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 30000) {
    delay(250);
    esp_task_wdt_reset(); // Feed watchdog while waiting
  }

  if (WiFi.status() == WL_CONNECTED) {
     lcd.setCursor(0, 1); lcd.print("WiFi OK!      ");
     delay(1000);
  } else {
     lcd.setCursor(0, 1); lcd.print("WiFi Failed   ");
     delay(2000);
  }

  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("ATTENDANCE");
  delay(1000); 
  lcd.clear();
  lastActivityTime = millis(); 
  
  Serial.println("Setup Complete.");
}

// ---------- LOOP ----------
void loop() {
  esp_task_wdt_reset(); // Feed watchdog

  checkInactivity(); 

  if (screenNeedsUpdate) {
    lcd.clear();
    lcd.setCursor(3, 0); lcd.print("ATTENDANCE");
    lcd.setCursor(0, 1); lcd.print("Tap Card...");
    screenNeedsUpdate = false; 
  }

  String uid;
  // REMOVED delay(20) for snappier scanning
  if (!readUID(uid)) {
    return; 
  }

  // === CARD SCANNED ===
  lastActivityTime = millis();
  if (isLowPowerMode) {
    isLowPowerMode = false;
    lcd.backlight();
  }

  buzzTone();
  digitalWrite(LED_PIN, HIGH);

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Processing...");

  http_Attendance(uid);

  digitalWrite(LED_PIN, LOW);
  screenNeedsUpdate = true; 
}