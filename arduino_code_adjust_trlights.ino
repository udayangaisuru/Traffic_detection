#include <Arduino.h>

const int VEH_GREEN  = 13;
const int VEH_YELLOW = 12;
const int VEH_RED    = 14;
const int PED_RED    = 15;
const int PED_GREEN  = 16;

void setup() {
    pinMode(VEH_GREEN, OUTPUT);
    pinMode(VEH_YELLOW, OUTPUT);
    pinMode(VEH_RED, OUTPUT);
    pinMode(PED_RED, OUTPUT);
    pinMode(PED_GREEN, OUTPUT);
    Serial.begin(9600);
    delay(1000);
    Serial.println("READY");
}

void reset_lights() {
    digitalWrite(VEH_GREEN, LOW);
    digitalWrite(VEH_YELLOW, LOW);
    digitalWrite(VEH_RED, HIGH);
    digitalWrite(PED_RED, HIGH);
    digitalWrite(PED_GREEN, LOW);
}

void loop() {
    static unsigned long startTime = 0;
    static unsigned long duration  = 0;
    static bool isTiming          = false;
    static String currentType     = "";
    static unsigned long lastCommandTime = millis();
    const unsigned long TIMEOUT = 60000;

    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd=="INIT") { Serial.println("READY"); return; }
        if (cmd=="PING") { Serial.println("PONG"); lastCommandTime=millis(); return; }
        int comma = cmd.indexOf(',');
        if (comma<0) { Serial.println("Invalid"); return; }
        String type = cmd.substring(0,comma);
        int t = cmd.substring(comma+1).toInt();
        if (t<=0) { Serial.println("Invalid time"); return; }

        duration = t*1000;
        startTime = millis();
        isTiming = true;
        currentType = type;
        lastCommandTime = millis();

        if (type=="V_G") {
            digitalWrite(VEH_RED, LOW);
            digitalWrite(VEH_YELLOW, LOW);
            digitalWrite(VEH_GREEN, HIGH);
            digitalWrite(PED_RED, HIGH);
            digitalWrite(PED_GREEN, LOW);
            Serial.println("OK:V_G");

        } else if (type=="V_Y") {
            digitalWrite(VEH_RED, LOW);
            digitalWrite(VEH_YELLOW, HIGH);
            digitalWrite(VEH_GREEN, LOW);
            digitalWrite(PED_RED, HIGH);
            digitalWrite(PED_GREEN, LOW);
            Serial.println("OK:V_Y");

        } else if (type=="P_G") {
            digitalWrite(VEH_RED, HIGH);
            digitalWrite(VEH_YELLOW, LOW);
            digitalWrite(VEH_GREEN, LOW);
            digitalWrite(PED_RED, LOW);
            digitalWrite(PED_GREEN, HIGH);
            Serial.println("OK:P_G");

        } else {
            Serial.println("ERROR:Unknown");
            isTiming = false;
        }
    }

    // Safety timeout
    if (!isTiming && (millis()-lastCommandTime>TIMEOUT)) {
        reset_lights();
        Serial.println("Timeout:Safe");
    }

    // Phase‐end transitions
    if (isTiming && (millis()-startTime>=duration)) {
        isTiming = false;
        if (currentType=="P_G") {
            // after pedestrian green → yellow
            digitalWrite(VEH_RED, LOW);
            digitalWrite(VEH_YELLOW, HIGH);
            digitalWrite(VEH_GREEN, LOW);
            digitalWrite(PED_RED, HIGH);
            digitalWrite(PED_GREEN, LOW);
            Serial.println("OK:V_Y");
            startTime=millis();
            duration=5000;
            isTiming=true;
            currentType="V_Y";

        } else if (currentType=="V_Y") {
            // yellow → vehicle green
            digitalWrite(VEH_RED, LOW);
            digitalWrite(VEH_YELLOW, LOW);
            digitalWrite(VEH_GREEN, HIGH);
            digitalWrite(PED_RED, HIGH);
            digitalWrite(PED_GREEN, LOW);
            Serial.println("OK:V_G");
            isTiming=false;
        }
    }
}
