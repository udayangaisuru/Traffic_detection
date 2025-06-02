#include <Arduino.h>

#define VEH_GREEN 13   
#define VEH_YELLOW 12
#define VEH_RED 14 
#define PED_RED 15   
#define PED_GREEN 16 

void setup() {
  // Initialize pins as outputs
  pinMode(VEH_GREEN, OUTPUT);
  pinMode(VEH_YELLOW, OUTPUT);
  pinMode(VEH_RED, OUTPUT);
  pinMode(PED_RED, OUTPUT);
  pinMode(PED_GREEN, OUTPUT);
  
  // Start serial communication
  Serial.begin(9600);
  delay(1000);  // Wait for serial to initialize
  
  // Set initial state (vehicles stopped, pedestrians stopped)
  reset_lights();
  
  // Debug: Confirm startup
  Serial.println("ESP32 Ready");
}

void reset_lights() {
  digitalWrite(VEH_GREEN, LOW);
  digitalWrite(VEH_YELLOW, LOW);
  digitalWrite(VEH_RED, HIGH);
  digitalWrite(PED_RED, HIGH);
  digitalWrite(PED_GREEN, LOW);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    Serial.print("Received: ");
    Serial.println(command);

    // Parse the command
    int commaIndex = command.indexOf(',');
    if (commaIndex == -1) {
      Serial.println("Invalid command format");
      return;
    }

    String type = command.substring(0, commaIndex);
    int time = command.substring(commaIndex + 1).toInt();
    Serial.print("Type: ");
    Serial.print(type);
    Serial.print(", Time: ");
    Serial.println(time); 

    if (type == "V_G") { 
      digitalWrite(VEH_RED, LOW);
      digitalWrite(VEH_YELLOW, LOW);
      digitalWrite(VEH_GREEN, HIGH);
      digitalWrite(PED_RED, HIGH);
      digitalWrite(PED_GREEN, LOW);
      delay(time * 1000); 
      reset_lights();
    } else if (type == "V_Y") { 
      digitalWrite(VEH_RED, LOW);
      digitalWrite(VEH_YELLOW, HIGH);
      digitalWrite(VEH_GREEN, LOW);
      digitalWrite(PED_RED, HIGH);
      digitalWrite(PED_GREEN, LOW);
      delay(time * 1000); 
      reset_lights();
    } else if (type == "P_G") {  
      digitalWrite(VEH_RED, HIGH); 
      digitalWrite(VEH_YELLOW, LOW);
      digitalWrite(VEH_GREEN, LOW);
      digitalWrite(PED_RED, LOW);
      digitalWrite(PED_GREEN, HIGH);  
      delay(time * 1000);  
      reset_lights();
    } else {
      Serial.println("Unknown command");
    }
  }
}