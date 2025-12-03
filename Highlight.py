import easyocr
import cv2
import numpy as np

# ==================== CONFIGURATION ====================
IMAGE_PATH = "clean_screen.jpeg" 

# 1. BOX SETTINGS
BOX_WIDTH = 1125 
Y_PADDING = 10 

# 2. CUSTOM SHIFTS
# Your custom alignment numbers
PER_LABEL_SHIFTS = {
    "Opportunity Name": -300,      
    "Opportunity Record": -300,    
    "Type": -100,                  
    "Stage": -100,
    "Close Date": -150
}

TARGET_LABELS = [
    "Opportunity Name", 
    "Opportunity Record", 
    "Type",
    "Stage", 
    "Close Date"
]
# =======================================================

def highlight_fields_final():
    print("1. Loading AI Reader...")
    reader = easyocr.Reader(['en']) 
    
    print(f"2. Loading Image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("ERROR: Image not found. Check filename.")
        return

    print("3. Scanning image...")
    results = reader.readtext(img, paragraph=False) 

    # --- STEP 1: FIND THE ANCHOR (Opportunity Record) ---
    # We need to know where "Record" is so we can filter "Type" relative to it.
    record_y_location = -9999 # Default to impossible number
    
    for (bbox, text, prob) in results:
        # Check for the Record label
        if "Opportunity Record" in text or ("Record" in text and "Type" in text):
            (_, tr, _, _) = bbox
            record_y_location = tr[1] # Save the Y-coordinate
            print(f"   [Anchor Found] 'Opportunity Record' detected at Y={record_y_location}")
            # We don't break here because we want to draw the box for it later
            
    # --- STEP 2: DRAW BOXES & FILTER DUPLICATES ---
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        current_y = tr[1]
        
        matched_label = None
        for target in TARGET_LABELS:
            clean_text = text.lower().strip()
            
            if target.lower() in clean_text:
                
                # === THE ULTIMATE DOUBLE BOX FIX ===
                if target == "Type":
                    # 1. Check strict word containment
                    # If the line says "Opportunity Record Type", do NOT treat it as "Type"
                    if "record" in clean_text:
                        continue 

                    # 2. Check Distance (Proximity)
                    # Calculate vertical distance between this "Type" and "Opportunity Record"
                    distance = abs(current_y - record_y_location)
                    
                    # If it's within 150 pixels, it is definitely the Duplicate. SKIP IT.
                    if distance < 150:
                        print(f"   [Skipping] Found 'Type' but it's too close to Record (Dist: {distance})")
                        continue 
                # ===================================

                matched_label = target
                break
        
        if matched_label:
            print(f"   -> Drawing Box: '{text}'")
            
            # Apply Shifts
            shift_val = PER_LABEL_SHIFTS.get(matched_label, -10)
            
            # Calculate Box
            x_start = int(tr[0]) + shift_val
            x_end = x_start + BOX_WIDTH
            y_start = int(tr[1]) - Y_PADDING
            y_end = int(br[1]) + Y_PADDING

            # Tall Box Rule for Record Type
            if "Record" in text or "Opportunity Record" in text:
                y_start = y_start - 25
                y_end = y_end + 25

            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

    output_filename = "final_result_fixed.jpg"
    cv2.imwrite(output_filename, img)
    print(f"\nSUCCESS! Image saved as '{output_filename}'")

if __name__ == "__main__":
    highlight_fields_final()