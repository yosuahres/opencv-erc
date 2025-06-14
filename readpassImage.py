import cv2
import numpy as np

# --- 1. Image Pre-processing Function ---
def adjust_gamma(image, gamma=1.0):
    """
    Applies gamma correction to an image.
    A gamma value < 1.0 will make the image darker (useful for overexposed images).
    A gamma value > 1.0 will make the image brighter.
    """
    invGamma = 1.0 / gamma
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# --- 2. Modified classify_pixel function with Tunable HSV Ranges ---
def classify_pixel(bgr_pixel):
    """
    Classifies a BGR pixel into a Morse code dot (.), dash (-), or unclassified (empty string).
    HSV ranges are defined for more precise control, especially for over-bright LEDs.
    """
    hsv = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # --- TUNE THESE HSV THRESHOLDS CAREFULLY BASED ON YOUR PROCESSED IMAGE ---
    # After applying gamma correction, observe the HSV values of your LED colors.
    # Hue (H): 0-179 (Red typically wraps around 0 and 179)
    # Saturation (S): 0-255 (Lower values indicate less pure color, more washed out/white-ish)
    # Value (V): 0-255 (Brightness - high for bright LEDs)

    # Red (Dot) Ranges
    # For over-bright, potentially washed-out red, you might need to lower RED_SAT_MIN
    # and ensure RED_VAL_MIN is high enough to capture the bright LED, but not too high
    # that it misses slightly dimmer parts.
    RED_HUE_MIN1, RED_HUE_MAX1 = 0, 10    # First segment of Red Hue
    RED_HUE_MIN2, RED_HUE_MAX2 = 160, 179 # Second segment of Red Hue
    RED_SAT_MIN, RED_SAT_MAX = 40, 255    # Adjust MIN if red looks too white/pink (e.g., 30-50)
    RED_VAL_MIN, RED_VAL_MAX = 80, 255    # Adjust MIN if LED is still too bright (e.g., 70-90)

    # Blue (Dash) Ranges
    # Similar adjustments for blue saturation and value.
    BLUE_HUE_MIN, BLUE_HUE_MAX = 90, 130 # Standard Blue Hue range
    BLUE_SAT_MIN, BLUE_SAT_MAX = 40, 255  # Adjust MIN if blue looks too white/light-blue (e.g., 30-50)
    BLUE_VAL_MIN, BLUE_VAL_MAX = 80, 255  # Adjust MIN based on brightness (e.g., 70-90)

    # Check for Red pixel
    is_red = ((h >= RED_HUE_MIN1 and h <= RED_HUE_MAX1) or \
              (h >= RED_HUE_MIN2 and h <= RED_HUE_MAX2)) and \
             (s >= RED_SAT_MIN and s <= RED_SAT_MAX) and \
             (v >= RED_VAL_MIN and v <= RED_VAL_MAX)

    if is_red:
        return '.'

    # Check for Blue pixel
    is_blue = (h >= BLUE_HUE_MIN and h <= BLUE_HUE_MAX) and \
              (s >= BLUE_SAT_MIN and s <= BLUE_SAT_MAX) and \
              (v >= BLUE_VAL_MIN and v <= BLUE_VAL_MAX)

    if is_blue:
        return '-'

    # If not red or blue, return an empty string. This will implicitly handle
    # unlit pixels or other colors as separators/background.
    return ''

# --- Your existing extract_morse_sequences function (no changes needed here) ---
def extract_morse_sequences(image):
    """
    Resizes the image to 16x16 and extracts Morse code sequences.
    Each 16x16 image is treated as 16 rows, with each row containing two
    8-pixel "words" (Morse code sequences), read left-to-right, top-to-bottom.
    """
    resized = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    
    morse_sequences = [] 

    for row in range(16):
        row_data = resized[row]  
        
        for word_idx in range(2):
            start_pixel = word_idx * 8  
            current_morse_sequence = ''
            
            for i in range(8):
                pixel = row_data[start_pixel + i] 
                current_morse_sequence += classify_pixel(pixel)
            
            morse_sequences.append(current_morse_sequence.strip())
            
    return morse_sequences

# --- Your existing Morse Code Dictionary (no changes needed here) ---
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '---...': ':', '..--..': '?',
    '-....-': '-', '.-..-.': '"', '-.-.--': '!', '.-.-.': '+', 
    '.-..-.': "'", '---.': '(', '-.--.-': ')', '.-...': '&',
    '--.-.': '@', '...-.-': '$', '.-.-..': '_', '..--.-': '/',
    '...---...': 'SOS' 
}

# --- Your existing morse_to_text function (no changes needed here) ---
def morse_to_text(morse_code_sequence):
    """
    Translates a Morse code sequence (e.g., ".-") into its corresponding character.
    Returns '?' if the sequence is not found in the dictionary, indicating an unrecognized pattern.
    """
    return MORSE_CODE_DICT.get(morse_code_sequence, '?')

# --- Modified main function ---
def main():
    image_path = "test5.png"  # Path to your input image
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Cannot load image at: {image_path}")
        print("Please ensure the image path is correct and the file exists.")
        return

    # --- Step 1: Image Pre-processing for Brightness ---
    # Apply gamma correction to darken the image if LEDs are over-bright.
    # You MUST TUNE the 'gamma' value:
    #   - Start with 0.5 or 0.7.
    #   - Lower it (e.g., to 0.4, 0.3) if the image is still too bright.
    #   - Raise it if the image becomes too dark.
    processed_image = adjust_gamma(image, gamma=0.4) # <-- TUNE THIS GAMMA VALUE!

    # Optional: Apply a slight Gaussian blur if there's significant pixel bleeding/noise
    # processed_image = cv2.GaussianBlur(processed_image, (3, 3), 0)

    # Step 2: Extract Morse code sequences from the processed image
    morse_sequences = extract_morse_sequences(processed_image)

    print("\n📷 Extracted Morse Code Sequences (Red = dot, Blue = dash):")
    translated_characters = [] # To build the final translated message

    # Step 3: Process and translate each extracted Morse sequence
    for i, morse_seq in enumerate(morse_sequences):
        translated_char = morse_to_text(morse_seq)
        
        # Print the original Morse sequence and its translation for debugging/verification
        print(f"{i+1:02}: Morse: '{morse_seq.ljust(8)}' -> Text: '{translated_char}'")
        
        # Build the final translated message:
        # Treat empty sequences (all unlit pixels in an 8-pixel segment) as spaces.
        # Otherwise, append the translated character if recognized, or a placeholder '?'
        if morse_seq == '': 
            translated_characters.append(' ') 
        elif translated_char != '?':
            translated_characters.append(translated_char)
        else:
            # If you want to include '?' for unrecognized sequences in the final text:
            # translated_characters.append('?')
            pass # Or simply ignore unrecognized sequences

    print("\n📝 Translated Message:")
    # Join all translated characters to form the final message.
    print("".join(translated_characters).strip()) # .strip() removes leading/trailing spaces

    # Optional: Visualize the resized (and processed) 16x16 image
    # This helps you see the effect of gamma correction and judge color clarity.
    tiny = cv2.resize(processed_image, (160, 160), interpolation=cv2.INTER_NEAREST)
    window_name = "Resized 16x16 View (Processed Image) - Press 'q' or close window to quit"
    cv2.imshow(window_name, tiny)
    
    # Keep the window open until 'q' is pressed or the window is closed manually
    while True:
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows() # Close all OpenCV windows when done

if __name__ == '__main__':
    main()