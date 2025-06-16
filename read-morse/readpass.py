import cv2
import numpy as np
import time

def adjust_gamma(image, gamma=1.0):
    """
    Applies gamma correction to an image.
    A gamma value < 1.0 makes the image darker (useful for overexposed images).
    A gamma value > 1.0 makes the image brighter.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def classify_pixel(bgr_pixel):
    """
    Classifies a BGR pixel into a Morse code dot (.), dash (-), or unclassified (empty string).
    HSV ranges are defined for more precise control, especially for over-bright LEDs.
    """
    hsv = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # Hue (H): 0-179 (Red typically wraps around 0 and 179).
    # Saturation (S): 0-255 (Lower values mean less pure color, more washed out/white-ish).
    # Value (V): 0-255 (Brightness - high for bright LEDs).

    # perlu tuning
    # If red looks too white/pink, try lowering RED_SAT_MIN (e.g., from 40 to 30 or 20).
    # If background noise is picked up as red, try raising RED_VAL_MIN.
    RED_HUE_MIN1, RED_HUE_MAX1 = 0, 10    # First segment of Red Hue
    RED_HUE_MIN2, RED_HUE_MAX2 = 160, 179 # Second segment of Red Hue
    RED_SAT_MIN, RED_SAT_MAX = 40, 255    
    RED_VAL_MIN, RED_VAL_MAX = 80, 255    

    # perlu tuning
    # If blue looks too white/light-blue, try lowering BLUE_SAT_MIN.
    # If background noise is picked up as blue, try raising BLUE_VAL_MIN.
    BLUE_HUE_MIN, BLUE_HUE_MAX = 90, 130 
    BLUE_SAT_MIN, BLUE_SAT_MAX = 40, 255  
    BLUE_VAL_MIN, BLUE_VAL_MAX = 80, 255  

    is_red = ((h >= RED_HUE_MIN1 and h <= RED_HUE_MAX1) or \
              (h >= RED_HUE_MIN2 and h <= RED_HUE_MAX2)) and \
             (s >= RED_SAT_MIN and s <= RED_SAT_MAX) and \
             (v >= RED_VAL_MIN and v <= RED_VAL_MAX)

    if is_red:
        return '.'

    is_blue = (h >= BLUE_HUE_MIN and h <= BLUE_HUE_MAX) and \
              (s >= BLUE_SAT_MIN and s <= BLUE_SAT_MAX) and \
              (v >= BLUE_VAL_MIN and v <= BLUE_VAL_MAX)

    if is_blue:
        return '-'

    return ''

# --- Extract Morse Code Sequences ---
def extract_morse_sequences(image):
    """
    Resizes the image to 16x16 and extracts Morse code sequences.
    Assumes 16 rows, with each row containing two 8-pixel "words".
    Reads left-to-right, top-to-bottom.
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

def morse_to_text(morse_code_sequence):
    """
    Translates a Morse code sequence (e.g., ".-") into its corresponding character.
    Returns '?' if the sequence is not found in the dictionary, indicating an unrecognized pattern.
    """
    return MORSE_CODE_DICT.get(morse_code_sequence, '?')


## Main Program Execution

def main():
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        print("Please ensure the camera is connected and not in use by another application.")
        return

    # CAP_PROP_AUTO_EXPOSURE: manual exposure mode (0.25).
    # CAP_PROP_EXPOSURE: semakin rendah untuk mengurangkan kecerahan (contoh: -6.0).
    # CAP_PROP_GAIN:semakin rendah untuk mengurangkan noise (contoh: 0.0).
    
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)    
    cap.set(cv2.CAP_PROP_GAIN, 0.0)          

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n--- Morse Code Translator Started ---")
    print("Point your camera at the 16x16 LED matrix.")
    print("Tune camera settings or 'gamma' if LEDs are too bright or colors are washed out.")
    print("Press 'q' or close the window to quit.")
    print("-----------------------------------")

    last_translated_text = "" 

    while True:
        ret, frame = cap.read() 
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # semakin rendah gamma, makin gelap jing
        processed_frame = adjust_gamma(frame, gamma=0.4) 

        # filter
        # processed_frame = cv2.GaussianBlur(processed_frame, (3, 3), 0)

        morse_sequences = extract_morse_sequences(processed_frame)

        current_translated_characters = []
        for morse_seq in morse_sequences:
            translated_char = morse_to_text(morse_seq)
            if morse_seq == '': 
                current_translated_characters.append(' ') 
            elif translated_char != '?':
                current_translated_characters.append(translated_char)
        
        current_translated_text = "".join(current_translated_characters).strip()

        if current_translated_text != last_translated_text:
            print(f"\n--- {time.strftime('%H:%M:%S', time.localtime())} ---")
            print(f"Morse Sequences: {morse_sequences}") 
            print(f"Translated Text: {current_translated_text}")
            print("-----------------------------------")
            last_translated_text = current_translated_text

        viewer_display = cv2.resize(processed_frame, (640, 480), interpolation=cv2.INTER_NEAREST)
        
        window_name = "LED Matrix Viewer (Processed Frame)"
        cv2.imshow(window_name, viewer_display) 
        
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'): 
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: 
            break
            
    cap.release() 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()