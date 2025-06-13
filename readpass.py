import cv2
import numpy as np

def classify_pixel(bgr_pixel):
    hsv = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # Red (dot)
    if ((h < 10 or h > 160) and s > 80 and v > 80):
        return '.'

    # Blue (dash)
    if (90 < h < 130) and s > 80 and v > 50:
        return '-'

    # Black or other
    return ''

def extract_morse_words(image):
    resized = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    words = []

    for row in range(16):
        row_data = resized[row]  # shape: (16, 3)
        for word_idx in range(2):
            start = word_idx * 8
            word = ''
            for i in range(8):
                pixel = row_data[start + i]
                word += classify_pixel(pixel)
            words.append(word.strip())
    return words

def main():
    image_path = "test3.png"  # or your new blue-dash image
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Cannot load image at: {image_path}")
        return

    morse_words = extract_morse_words(image)

    print("\n📷 Morse Code from Image (Red = dot, Blue = dash):")
    for i, word in enumerate(morse_words):
        print(f"{i+1:02}: {word}")

    # Optional: visualize
    tiny = cv2.resize(image, (160, 160), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Resized 16x16 View", tiny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
