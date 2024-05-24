import cv2
import numpy as np
import webcolors

def closest_color(col):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        rc, gc, bc = webcolors.hex_to_rgb(key)
        rd = (rc - col[0]) ** 2
        gd = (gc - col[1]) ** 2
        bd = (bc - col[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def process_frame(frame, blur_ksize, canny_threshold1, canny_threshold2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # kernel must be positive and odd
    blur_ksize = max(1, blur_ksize)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edged = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_copy = frame.copy()

    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print(f"{len(contours)} objects were found in this frame.")

    j = 1

    for contour in contours:
        # bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # middle
        mp = (x + w // 2, y + h // 2)

        # circle at middle
        cv2.circle(image_copy, mp, 3, (251, 3, 213), 2)

        bgr_color = frame[y + h // 2, x + w // 2]
        print(f"Wire {j}")
        print(f"BGR color : {bgr_color}")

        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        print(f"RGB color : {rgb_color}")

        try:
            color_name = webcolors.rgb_to_name(rgb_color)
        except ValueError:
            color_name = closest_color(rgb_color)

        print(f"Color name : {color_name}")

        j += 1

    return image_copy, edged

def on_trackbar(val):
    pass

def create_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('Blur', 'Trackbars', 1, 50, on_trackbar)
    cv2.createTrackbar('Canny Threshold 1', 'Trackbars', 50, 500, on_trackbar)
    cv2.createTrackbar('Canny Threshold 2', 'Trackbars', 200, 500, on_trackbar)

def get_trackbar_values():
    blur_ksize = cv2.getTrackbarPos('Blur', 'Trackbars')
    canny_threshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Trackbars')
    canny_threshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Trackbars')
    return blur_ksize, canny_threshold1, canny_threshold2

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image")
        return

    create_trackbars()

    # Initialize
    prev_ksize, prev_canny_threshold1, prev_canny_threshold2 = 5, 50, 200

    while True:
        blur_ksize, canny_threshold1, canny_threshold2 = get_trackbar_values()

        # Update
        if (blur_ksize != prev_ksize or
            canny_threshold1 != prev_canny_threshold1 or
            canny_threshold2 != prev_canny_threshold2):

            prev_ksize, prev_canny_threshold1, prev_canny_threshold2 = blur_ksize, canny_threshold1, canny_threshold2

            image_copy, edged = process_frame(image, blur_ksize, canny_threshold1, canny_threshold2)

            cv2.imshow("Contours", image_copy)
            cv2.imshow("Edged", edged)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def live_feed():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    create_trackbars()

    # Initialize 
    prev_ksize, prev_canny_threshold1, prev_canny_threshold2 = -1, -1, -1

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        blur_ksize, canny_threshold1, canny_threshold2 = get_trackbar_values()

        # Update
        if (blur_ksize != prev_ksize or
            canny_threshold1 != prev_canny_threshold1 or
            canny_threshold2 != prev_canny_threshold2):

            prev_ksize, prev_canny_threshold1, prev_canny_threshold2 = blur_ksize, canny_threshold1, canny_threshold2

            image_copy, edged = process_frame(frame, blur_ksize, canny_threshold1, canny_threshold2)

            cv2.imshow("Contours", image_copy)
            cv2.imshow("Edged", edged)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    choice = input("Enter 'image' to process an image or 'live' for live feed: ").strip().lower()
    if choice == 'image':
        image_path = input("Enter the path to the image: ").strip()
        process_image(image_path)
    elif choice == 'live':
        live_feed()
    else:
        print("Invalid choice. Please enter 'image' or 'live'.")

if __name__ == "__main__":
    main()
