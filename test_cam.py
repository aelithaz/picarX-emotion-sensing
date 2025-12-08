import cv2

def try_index(idx):
    print(f"Trying index {idx}...")
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  -> failed to open index {idx}")
        return False

    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"  -> opened but failed to read frame at index {idx}")
        cap.release()
        return False

    h, w = frame.shape[:2]
    print(f"  -> SUCCESS at index {idx}, frame size = {w}x{h}")
    cap.release()
    return True

def main():
    # Try first few indices that actually exist
    for idx in range(0, 8):
        if try_index(idx):
            print(f"\n*** Use camera index {idx} in your app.py ***")
            return
    print("\nNo usable camera index found.")

if __name__ == "__main__":
    main()
