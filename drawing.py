import cv2
import numpy as np

# Load the image
image = cv2.imread('naruto.jpg')
original_image = image.copy()
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Mask initialized as black
drawing = False  # True if the user is drawing
points = []  # Store points to create the polygon

# Mouse callback function
def draw_polygon(event, x, y, flags, param):
    global drawing, points, mask
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing on left mouse button down
        drawing = True
        points = [(x, y)]  # Initialize points with the starting point
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))  # Add points as the mouse moves
            cv2.line(image, points[-2], points[-1], (0, 255, 0), 2)  # Draw green lines
    
    elif event == cv2.EVENT_LBUTTONUP:  # Stop drawing on left mouse button up
        drawing = False
        points.append((x, y))
        
        # Draw final line to close the shape
        cv2.line(image, points[-1], points[0], (0, 255, 0), 2)
        
        # Fill the polygon on the mask
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)  # Fill inside of the drawn shape with white on the mask

# Set up window and bind mouse callback
cv2.namedWindow('Draw Mask')
cv2.setMouseCallback('Draw Mask', draw_polygon)

while True:
    cv2.imshow('Draw Mask', image)
    
    # Display the masked result in real-time
    masked_result = np.where(mask[:, :, None] == 255, 255, 0).astype(np.uint8)
    cv2.imshow('Masked Result', masked_result)
    
    key = cv2.waitKey(1)
    
    if key == ord('r'):  # Press 'r' to reset the drawing
        image = original_image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    elif key == ord('q'):  # Press 'q' to quit and save
        break

# Save the final mask
cv2.imwrite('mask-naruto.jpg', mask)
cv2.destroyAllWindows()

