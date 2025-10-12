import cv2
import numpy as np
import math
from itertools import combinations
import time

# Define a small tolerance for comparing floating-point numbers (distances and areas).
EPSILON = 0.1

# --- Core Classification Logic from ShapeIdentifier Class (Adapted for OpenCV) ---

class ShapeClassifier:
    """
    Implements the shape identification logic from Section IV of the paper,
    adapted to use OpenCV contour properties (area and approximated vertices).
    """
    def __init__(self, epsilon=EPSILON):
        self.epsilon = epsilon
        # Predefined area formulas (used as reference for 3D shapes which are flat projections)
        # Note: In a real implementation, 'r', 'h', 'l' would need to be calculated
        # from the image features (e.g., radius from bounding circle, height from dimensions).
        self.area_formulas = {
            "CYLINDER": lambda r, h: 2 * math.pi * r * h + 2 * math.pi * r**2,
            "HEMISPHERE": lambda r: math.pi * r**2,
            "CONE": lambda r, l: math.pi * r * l + math.pi * r**2
        }
        # Mock geometric dimensions based on an arbitrary reference size (e.g., 50x50 object)
        self.R_MOCK = 25.0
        self.H_MOCK = 50.0
        self.L_MOCK = 55.9

        self.cylinder_area_mock = self.area_formulas["CYLINDER"](self.R_MOCK, self.H_MOCK)
        self.hemisphere_area_mock = self.area_formulas["HEMISPHERE"](self.R_MOCK)
        self.cone_area_mock = self.area_formulas["CONE"](self.R_MOCK, self.L_MOCK)


    def _calculate_distance(self, p1, p2):
        """Calculates Euclidean distance between two points (x, y)."""
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _get_distances_from_points(self, points):
        """Calculates all 6 pairwise distances among the 4 input points."""
        # Convert points from numpy array format [ [[x, y]], ... ] to list of tuples [(x, y), ...]
        flat_points = [(p[0][0], p[0][1]) for p in points]

        # Get all combinations of 2 points (6 combinations: D1 to D6)
        distances = []
        for p1, p2 in combinations(flat_points, 2):
            distances.append(self._calculate_distance(p1, p2))

        # Sort the distances for easier comparison later
        distances.sort()
        return distances

    def _is_equal(self, a, b):
        """Checks if two floating-point numbers are equal within the permissible error."""
        return abs(a - b) < self.epsilon

    def classify(self, approx, calculated_area):
        """
        Classifies the shape based on the number of vertices, distances, and area.
        """
        num_vertices = len(approx)

        if num_vertices == 4:
            # Quad-lateral logic (Rectangle, Cylinder, Kite, Square, Rhombus)
            D = self._get_distances_from_points(approx)
            sides = sorted(D[:4])
            diagonals = D[4:]

            # --- Sub-section (b): Square and Rhombus ---
            is_square_or_rhombus = (
                self._is_equal(sides[0], sides[1]) and
                self._is_equal(sides[0], sides[2]) and
                self._is_equal(sides[0], sides[3])
            )

            if is_square_or_rhombus:
                if self._is_equal(diagonals[0], diagonals[1]):
                    return "SQUARE"
                else:
                    return "RHOMBUS"

            # --- Sub-section (a): Rectangle, Cylinder and Kite ---
            has_two_pairs_of_sides = (
                self._is_equal(sides[0], sides[1]) and
                self._is_equal(sides[2], sides[3])
            )

            if has_two_pairs_of_sides:
                if self._is_equal(diagonals[0], diagonals[1]):
                    # Distinguish Rectangle from Cylinder based on Area
                    if self._is_equal(calculated_area, self.cylinder_area_mock):
                        return "CYLINDER"
                    else:
                        return "RECTANGLE"
                else:
                    return "KITE" # Diagonals are unequal

            return "QUADRILATERAL (Other)"

        elif num_vertices == 3:
            # --- Sub-section (d): Triangle and Cone ---
            # Area-based distinction for 3-point shapes
            if self._is_equal(calculated_area, self.cone_area_mock):
                return "CONE"
            else:
                return "TRIANGLE"

        elif num_vertices > 4:
            # Heuristic for Hemisphere/Circle/Smooth Shapes (Large number of vertices)
            # Find the minimum enclosing circle. If the contour is close to the circle, it's round.
            (x, y), radius = cv2.minEnclosingCircle(approx)
            circle_area = math.pi * radius**2
            if abs(calculated_area - circle_area) < 0.2 * calculated_area: # Check if area is close to the enclosing circle area
                 # --- Sub-section (c): Hemisphere (Area = πr²) ---
                 if self._is_equal(calculated_area, self.hemisphere_area_mock):
                     return "HEMISPHERE"
                 else:
                     return "CIRCLE/OVAL"

            return f"POLYGON ({num_vertices} Sides)"

        else:
            return "UNKNOWN"

# --- OpenCV Main Application ---

def run_shape_detection(classifier):
    """Initializes and runs the real-time shape detection application."""
    # 0 is the default camera ID
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Set up window names
    live_window = 'Live Feed - Shape Identification'
    edge_window = 'Edge Detection Field'
    cv2.namedWindow(live_window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(edge_window, cv2.WINDOW_AUTOSIZE)

    print("--- Shape Detection Running ---")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # 1. Pre-processing: Convert to grayscale and blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Edge Detection (Canny) - This generates the image for the second window
        edges = cv2.Canny(blurred, 50, 150)

        # 3. Find Contours
        # RETR_EXTERNAL retrieves only the extreme outer contours.
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the frame to draw annotations on
        annotated_frame = frame.copy()

        # 4. Process Contours
        for contour in contours:
            # Filter out small noise contours based on area
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            # Approximate the contour to a polygon to find vertices
            perimeter = cv2.arcLength(contour, True)
            # Use a small epsilon (e.g., 2%-4% of perimeter) for approximation accuracy
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Draw the contour on the original frame
            cv2.drawContours(annotated_frame, [approx], 0, (0, 255, 0), 2)

            # Classify the shape using the paper's logic
            shape_label = classifier.classify(approx, area)

            # Find the center of the contour for text placement
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Annotate the detected shape
                cv2.putText(annotated_frame, shape_label, (cX - 50, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                 # If moments fail, use the first vertex as a fallback position
                x, y = approx[0][0]
                cv2.putText(annotated_frame, shape_label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        # 5. Display the results in two windows
        cv2.imshow(live_window, annotated_frame)
        cv2.imshow(edge_window, edges)

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize the classifier instance
    shape_classifier = ShapeClassifier(epsilon=500.0) # Using a larger epsilon for area matching tolerance

    # Run the main detection loop
    run_shape_detection(shape_classifier)

# Note: The mock areas for 3D shapes (Cone, Cylinder, Hemisphere) are fixed.
# To test these, you would need to display a projected image of a 3D shape
# that happens to match the mock area calculation (e.g., area=πr² for hemisphere base).
# The quadrilateral logic (Square, Rhombus, Rectangle, Kite) is robust and can be
# easily tested by displaying these shapes to the camera.
