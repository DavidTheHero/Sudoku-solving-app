import cv2
import numpy as np
from tensorflow.keras.models import load_model
from solvers.backtrack import backtracking_solve

class SudokuSolver:
    def __init__(self, digit_model_path):
        self.digit_model = load_model(digit_model_path)
        self.solvers = {
            'backtrack': backtracking_solve
        }

    def solve(self, image_path):
        # Step 1: Preprocess image and extract grid
        warped = self._extract_grid(image_path)
        # print("grid extracted")
        # Step 2: Recognize digits
        puzzle = self._recognize_digits(warped)
        # print('digits extracted')
        # print(puzzle)
        solution = [row.copy() for row in puzzle]  # Create a mutable copy
        self.solvers['backtrack'](solution)
        return puzzle, solution

    def _extract_grid(self, image_path):
        """Align and extract Sudoku grid from image."""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find and warp grid
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest, True)
        corners = cv2.approxPolyDP(largest, epsilon, True)
        
        # Warp perspective
        pts = corners.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        width = height = 450
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (width, height))
    
    def _preprocess_digit(self, cell_img):
        """Modified to return preprocessed image for visualization"""
        processed = cv2.resize(cell_img, (28,28))
        processed = cv2.threshold(processed, 128, 255, cv2.THRESH_BINARY_INV)[1]
        return processed.astype('float32')/255.0

    def _recognize_digit(self, processed_img):
        """Modified to accept preprocessed images"""
        return np.argmax(self.digit_model.predict(np.array([processed_img])))

# Example Usage
if __name__ == "__main__":
    solver = SudokuSolver('digit_recognition.keras')
    # sudoku_dataset/grid_0.png
    puzzle, solution = solver.solve('test.png')
    
    print("Detected Puzzle:")
    print(np.array(puzzle))
    print("\nSolved Puzzle:")
    print(np.array(solution))