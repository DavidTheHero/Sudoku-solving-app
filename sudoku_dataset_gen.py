import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import random

# --- Config ---
OUTPUT_DIR = "sudoku_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_SIZE = 9
CELL_SIZE = 64  # Each Sudoku cell is 64x64 pixels
MARGIN = 16    # Margin around the grid
IMG_SIZE = CELL_SIZE * GRID_SIZE + 2 * MARGIN

# More diverse fonts (try to include handwritten-style fonts)
FONT_PATHS = [
    "arial.ttf",  # Windows
    "arialbd.ttf",  # Bold Arial
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "/Library/Fonts/Arial.ttf",  # Mac
    "times.ttf",  # Serif font
    "cour.ttf",  # Courier monospace
]

# --- Helper Functions ---
def generate_sudoku():
    """Generate a valid Sudoku puzzle (9x9 numpy array)."""
    base = np.arange(1, 10)
    np.random.shuffle(base)
    
    # Create a solved Sudoku grid
    grid = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            grid[i][j] = base[(i * 3 + i // 3 + j) % 9]
    
    # Randomly remove cells to create a puzzle
    mask = np.random.choice([0, 1], size=(9, 9), p=[0.6, 0.4])  # 40% digits kept
    puzzle = grid * mask
    return puzzle, grid  # (puzzle, solution)

def apply_distortion(image):
    """Apply random perspective distortion."""
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [random.randint(-10, 10), random.randint(-10, 10)],
        [w - random.randint(-10, 10), random.randint(-10, 10)],
        [random.randint(-10, 10), h - random.randint(-10, 10)],
        [w - random.randint(-10, 10), h - random.randint(-10, 10)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    distorted = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return distorted

def apply_noise(image):
    """Add Gaussian noise and blur."""
    noise = np.random.randn(*image.shape) * random.randint(5, 20)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(noisy, (3, 3), 0)
    return blurred

def load_fonts(font_paths, size=48):
    """Load multiple fonts with fallback."""
    fonts = []
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, size)
            fonts.append(font)
        except:
            continue
    if not fonts:
        raise FileNotFoundError("No usable fonts found.")
    return fonts

def apply_digit_distortion(digit_img):
    """Apply random distortions to a single digit."""
    digit_img = Image.fromarray(digit_img)
    
    # Random rotation (-15° to +15°)
    if random.random() > 0.7:
        angle = random.uniform(-15, 15)
        digit_img = digit_img.rotate(angle, expand=False, fillcolor=255)
    
    # Random scaling (80% to 120%)
    if random.random() > 0.7:
        scale = random.uniform(0.8, 1.2)
        new_size = (int(digit_img.width * scale), int(digit_img.height * scale))
        digit_img = digit_img.resize(new_size, Image.LANCZOS)
        # Pad back to original size
        pad_x = (CELL_SIZE - new_size[0]) // 2
        pad_y = (CELL_SIZE - new_size[1]) // 2
        digit_img = ImageOps.expand(digit_img, (pad_x, pad_y, CELL_SIZE - new_size[0] - pad_x, CELL_SIZE - new_size[1] - pad_y), fill=255)
    
    # Random shear (skew)
    if random.random() > 0.7:
        shear_x = random.uniform(-0.2, 0.2)
        shear_y = random.uniform(-0.2, 0.2)
        digit_img = digit_img.transform(
            (CELL_SIZE, CELL_SIZE),
            Image.AFFINE,
            (1, shear_x, 0, shear_y, 1, 0),
            fillcolor=255
        )
    
    return np.array(digit_img)

def apply_ink_variations(image):
    """Simulate ink variations (smudges, fading)."""
    if random.random() > 0.5:
        # Randomly fade parts of the digit
        mask = np.random.random(image.shape) > 0.9
        image[mask] = np.clip(image[mask] + random.randint(30, 100), 0, 255)
    
    if random.random() > 0.5:
        # Add salt-and-pepper noise
        noise = np.random.randint(0, 256, image.shape)
        image[noise < 10] = 0   # Black speckles
        image[noise > 245] = 255 # White speckles
    
    return image

def draw_sudoku(puzzle, fonts):
    """Render Sudoku grid with digits (now with digit-level augmentations)."""
    image = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(image)
    
    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        line_width = 4 if i % 3 == 0 else 2
        x = MARGIN + i * CELL_SIZE
        y = MARGIN + i * CELL_SIZE
        draw.line([(x, MARGIN), (x, IMG_SIZE - MARGIN)], width=line_width, fill=0)
        draw.line([(MARGIN, y), (IMG_SIZE - MARGIN, y)], width=line_width, fill=0)
    
    # Draw digits with variations
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            digit = puzzle[i, j]
            if digit != 0:
                # Choose a random font for variation
                font = random.choice(fonts)
                
                # Create a temporary image for the digit
                digit_img = Image.new("L", (CELL_SIZE, CELL_SIZE), color=255)
                digit_draw = ImageDraw.Draw(digit_img)
                
                # Draw the digit
                x, y = CELL_SIZE // 2, CELL_SIZE // 2
                digit_draw.text((x, y), str(digit), fill=0, font=font, anchor="mm")
                
                # Apply distortions to the digit
                digit_img = apply_digit_distortion(np.array(digit_img))
                digit_img = apply_ink_variations(digit_img)
                
                # Paste the digit into the grid
                x_pos = MARGIN + j * CELL_SIZE
                y_pos = MARGIN + i * CELL_SIZE
                image.paste(Image.fromarray(digit_img), (x_pos, y_pos))
    
    return np.array(image)

def generate_synthetic_sample(fonts):
    puzzle, solution = generate_sudoku()
    image = draw_sudoku(puzzle, fonts)
    
    # Apply global distortions
    if random.random() > 0.5:
        image = apply_distortion(image)
    if random.random() > 0.5:
        image = apply_noise(image)
    
    # Extract cells and labels
    cells = []
    labels = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x1 = MARGIN + j * CELL_SIZE
            y1 = MARGIN + i * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            cell = image[y1:y2, x1:x2]
            cell = cv2.resize(cell, (28, 28))  # Resize to MNIST-like format
            cells.append(cell)
            labels.append(puzzle[i, j])
    
    return image, cells, labels

# --- Generate Dataset ---
def generate_dataset(num_samples=1000):
    fonts = load_fonts(FONT_PATHS)
    
    for sample_idx in range(num_samples):
        image, cells, labels = generate_synthetic_sample(fonts)
        
        # Save full grid image (optional)
        cv2.imwrite(f"{OUTPUT_DIR}/grid_{sample_idx}.png", image)
        
        # Save individual cells and labels
        for cell_idx, (cell, label) in enumerate(zip(cells, labels)):
            cell_filename = f"{OUTPUT_DIR}/sample_{sample_idx}_cell_{cell_idx}_label_{label}.png"
            cv2.imwrite(cell_filename, cell)
        
        if (sample_idx + 1) % 100 == 0:
            print(f"Generated {sample_idx + 1}/{num_samples} samples")

if __name__ == "__main__":
    generate_dataset(num_samples=1000)
    print(f"dataset saved to {OUTPUT_DIR}")