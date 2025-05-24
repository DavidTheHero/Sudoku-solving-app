import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pipeline import SudokuSolver
from solvers.backtrack import backtracking_solve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SudokuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver")
        self.root.geometry("900x900")
        self.configure_styles()
        
        # Solver setup
        self.solver = SudokuSolver('digit_recognition.keras')
        self.solver_method = tk.StringVar(value="backtrack")
        self.current_puzzle = np.zeros((9, 9), dtype=int)
        
        # UI Elements
        self.setup_ui()
        
    def configure_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('TRadiobutton', font=('Arial', 10))
        style.configure('TEntry', font=('Arial', 16), justify='center')
        style.configure('Bold.TEntry', font=('Arial', 16, 'bold'))
        style.map('Selected.TEntry',
                 fieldbackground=[('active', 'lightblue')])
        
        # Configure grid cell colors
        for i in range(9):
            for j in range(9):
                region = (i//3)*3 + (j//3)
                style.configure(f'Region{region}.TEntry',
                              background=self.get_region_color(region))
    
    def get_region_color(self, region):
        """Get alternating background colors for 3x3 regions"""
        return '#f0f0f0' if region % 2 else '#ffffff'
        
    def setup_ui(self):
        """Initialize all UI components"""
        # Top Frame (Image and Controls)
        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=10)
        
        # Left Column (Image)
        self.image_label = ttk.Label(top_frame)
        self.image_label.pack(side=tk.LEFT, padx=10)
        
        # Right Column (Controls)
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(side=tk.LEFT)
        
        buttons = [
            ("Upload Image", self.load_image),
            ("Recognize Digits", self.recognize_digits),
            ("Solve Sudoku", self.solve_sudoku),
            ("Check Solution", self.check_solution),
            ("Clear All", self.clear_all)
        ]
        
        for text, command in buttons:
            ttk.Button(control_frame, text=text, command=command).pack(pady=5, fill=tk.X)
        
        # Solver Selection
        solver_frame = ttk.LabelFrame(control_frame, text="Solver Method")
        solver_frame.pack(pady=5, fill=tk.X)
        ttk.Radiobutton(
            solver_frame, text="Backtracking", variable=self.solver_method, value="backtrack"
        ).pack(side=tk.LEFT, padx=5)
        
        # Sudoku Grid Frame
        self.grid_frame = ttk.Frame(self.root)
        self.grid_frame.pack(pady=10)
        self.create_editable_grid()
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(
            self.root, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W
        ).pack(fill=tk.X, pady=5)
        
    def create_editable_grid(self):
        """Create 9x9 grid of editable entry widgets with 3x3 region coloring"""
        self.cell_vars = []
        for i in range(9):
            row_vars = []
            for j in range(9):
                var = tk.StringVar()
                region = (i//3)*3 + (j//3)
                entry = ttk.Entry(
                    self.grid_frame, textvariable=var,
                    width=2, style=f'Region{region}.TEntry'
                )
                entry.grid(row=i, column=j, ipady=5, padx=(1,0), pady=(1,0))
                entry.bind('<FocusIn>', lambda e, i=i, j=j: self.on_cell_select(i, j))
                entry.bind('<Key>', self.validate_input)
                row_vars.append(var)
            self.cell_vars.append(row_vars)
    
    def validate_input(self, event):
        """Validate that only digits 1-9 are entered"""
        if event.char and (not event.char.isdigit() or event.char == '0'):
            return 'break'
    
    def on_cell_select(self, row, col):
        """Highlight selected cell and its row/column/region"""
        for i in range(9):
            for j in range(9):
                entry = self.grid_frame.grid_slaves(row=i, column=j)[0]
                if i == row and j == col:
                    entry.configure(style='Selected.TEntry')
                else:
                    region = (i//3)*3 + (j//3)
                    entry.configure(style=f'Region{region}.TEntry')
    
    def load_image(self):
        """Load Sudoku image from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            try:
                self.image_path = file_path
                img = Image.open(file_path)
                img.thumbnail((400, 400))  # Maintain aspect ratio
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
                self.status_var.set("Image loaded. Click 'Recognize Digits'.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def recognize_digits(self):
        """Recognize digits from image and show preprocessing results"""
        if not hasattr(self, 'image_path'):
            messagebox.showerror("Error", "Please upload an image first!")
            return
        
        try:
            self.status_var.set("Processing image...")
            self.root.update()
            
            warped = self.solver._extract_grid(self.image_path)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            # Create preview window
            preview = tk.Toplevel(self.root)
            preview.title("Digit Preprocessing Preview")
            preview.minsize(600, 600)
            
            fig, axes = plt.subplots(9, 9, figsize=(8,8))
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            
            cell_size = gray.shape[0] // 9
            self.current_puzzle = np.zeros((9,9), dtype=int)
            
            for i in range(9):
                for j in range(9):
                    # Extract and preprocess cell
                    x1, y1 = j*cell_size, i*cell_size
                    cell = gray[y1:y1+cell_size, x1:x1+cell_size]
                    
                    # Show original cell
                    axes[i,j].imshow(cell, cmap='gray')
                    axes[i,j].axis('off')
                    
                    # Preprocess and recognize
                    processed = self.solver._preprocess_digit(cell)
                    digit = self.solver._recognize_digit(processed)
                    self.current_puzzle[i,j] = digit
                    
                    # Label with recognized digit
                    axes[i,j].set_title(str(digit) if digit !=0 else "", fontsize=8)
            
            canvas = FigureCanvasTkAgg(fig, master=preview)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update main grid
            self.update_grid_from_puzzle()
            
            self.status_var.set("Digits recognized. Check preview window.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed:\n{str(e)}")
            self.status_var.set("Recognition failed")
    
    def update_grid_from_puzzle(self):
        """Update the grid display from current puzzle"""
        for i in range(9):
            for j in range(9):
                val = self.current_puzzle[i][j]
                self.cell_vars[i][j].set(str(val) if val !=0 else "")
    
    def get_grid_state(self):
        """Get current grid state from UI"""
        grid = np.zeros((9,9), dtype=int)
        for i in range(9):
            for j in range(9):
                val = self.cell_vars[i][j].get()
                grid[i][j] = int(val) if val else 0
        return grid
    
    def solve_sudoku(self):
        """Solve the current puzzle (either recognized or manually edited)"""
        try:
            self.current_puzzle = self.get_grid_state()
            
            if not self.is_valid_puzzle(self.current_puzzle):
                messagebox.showerror("Error", "Invalid puzzle configuration")
                return
            
            solution = self.current_puzzle.copy()
            if backtracking_solve(solution):
                self.current_puzzle = solution
                self.update_grid_from_puzzle()
                self.status_var.set("Sudoku solved!")
            else:
                messagebox.showinfo("Info", "No solution exists for this puzzle")
                self.status_var.set("No solution found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Solving failed:\n{str(e)}")
            self.status_var.set("Solving failed")
    
    def check_solution(self):
        """Check if current grid is a valid solution"""
        try:
            grid = self.get_grid_state()
            if self.is_valid_solution(grid):
                messagebox.showinfo("Solution Check", "Congratulations! The solution is correct.")
            else:
                messagebox.showwarning("Solution Check", "The solution is not correct.")
        except Exception as e:
            messagebox.showerror("Error", f"Validation failed:\n{str(e)}")
    
    def is_valid_puzzle(self, puzzle):
        """Check if puzzle configuration is valid"""
        # Check rows
        for i in range(9):
            row = puzzle[i,:]
            if len(set(row[row != 0])) != sum(row != 0):
                return False
        
        # Check columns
        for j in range(9):
            col = puzzle[:,j]
            if len(set(col[col != 0])) != sum(col != 0):
                return False
        
        # Check 3x3 regions
        for i in range(0,9,3):
            for j in range(0,9,3):
                region = puzzle[i:i+3,j:j+3].flatten()
                if len(set(region[region != 0])) != sum(region != 0):
                    return False
        return True
    
    def is_valid_solution(self, puzzle):
        """Check if puzzle is completely and correctly solved"""
        # Check all cells are filled
        if np.any(puzzle == 0):
            return False
        
        # Check all rows contain 1-9
        for i in range(9):
            if set(puzzle[i,:]) != set(range(1,10)):
                return False
        
        # Check all columns contain 1-9
        for j in range(9):
            if set(puzzle[:,j]) != set(range(1,10)):
                return False
        
        # Check all 3x3 regions contain 1-9
        for i in range(0,9,3):
            for j in range(0,9,3):
                if set(puzzle[i:i+3,j:j+3].flatten()) != set(range(1,10)):
                    return False
        
        return True
    
    def clear_all(self):
        """Clear the entire grid"""
        for i in range(9):
            for j in range(9):
                self.cell_vars[i][j].set("")
        self.current_puzzle = np.zeros((9,9), dtype=int)
        self.status_var.set("Grid cleared")

if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuApp(root)
    root.mainloop()