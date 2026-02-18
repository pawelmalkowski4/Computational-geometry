# Delaunay Triangulation

A project developed for the Computational Geometry course, focusing on the implementation and visualization of the Delaunay Triangulation algorithm using the iterative Bowyer-Watson method.

## Features
* **Two variants of finding the starting triangle:**
    * Basic search with a quadratic time complexity $O(N^2)$.
    * Topological neighborhood search, reducing the complexity to $O(N\sqrt{N})$ for larger datasets.
* **Graphical User Interface (`tniap`):** A Tkinter-based application for interactive point insertion via mouse clicks and exporting the collected coordinates to a JSON file.
* **Data generation:** A built-in function for creating random test point clouds with a uniform distribution.
* **Visualization:** Rendering static plots (Matplotlib) and generating frame-by-frame GIF animations (Pillow) to track the cavity removal and retriangulation process step-by-step.

## Project Structure
* `delaunay.py` – the core library defining basic geometric structures (`Point`, `Edge`, `Triangle`) and the main algorithm logic.
* `user.ipynb` – a Jupyter Notebook demonstrating how the code works, containing the GUI application code and usage examples.

## Technical Requirements
* Python 3.10+
* NumPy (2.4.0+)
* Matplotlib (3.10.8)
* Pillow (PIL)
* Tkinter (standard library)

## Example Usage

```python
from delaunay import walkingSearch, plot, render_gif, json_parser

# Load points from a JSON file generated in the GUI
points = json_parser('points.json')

# Generate the triangulation using the optimized algorithm
triangulation = walkingSearch(points)

# Display the generated mesh on a plot
plot(triangulation)

# Save the mesh generation process as an animation
render_gif(points, "animation.gif")
