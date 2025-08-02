# custom-blade-BEM-simulator
 Python-based Blade Element Momentum (BEM) simulator for custom wind turbine blade design. Supports user-drawn blade profiles, NeuralFoil-based polar generation, and full aerodynamic analysis across wind speeds and angles.

ChatGPT said:
Here’s the complete README.md (no emojis) and requirements.txt for your Custom Blade BEM Simulator. You can paste both directly into GitHub without modification.

README.md
markdown
Copy
Edit
# Custom Blade BEM Simulator

This is a Python-based wind turbine Blade Element Momentum (BEM) simulator that allows users to design custom wind turbine blades and simulate aerodynamic performance across a wide range of wind conditions. The tool integrates aerodynamic polar data generated using NeuralFoil and produces detailed heatmaps, polar plots, and performance metrics for each design.

## Features

- User-drawn blade geometry using a GUI
- Radial discretization of blade for sectional analysis
- NeuralFoil-based polar generation across AoA and Reynolds number
- Calculation of local induction factors (a, a′) and inflow angles
- BEM convergence solver for thrust, torque, and power
- Heatmap and contour plot generation of:
  - Thrust
  - Torque
  - Power
  - Power Coefficient (Cp)
- Cp vs TSR curve generation
- Modular visualization architecture

## File Structure

- `main.py`: GUI for drawing the blade and running the simulation
- `bem.py`: Core BEM computation engine and aerodynamic logic
- `visuals.py`: Generates all plots and data visualizations
- `assets/`: Stores GUI and icon images
- `radial_files/`: Stores interpolated blade sections and airfoil .dat files
- `plots/`: Stores all generated graphs and heatmaps

## How to Run

1. Clone the repository:

   ```
   git clone https://github.com/your-username/custom-blade-bem-simulator.git
   cd custom-blade-bem-simulator
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the GUI:

   ```
   python main.py
   ```

   You can draw your blade geometry, configure wind speed ranges, blade count, RPM, and export results.

## Dependencies

See `requirements.txt` for exact versions. Core libraries include:

- `numpy`
- `matplotlib`
- `scipy`
- `tkinter`
- `Pillow` (for GUI assets)
- `joblib` (for multiprocessing)
- `scikit-learn` (interpolation support)
- `imageio` (for saving animations)

## NeuralFoil

This tool uses precomputed aerodynamic polars generated using the [NeuralFoil](https://github.com/NREL/NeuralFoil) library. You must install and configure NeuralFoil separately for full functionality.

## License

This project is licensed under the MIT License.
