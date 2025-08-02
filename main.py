from tkinter import *
import matplotlib.pyplot as plt
import os, shutil, time, subprocess, sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image, ImageTk
from scipy.interpolate import interp1d
from bem import *
from visuals import *
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
import zipfile
from pathlib import Path

#creates the main window
root = Tk()
root.title("Windmill Blade Design")
root.geometry("1200x800")
root.config(bg="skyblue1")
root.resizable(False, False)
for i in range(16):
    root.grid_rowconfigure(i, weight=1)
    root.grid_columnconfigure(i, weight=1)

#loads a blade logo onto the main panel as a visual touch
img = Image.open("assets/windmill_logo.png")
tk_img = ImageTk.PhotoImage(img.resize((300, 300)))
image_label = Label(image=tk_img, bg="skyblue1")
image_label.grid(columnspan=16, row=1)

#creates the main titular label of the opening panel
main_label = Label(text="Windmill Blade Efficiency SimulIator", font=("Times New Roman", 40), fg="deepskyblue4", bg="skyblue1")
main_label.grid_configure(columnspan=16, row=0)

#defines some variables that are needed
final_points = []
final_parameters=["", "", "", "", "", "", "", "", "", "", ""]
valid = False
valid_par = False

#function that restarts the program
def restart_program():
    main_label.config(text="Restarting simulation...")
    root.update()
    time.sleep(1)
    root.destroy()
    subprocess.call([sys.executable, sys.argv[0]])

#function that split the points into differential elements
def densify_points(points, step_size):
    densified = []
    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])
        dist = np.linalg.norm(p2 - p1)
        steps = max(int(np.ceil(dist / step_size)), 1)
        interp = [p1 + (p2 - p1) * t for t in np.linspace(0, 1, steps, endpoint=False)]
        densified.extend(interp)
    densified.append(points[-1])
    return np.array(densified)

#function that adds a circular arc according to the provided radial
def add_circular_arc(center, radius, start_point, end_point, resolution=50):
    
    #calculates the angle of the start and end of arc
    def angle(p): return np.arctan2(p[1] - center[1], p[0] - center[0])
    theta1 = angle(start_point)
    theta2 = angle(end_point)

    #checks for angle wrapping
    if theta2 < theta1: theta2 += 2 * np.pi

    #builds the arc along radial and returns it
    arc_thetas = np.linspace(theta1, theta2, resolution)
    arc = np.stack([center[0] + radius * np.cos(arc_thetas), center[1] + radius * np.sin(arc_thetas)], axis=1)
    arc[0] = start_point
    arc[-1] = end_point
    return arc

#function that handles processing the user 
def export_radials(densified_points, epsilon=1e-8):
    #configures the directory that the files are stored into
    output_dir = "radial_files"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    #extracts the x, y, and configures the differential radials
    x, y = densified_points[:, 0], densified_points[:, 1]
    r_all = np.sqrt(x**2 + y**2)
    r_min = max(r_all.min(), epsilon)
    r_max = r_all.max()
    N = int(np.round(r_max * 20))
    r_edges = np.linspace(r_min, r_max, N + 1)

    radii, twist_angles, chord_lengths = [], [], []

    #iterates over all the differential radials for each airfoil
    for i in range(N):
        #finds the middle and edge radials of the airfoil
        r0, r1 = r_edges[i], r_edges[i + 1]
        r_mid = 0.5 * (r0 + r1)

        #creates shell points based on the point that are in the radial
        in_band = (r_all >= r0) & (r_all <= r1)
        shell_points = densified_points[in_band]

        #skips the foil section if its length is too little
        if len(shell_points) < 5: continue

        #sorts the shell points by angle
        sorted_shell = shell_points[np.argsort(np.arctan2(shell_points[:, 1], shell_points[:, 0]))]

        #creates masks to find points close to the airfoil for clusters
        dists = np.sqrt(shell_points[:, 0]**2 + shell_points[:, 1]**2)
        inner_mask = np.abs(dists - r0) < (r1 - r0) * 0.005
        outer_mask = np.abs(dists - r1) < (r1 - r0) * 0.005
        inner_points = shell_points[inner_mask]
        outer_points = shell_points[outer_mask]

        #function that counts the number of clusters at a radial
        def count_clusters(points, distance_threshold=1e-3):
            #sorts by angle from origin
            angles = np.arctan2(points[:, 1], points[:, 0])
            sorted_indices = np.argsort(angles)
            sorted_points = points[sorted_indices]

            #finds the Euclidean distance and gaps to seperate clusters and returns clusters = gaps + 1
            dists = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
            gap_indices = np.where(dists > distance_threshold)[0]
            return len(gap_indices) + 1

        #function that finds the gaps within the clusters
        def split_by_gaps(points, distance_threshold=1e-3):
            #calculates the angle and then sorted by it
            angles = np.arctan2(points[:, 1], points[:, 0])
            sorted_idx = np.argsort(angles)
            sorted_points = points[sorted_idx]

            #tries to find major gaps beteween clusters and returns 
            dists = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
            gap_indices = np.where(dists > distance_threshold)[0]
            split_indices = [0] + [g+1 for g in gap_indices] + [len(points)]
            clusters = [sorted_points[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices)-1)]
            return clusters

        #function that splits the user drawn shell points based on jumps
        def split_by_discontinuities(points, n_segments_expected):
            #sorts points by y 
            points_sorted = points[np.argsort(points[:, 1])]


            #compute differences in y and finds the index of largest expected gaps and returns
            diffs = np.abs(np.diff(points_sorted[:, 1]))
            split_indices = np.argsort(diffs)[-n_segments_expected+1:]
            split_indices = np.sort(split_indices + 1)
            segments = np.split(points_sorted, split_indices)
            return [seg for seg in segments if len(seg) > 2]
        
            #saves to file for debug (optional)
            """with open(f"{r_mid}.txt", "w") as f:
                for i, seg in enumerate(segments):
                    f.write(f"# Segment {i + 1} (shape={seg.shape})\n")
                    np.savetxt(f, seg, fmt="%.5f")
                    f.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")"""
        
        """ Unused function that is needed when processing branched airfoils for
         future expansion of blade into irregular windmills
        def match_inner_outer_by_jumps_clusters(inner_clusters, outer_clusters):
            def cluster_angle(cluster):
                cluster = np.atleast_2d(cluster)
                return np.mean(np.arctan2(cluster[:, 1], cluster[:, 0]))

            # Total airfoils = number of inner cluster pairs
            m = len(inner_clusters) // 2
            if m == 0 or len(outer_clusters) < m:
                return []

            # Sort both cluster sets by mean angle
            inner_sorted = sorted(inner_clusters, key=cluster_angle)
            outer_sorted = sorted(outer_clusters, key=cluster_angle)

            # Calculate mean angle of each outer cluster
            outer_angles = np.array([cluster_angle(c) for c in outer_sorted])

            outer_angles = np.array([cluster_angle(c) for c in outer_sorted])
            angle_diffs = np.diff(outer_angles)
            n_in = len(inner_clusters)
            n_out = len(outer_clusters)
            m = n_in // 2  # Number of airfoils
            n_gaps = m - 1

            # Sanity check
            if n_in % 2 != 0 or n_out < m * 2:
                return []
            
            print(n_gaps, n_in, n_out)

            if n_gaps > 0:
                gap_indices = np.argsort(angle_diffs)[-n_gaps:]
                gap_indices = np.sort((gap_indices + 1))
                split_indices = [0] + list(gap_indices) + [len(outer_sorted)]
            else:
                split_indices = [0, len(outer_sorted)]
            
            print(split_indices)

            # Now split outer_sorted clusters correctly
            outer_groups_raw = [outer_sorted[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices)-1)]

            # Stack each group of outer clusters
            outer_groups_stacked = [np.vstack(group) for group in outer_groups_raw if len(group) > 0]
            inner_groups = [np.vstack(inner_sorted[2*i:2*i+2]) for i in range(m)]

            print(f"# Inner groups: {len(inner_groups)}, # Outer groups: {len(outer_groups_stacked)}")
            if len(inner_groups) != len(outer_groups_stacked):
                print("Mismatch in matched inner and outer groups! Possible clustering issue.")

            return list(zip(inner_groups, outer_groups_stacked, outer_groups_raw))
        """

        #finds the number of inner and outer clusters to process blade geomeotry
        n_clusters_inner = count_clusters(inner_points, distance_threshold=0.0005)
        n_clusters_outer = count_clusters(outer_points, distance_threshold=0.0005)

        #throws an error in case
        if len(inner_points) < 2 or len(outer_points) < 2:
            print("too few points, skipping radial foil")
            continue

        #function responsible for making the airfoil at a normal continous sectioon
        def continous_foil(in_points, out_points, edge_points, n, huh=False):
            #calculates the angles and sorted points by angle
            inner_angles = np.arctan2(in_points[:, 1], in_points[:, 0])
            outer_angles = np.arctan2(out_points[:, 1], out_points[:, 0])
            inner_sorted = in_points[np.argsort(inner_angles)]
            outer_sorted = out_points[np.argsort(outer_angles)]

            #builds an arc around the outer and inner points for radial
            inner_arc = add_circular_arc(center=(0, 0), radius=r0, start_point=inner_sorted[0], end_point=inner_sorted[-1])
            outer_arc = add_circular_arc(center=(0, 0), radius=r1, start_point=outer_sorted[0], end_point=outer_sorted[-1])

            #find indices in sorted_shell corresponding to outer arc points
            def find_index(points, target): return np.argmin(np.linalg.norm(points - target, axis=1))
            i0 = find_index(edge_points, outer_sorted[0])
            i1 = find_index(edge_points, outer_sorted[-1])

            #splits the airfoil into sections to add arcs in proper order
            part1 = edge_points[:i0+1]
            part2 = edge_points[i1:]

            #sorts parts based on airfoil geometry
            part1 = part1[np.argsort(part1[:, 0])]    
            part2 = part2[np.argsort(-part2[:, 0])]  

            #builds the airfoil with all arcs and parts
            airfoil_shape = np.vstack([part1, outer_arc, part2, inner_arc[::-1]])

            #moves the foil so that the trialing edge is the first point
            te_index = np.argmax(airfoil_shape[:, 0])
            airfoil_shape = np.roll(airfoil_shape, -te_index, axis=0)

            #connects the airfoil if the shape is not closed
            if not (np.array_equal(airfoil_shape[0], airfoil_shape[-1])): airfoil_shape = np.vstack([airfoil_shape, airfoil_shape[0]])

            #finds the chord, chord length, and twist angle
            te_point = airfoil_shape[0]
            le_index = np.argmin(airfoil_shape[:, 0])
            le_point = airfoil_shape[le_index]
            chord = te_point - le_point
            chord_length = np.linalg.norm(chord)
            twist_angle = np.arctan2(chord[1], chord[0])
            twist_angle = np.degrees(twist_angle)

            #function that resamples the foil to have uniform spacing
            def uniform_resample(points, n_points=200):
                #removes duplicate closing point
                if np.allclose(points[0], points[-1]): points = points[:-1]

                #computes distances between points
                diffs = np.diff(points, axis=0)
                dists = np.linalg.norm(diffs, axis=1)
                arc_lengths = np.insert(np.cumsum(dists), 0, 0.0)

                #normalizes arc length and creates target
                total_length = arc_lengths[-1]
                arc_lengths /= total_length
                new_arcs = np.linspace(0, 1, n_points)

                #interpolates along the arc
                fx = interp1d(arc_lengths, points[:, 0], kind="linear")
                fy = interp1d(arc_lengths, points[:, 1], kind="linear")
                x_new = fx(new_arcs)
                y_new = fy(new_arcs)

                #closes loop and returns 
                resampled = np.column_stack([x_new, y_new])
                resampled = np.vstack([resampled, resampled[0]])
                return resampled
            airfoil_shape = uniform_resample(airfoil_shape)

            #saves the airfoil and returns the foil specifications
            filename = os.path.join(output_dir, f"{r_mid:.3f}_{n}.dat")
            np.savetxt(filename, airfoil_shape, fmt="%.6f",)
            return f"{r_mid:.3f}_{n}", twist_angle, chord_length    
        
        #function that helps process the airfoil shape: for now, only ordinary blade shapes can work full but support
        # for differnt sections is being added. Support for multiple continous normal sections is already present,
        # although full blade support for branched irregular blades is not yet available
        def process_section():
            #code segment to handle normal radial elements for current support based on normal cluster count
            if n_clusters_inner == n_clusters_outer == 2 or i == 0 or i == N-1:
                radius, twist_angle, chord_length = continous_foil(inner_points, outer_points, sorted_shell, 0)
                radii.append(radius)
                twist_angles.append(twist_angle)
                chord_lengths.append(chord_length)

            #code segment to handle radial with multiple foils. Although not currently supported fully, architecture for future expansion
            elif n_clusters_inner == n_clusters_outer != 2 and n_clusters_outer % 2 == 0:
                #finds the number of foils in the sectoon
                m = n_clusters_outer // 2
                inner_clusters = split_by_gaps(inner_points, distance_threshold=0.0005)
                outer_clusters = split_by_gaps(outer_points, distance_threshold=0.0005)

                #sorts based on angle
                def cluster_angle(cluster): return np.mean(np.arctan2(cluster[:, 1], cluster[:, 0]))
                inner_clusters.sort(key=cluster_angle)
                outer_clusters.sort(key=cluster_angle)

                #splits the shell points based on the number of expected jumps
                shell_regions = split_by_discontinuities(sorted_shell, m*2)
                shell_regions.sort(key=lambda seg: np.mean(np.arctan2(seg[:, 1], seg[:, 0])))

                #builds the concurrent foils by treating them as individual continous foils on the same radial
                for j in range(m):
                    in_cluster = np.vstack([inner_clusters[2*j], inner_clusters[(2*j)+1]])
                    out_cluster = np.vstack([outer_clusters[2*j], outer_clusters[(2*j+1)]])
                    sub_shell = np.vstack([shell_regions[2*j], shell_regions[2*j+1]])

                    radius, twist_angle, chord_length = continous_foil(in_cluster[::-1], out_cluster[::-1], sub_shell, j)
                    radii.append(radius)
                    twist_angles.append(twist_angle)
                    chord_lengths.append(chord_length)
            
            #print statement for geometry not currently supported
            else:
                print(f"Geometry currently not supported, skipping section at {r_mid}")
            
            """
            Optional code implementation for working on foils for nonstandard blade shapes (spirals, branches)
            elif n_clusters_outer > n_clusters_inner and n_clusters_outer % 2 == 0 and n_clusters_inner % 2 == 0:
                m = n_clusters_outer // 2
                inner_clusters = split_by_gaps(inner_points, distance_threshold=0.0005)
                outer_clusters = split_by_gaps(outer_points, distance_threshold=0.0005)
                matched = (match_inner_outer_by_jumps_clusters(inner_clusters, outer_clusters))
                for i, (inner, outer, outer_raw) in enumerate(matched):
                    inner_cluster_count = 2  # always 2 clusters combined
                    outer_cluster_count = len(outer_raw)
                    print(f"Foil {i+1}: {inner_cluster_count} inner clusters, {outer_cluster_count} outer clusters")
            """

        #calls the function to process the airfoil shape
        process_section()

    #returns the values that need to be used 
    return radii, twist_angles, chord_lengths

#creates the labels that display what the user has displaying
labels = []
label_names = ["Blade Design: ", "RPM: ", "Min Wind Speed (m/s): ", "Max Wind Speed (m/s): ", "Wind Speed Divisions: ", "Min Wind Angle(°): ", "Max Wind Angle(°): ", "Wind Angles Divisions: ", "Air Density(kg/m^2): ", "Air Viscosity (Pa*s): ", "Speed of Sound (m/s): ", "Blade Count: "]
#function that actually creates the labels
def create_labels():
    #loads the 12 labels in two columns
    for i, label in enumerate(label_names):
        labels.append(Label(text=label, font=("Times New Roman", 22), fg="deepskyblue4", bg="skyblue1", justify="left", anchor="w"))
        row = 2 * (i % 6) + 2      
        column = 0 if i < 6 else 7
        labels[i].grid(columnspan=8, row = row, column = column, padx=5)

#function that changes the label for whether the blade is draen
def update_blade_label():
    if valid:
        labels[0].config(text=f"{label_names[0]}Custom User Design")

#function that changes the labels for the simulation scope
def update_settings_label():
    for i in range(1, len(labels)):
        labels[i].config(text=f"{label_names[i]}{final_parameters[i-1]}")

#function that clears the entry text upon clicking
def clear(event, entry, text):
    if entry.get() == text:
        entry.delete(0, END)
    entry.config(fg = "deepskyblue4")

#function that restores the placeholder text if empty
def restore(event, entry, text):
    if not entry.get():
        entry.insert(0, text)
    entry.config(fg = "deepskyblue4")

# functions to change the simulation parameters
def change_settings():
    #creates the pop up window and configures it with a frame
    top = Toplevel(root, bg="skyblue1")
    top.title("Simulation Settings")
    top.geometry("600x600")
    top.resizable(False, False)
    frame = Frame(top, bg="skyblue1")
    for i in range(18): frame.grid_columnconfigure(i, weight=1); frame.grid_rowconfigure(i, weight=1)
    frame.pack(fill="both", expand=True)

    #creates the title label for the central body settings
    title_label = Label(frame, text="Simulation Settings", bg="skyblue1", fg="deepskyblue4", font=("Times New Roman", 24, "bold"), anchor="center", justify="center")
    title_label.grid(row=0, column=0, columnspan=18, sticky="nsew")

    #creates the secondary label for the central body settings
    secondary_label = Label(frame, text="Click to Change Windmill Simulation Settings", bg="skyblue1", fg="deepskyblue4", font=("Times New Roman", 18), anchor="center", justify="center")
    secondary_label.grid(row=1, column=0, columnspan=18, sticky="nsew")

    #function to create the labels and entry fields for the central body settings
    def labeled_entry(frame, row, label_text, default_value):
        label = Label(frame, text=label_text, bg="skyblue1", fg="deepskyblue4", font=("Times New Roman", 18), anchor="e", justify="right")
        label.grid(row=row, column=0, columnspan=9, sticky="nsew")
        
        entry = Entry(frame, bg="skyblue1", fg="deepskyblue4", font=("Times New Roman", 18), highlightthickness=0, relief="flat", insertbackground="#00688b")
        entry.insert(0, str(default_value))
        entry.grid(row=row, column=9, columnspan=9, sticky="nsew")
        entry.bind("<FocusIn>", lambda e, ent = entry, txt = str(default_value): clear(e, ent, txt))
        entry.bind("<FocusOut>", lambda e, ent = entry, txt = str(default_value): restore(e, ent, txt))
        
        return entry
    
    #creates the entries and buttons
    entries = []
    for i, name in enumerate(label_names):
        if i == 0:
            continue
        entries.append(labeled_entry(frame, i+3, name, final_parameters[i-1]))

    #function to save the changes and close the window
    def save_changes():
        #updates the global variables with the new values
        try:
            #checks to makle sure that none of the values have any errors
            rpm = float(entries[0].get()); min_wind = float(entries[1].get())
            max_wind = float(entries[2].get()); wind_space = int(entries[3].get())
            min_ang = float(entries[4].get()); max_ang = float(entries[5].get())
            ang_space = int(entries[6].get()); den = float(entries[7].get())
            vis = float(entries[8].get()); sos = float(entries[9].get())
            blade_count = int(entries[10].get())
            
            #checks for issues in wind parameters
            if(min_wind >= max_wind or min_wind < 0 or wind_space < 2):
                secondary_label.config(text="Please check wind paramters")
                return
            
            #checks for issues in angle parameters
            if(min_ang >= max_ang or ang_space < 2 or max_ang > 360 or min_ang < 0):
                secondary_label.config(text="Please check angle paramters")
                return          

            #checks for issues in air settings
            if(vis <= 0 or den <= 0 or sos <= 0):
                secondary_label.config(text="Please check air paramters")
                return             
            
            #checks for issues in blade count
            if(blade_count < 1):
                secondary_label.config(text="Please check blade count")
                return

            #updates the variables and calls function to update labels
            global final_parameters, valid_par
            final_parameters = [rpm, min_wind, max_wind, wind_space, min_ang, max_ang, ang_space, den, vis, sos, blade_count]
            update_settings_label()
            valid_par = True
            top.destroy()
    
        except ValueError:
            #if the values are not floats, it will display an error message
            secondary_label.config(text="Please enter valid numerical values")

    #function that sets the default values according to sea level
    def set_default():
        entries[7].delete(0, END)
        entries[7].insert(0, str(1.225))

        entries[8].delete(0, END)
        entries[8].insert(0, str(1.7894e-5))

        entries[9].delete(0, END)
        entries[9].insert(0, 340.294)

    #creates the button to use default (close to ground) values
    def_button = Button(frame, text="Sea Level", bg="skyblue1", fg="deepskyblue4", font=("Times New Roman", 20), command=lambda: set_default(), relief="flat", highlightthickness=0, bd=0)
    def_button.grid(row=18, column=0, columnspan=9, sticky="nsew")

    #creates the button to save the changes and close the window
    save_button = Button(frame, text="Save Changes", bg="skyblue1", fg="deepskyblue4", font=("Times New Roman", 20), command=lambda: save_changes(), relief="flat", highlightthickness=0, bd=0)
    
    save_button.grid(row=18, column=9, columnspan=9, sticky="nsew")

#function that creates the panel for the user to draw blade
def creation_panel():
    #creates and configures the top level
    creation_top = Toplevel(root)
    creation_top.title("Creation Panel")
    creation_top.geometry("1000x800")
    creation_top.config(bg="skyblue1")
    creation_top.resizable(False, False)

    #creates a frame to hold the drawing area and buttons
    creation_frame = Frame(creation_top)
    for i in range(160):
        creation_frame.grid_rowconfigure(i, weight=1)
        creation_frame.grid_columnconfigure(i, weight=1)
    creation_frame.config(bg="skyblue1")
    creation_frame.pack(fill=BOTH, expand=True)

    #holds the points clicked by the user
    points = []
    points.append((0, 0))

    #creates the matplotlib figure and axes for drawing
    fig, ax = plt.subplots(figsize=(5, 3))

    #function to define the drawing area
    def define_area():
        #changes the face color
        ax.set_facecolor("#b4deff")

        #defines the range of the drawing
        ax.set_xlim(0, 5)
        ax.set_ylim(-2, 2)

        #sets the tickets of the drawing
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        ax.grid(True, which="major", linestyle="-", linewidth=0.7, color="#63bcfc") 
        ax.grid(True, which="minor", linestyle="--", linewidth=0.7, color="#87ceff")

        #sets the titles and the labels for the axis
        ax.set_title("Click to add points in order to windmill blade design", fontsize=21, color="#03526d", fontname="Times New Roman")
        ax.set_xlabel("Horizontal distance from mount (m)", fontsize=17, color="#03526d", fontname="Times New Roman")
        ax.set_ylabel("Vertical distance from mount (m)", fontsize=17, color="#03526d", fontname="Times New Roman")

        #changes the colors of the other elements of the area
        ax.axhline(0, color="#03526d", linewidth=2)
        fig.patch.set_facecolor("#87ceff")

        #configures the spine of the area
        for spine in ax.spines.values(): spine.set_color("#03526d"); spine.set_linewidth(2)
        ax.tick_params(axis="both", colors="#03526d", length=8, width=2)
        ax.tick_params(axis="x", labelcolor="#03526d", labelsize=13)
        ax.tick_params(axis="y", labelcolor="#03526d", labelsize=13)

        #adds 0,0 as a drawing point always
        ax.plot(0, 0, "o", color="#016080")     
    
    #creates a canvas to embed the matplotlib figure in the Tkinter window
    define_area()
    canvas = FigureCanvasTkAgg(fig, master=creation_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=146, columnspan=160, sticky="nsew")

    #function to handle mouse clicks and store points
    def on_click(event):
        if event.inaxes:
            #places a point at the user defined location
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, "o", color="#016080") 
            canvas.draw()

    #function to connect points and draw the airfoil
    def connect_points():
        #returns an error if there are less than 2 drawn points
        if len(points) < 3:
            ax.set_title("Please draw at least 2 points", fontsize=21, color="#03526d", fontname="Times New Roman")
            canvas.draw()
            return
        
        #adds 0,0 at the end and then turns the points into a numpy array
        points.append((0, 0)) 
        pts = np.array(points)

        # Plot lines connecting the points in order
        ax.plot(pts[:, 0], pts[:, 1], "-", color="#1e7d9c")

        #resets title after error message
        ax.set_title("Click to add points in order to windmill blade design", fontsize=21, color="#03526d", fontname="Times New Roman")
        canvas.draw()

        #confirms that it is valid
        global valid
        valid = True

    # function to clear the drawing area
    def clear_drawing():
        #resets the points
        points.clear()
        points.append((0, 0)) 

        #says that is no longer valid
        global valid
        valid = False

        #resets the drawing
        ax.clear()
        define_area()
        canvas.draw()

    def confirm():
        #confirms if the drawing is valid (drawn)
        global valid
        if valid:
            #if valid, makes the final point the current points then closes top level
            global final_points 
            final_points = points
            creation_top.destroy()

            #updates the main display label
            update_blade_label()
        else:
            #if not it displays an error message
            ax.set_title("Please draw a valid design or connect points", fontsize=21, color="#03526d", fontname="Times New Roman")
            canvas.draw()
            return

    #draws previous points so user can adjust from saved drawing, checking if its not 0
    global final_points
    if len(final_points) > 0:
        points = final_points
        for point in final_points:
            ax.plot(point[0], point[1], "o", color="#016080" )
        connect_points()

    #button to connect points
    draw_btn = Button(creation_frame, text="Connect Points", command=connect_points, bg="skyblue1", fg="deepskyblue4", font=("Times New Roman", 26), height=1, relief="flat", highlightthickness=0, bd=0)
    draw_btn.grid(row=160, column=0, columnspan=40, ipady=0, ipadx=0, sticky="new")

    #button to clear points
    clear_btn = Button(creation_frame, text="Clear", command=clear_drawing, bg="skyblue1",fg="deepskyblue4", font=("Times New Roman", 26), height=1, relief="flat", highlightthickness=0, bd=0)
    clear_btn.grid(row=160, column=40, columnspan=40, ipady=0, ipadx=0, sticky="new")

    #button to confirm points
    confirm_button = Button(creation_frame, text="Confirm", command=confirm, bg="skyblue1",fg="deepskyblue4", font=("Times New Roman", 26), height=1, relief="flat", highlightthickness=0, bd=0)
    confirm_button.grid(row=160, column=80, columnspan=40, ipady=0, ipadx=0, sticky="new")

    #button to exit without saving
    exit_button = Button(creation_frame, text="Exit", command=creation_top.destroy, bg="skyblue1",fg="deepskyblue4", font=("Times New Roman", 26), height=1, relief="flat", highlightthickness=0, bd=0)
    exit_button.grid(row=160, column=120, columnspan=40, ipady=0, ipadx=0, sticky="new")

    #connects the canvas with the users pressing to draw points
    canvas.mpl_connect("button_press_event", on_click)

#function that confirms that the inputs are valid and runs calculations
def calculations():
    #chekcs if the current inputs are valid and displays a message if so
    if not valid or not valid_par:
        main_label.config(text="Please enter valid inputs and blade design")
        root.update()
        root.after(3000, lambda: main_label.config(text="Windmill Blade Design"))
        return

    #splits the points and then calls the function to create the .dat files
    global final_points
    final_points = densify_points(final_points, 0.00001)
    radii, twist_angles, chord_lengths = export_radials(final_points)

    #removes every widget except the main label
    for widget in root.winfo_children():
        if widget != main_label and widget != image_label:
            widget.grid_remove()
            widget.destroy()
    main_label.grid(row=6)
    image_label.grid(row=1)
    main_label.config(text="Processing Blade Design")
    root.update()

    #creates a progress bar in the root window
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
    progress_bar.grid(row=7, column=0, columnspan=16, pady=30)
    style = ttk.Style()
    style.theme_use("default")
    style.configure("custom.Horizontal.TProgressbar", troughcolor="#b4e0ff", bordercolor="#76c6ff", background="deepskyblue4", lightcolor="deepskyblue4", darkcolor="deepskyblue4", thickness=100)
    progress_bar.config(style="custom.Horizontal.TProgressbar")
    progress_bar["value"] = 0
    root.update()

    #callback to update the progress bar from bem.py for precomputation
    def update_precompute_progress(val, total):
        progress_bar["maximum"] = total
        progress_bar["value"] = val
        main_label.config(text=f"Precomputing Polars: {val}/{total}")
        root.update()

    #callback to update the progress bar from bem.py for BEM wind speed loop
    def update_bem_progress(val, total):
        progress_bar["maximum"] = total
        progress_bar["value"] = val
        main_label.config(text=f"Running Bem Calculations: {val}/{total}")
        root.update()

    #calls the bem solver from the bem file, passing the progress callbacks
    results = {}
    try: 
        results = bem_solver(radii, twist_angles, chord_lengths, final_parameters, update_precompute_progress, update_bem_progress)
    except Exception as e:
        progress_bar.grid_remove()
        main_label.config(text=f"There was an error during calculations\n{e}", font=("Times New Roman", 25))

    #updates the main label and progress bar
    main_label.config(text="Processing Results")
    root.update()
    progress_bar["value"] = 0
    progress_bar["maximum"] = 20
    root.update()

    #processes the results as arrays
    wind_speeds = sorted(set(v for v, a in results))
    angles = sorted(set(a for v, a in results))
    thrust = np.array([[results[(v, a)]["thrust"] for a in angles] for v in wind_speeds])
    torque = np.array([[results[(v, a)]["torque"] for a in angles] for v in wind_speeds])
    power = np.array([[results[(v, a)]["power"] for a in angles] for v in wind_speeds])
    cp = np.array([[results[(v, a)]["Cp"] for a in angles] for v in wind_speeds])

    #defines the plots folder
    shutil.rmtree("plots")
    os.makedirs("plots", exist_ok=True)

    #stores the filepaths of all images
    file_paths = []

    #updates the progress bar for the visuals
    plot_count = 0
    #calls the heatmap creation function from the visuals file
    file_paths.append(heatmap(thrust, angles, wind_speeds, "Thrust Distribution (N)", "Wind Direction (°)", "Wind Speed (m/s)", "thrust_heatmap.png"))
    plot_count += 1
    progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()
    file_paths.append(heatmap(torque, angles, wind_speeds, "Torque Distribution (N·m)", "Wind Direction (°)", "Wind Speed (m/s)", "torque_heatmap.png"))
    plot_count += 1
    progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()
    file_paths.append(heatmap(power, angles, wind_speeds, "Power Distribution (W)", "Wind Direction (°)", "Wind Speed (m/s)", "power_heatmap.png"))
    plot_count += 1
    progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()
    file_paths.append(heatmap(cp, angles, wind_speeds, "Cp Distribution", "Wind Direction (°)", "Wind Speed (m/s)", "cp_heatmap.png"))
    plot_count += 1
    progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()

    #calls the polar plots creation function from the visuals file
    for fp in polar_plot(results, wind_speeds, angles, "Cp"):
        file_paths.append(fp)
        plot_count += 1
        progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()
    for fp in polar_plot(results, wind_speeds, angles, "power"):
        file_paths.append(fp)
        plot_count += 1
        progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()
    for fp in polar_plot(results, wind_speeds, angles, "torque"):
        file_paths.append(fp)
        plot_count += 1
        progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()
    for fp in polar_plot(results, wind_speeds, angles, "thrust"):
        file_paths.append(fp)
        plot_count += 1
        progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()

    #calls the function to plot the cp vs tip speed ratio from the visuals file
    file_paths.append(plot_cp_vs_tsr(results, final_parameters, radii))
    plot_count += 1
    progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()

    #calls the counter map creation function from the visuals file
    file_paths.append(contour_plot(results, angles, wind_speeds, "Cp Distribution", "Cp", "cp_contour_map.png"))
    plot_count += 1
    progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()
    file_paths.append(contour_plot(results, angles, wind_speeds, "Power Distribution (W)", "power", "power_contour_map.png"))
    plot_count += 1
    progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()
    file_paths.append(contour_plot(results, angles, wind_speeds, "Torque Distribution (N·m)", "torque", "thrust_contour_map.png"))
    plot_count += 1
    progress_bar["value"] = plot_count; main_label.config(text=f"Creating Plots: {plot_count}/20"); root.update()

    # Remove the progress bar after all plots are created
    progress_bar.grid_remove()
    main_label.config(text="BEM Calculation Results")
    main_label.grid(row=0)

    #function that opens an image in a new top-level window
    def open_image_window(filepath):
        #creates a new top-level window for the image
        top = Toplevel(root)
        top.title(os.path.basename(filepath))
        top.config(bg="#b4deff")
        top.geometry("650x650")
        top.resizable(False, False)

        #loads the image and resizes it to fit
        image = Image.open(filepath)
        max_size = (630, 630)  # leave some padding
        image.thumbnail(max_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label = Label(top, image=photo, bg="#b4deff")
        label.image = photo
        label.pack(padx=10, pady=10, expand=True)

    #function to export the results
    def export_simulation_results():
        #converts the results dictionary to a dataframe
        df = pd.DataFrame.from_dict(results, orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Wind Speed (m/s)", "Wind Angle (deg)"])
        df.reset_index(inplace=True)

        #asks user for file path
        file_path = filedialog.asksaveasfilename(title="Export Simulation Data", defaultextension=".csv", filetypes=[("CSV File", "*.csv"), ("Text File", "*.txt")], initialfile="simulation_results")
        if not file_path: return

        #aves to filepat and shows an error message if unsuccesful
        try:
            if file_path.endswith(".txt"): df.to_csv(file_path, sep='\t', index=False)
            else: df.to_csv(file_path, index=False)
            main_label.config(text=f"Data successfully saved to {file_path}")
        except Exception as e:
            main_label.config(text=f"An error occurred during export:\n{e}")
        root.update()
        root.after(3000, lambda: main_label.config(text="BEM Calculation Results"))

    #function to export all the graphs as a zip file
    def export_graphs_as_zip():
        #asks user for zip file path
        zip_path = filedialog.asksaveasfilename(title="Export Graphs", defaultextension=".zip", filetypes=[("ZIP File", "*.zip")], initialfile="graphs_export")
        
        if not zip_path: return
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for png_file in Path("plots").glob("*.png"):
                    zipf.write(png_file, arcname=png_file.name)
            main_label.config(text=f"Graphs successfully zipped to {zip_path}")
        except Exception as e: main_label.config(text=f"Error while zipping: {e}")
        root.update()
        root.after(3000, lambda: main_label.config(text="BEM Calculation Results"))

    #creates buttons for each result image file
    for idx, filepath in enumerate(file_paths):
        #creates a button for each image file
        btn = Button(root, text=os.path.basename(filepath).replace("_", " ").replace(".png", "").title(), command=lambda fp=filepath: open_image_window(fp), bg="#87ceff", fg="deepskyblue4", font=("Times New Roman", 15, "bold"), relief="flat", activebackground="#b4e0ff", activeforeground="deepskyblue4", highlightthickness=0, bd=0, cursor="hand2")
        btn.grid(row= 4 + 2*(idx // 4), column=(idx % 4) * 4, columnspan=4, sticky="nsew", padx=5, pady=5)

    #creates the button to export the results as a file
    export_button = Button(root, text="Export Calculation Results", command=export_simulation_results, bg="#87ceff", fg="deepskyblue4", font=("Times New Roman", 18, "bold"), relief="flat", activebackground="#b4e0ff", activeforeground="deepskyblue4", highlightthickness=0, bd=0, cursor="hand2")
    export_button.grid(row=16, column=0, columnspan=6, sticky="nsew", padx=5, pady=10)

    #creates the button to export the graphs as a zip file
    zip_button = Button(root, text="Export Graphs", command=export_graphs_as_zip, bg="#87ceff", fg="deepskyblue4", font=("Times New Roman", 18, "bold"), relief="flat", activebackground="#b4e0ff", activeforeground="deepskyblue4", highlightthickness=0, bd=0, cursor="hand2")
    zip_button.grid(row=16, column=6, columnspan=5, sticky="nsew", padx=5, pady=10)

    #creates the button to restart the simulation
    restart_button = Button(root, text="Restart Simulation", command=restart_program, bg="#87ceff", fg="deepskyblue4", font=("Times New Roman", 18, "bold"), relief="flat", activebackground="#b4e0ff", activeforeground="deepskyblue4", highlightthickness=0, bd=0, cursor="hand2")
    restart_button.grid(row=16, column=11, columnspan=5, sticky="nsew", padx=5, pady=10)

#places button that opens the drawing panel on the grid
creation_button = Button(text="Create Blade", command=creation_panel)
creation_button.grid(row=15, column=0, columnspan=5, sticky="sew")
creation_button.config(font=("Times New Roman", 28), fg="deepskyblue4", bg="#87ceff", relief="flat", highlightthickness=0, bd=0)

#places button that opens the drawing panel on the grid
config_button = Button(text="Simulation Settings", command=change_settings)
config_button.grid(row=15, column=5, columnspan=6, sticky="sew")
config_button.config(font=("Times New Roman", 28), fg="deepskyblue4", bg="#87ceff", relief="flat", highlightthickness=0, bd=0)

#places button that starts the program
start_button = Button(text="Begin Calculations", command=calculations)
start_button.grid(row=15, column=11, columnspan=5, sticky="sew")
start_button.config(font=("Times New Roman", 28), fg="deepskyblue4", bg="#87ceff", relief="flat", highlightthickness=0, bd=0)

#calls the function to display the labels
create_labels()

#runs the program
if __name__ == "__main__":
    root.mainloop()












