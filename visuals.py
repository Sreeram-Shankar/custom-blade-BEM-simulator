import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

#function to create a heatmap of different quantaties
def heatmap(data, x_labels, y_labels, title, xlab, ylab, filename, font="Times New Roman", bg="#b4deff", fg="#03526d"):
    #creates the filepath to save
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)

    #initializes heatmap
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    #plots the heatmap using imshow with smooth interpolation
    img = ax.imshow(data, cmap="viridis", interpolation="bilinear", aspect="auto")

    #sets background color
    plt.gcf().patch.set_facecolor(bg)

    #sets title and labels of the heatmap
    ax.set_title(title, fontsize=16, color=fg, fontname=font)
    ax.set_xlabel(xlab, fontsize=14, color=fg, fontname=font)
    ax.set_ylabel(ylab, fontsize=14, color=fg, fontname=font)

    #sets ticks and tick labels of the heatmap
    n_ticks = 10
    x_indices = np.linspace(0, len(x_labels) - 1, n_ticks+1, dtype=int)
    y_indices = np.linspace(0, len(y_labels) - 1, n_ticks+1, dtype=int)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([x_labels[i] for i in x_indices], color=fg, fontsize=10)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([y_labels[i] for i in y_indices], color=fg, fontsize=10)
    ax.invert_yaxis()
    ax.tick_params(colors=fg)

    #adds colorbar for visualization
    cbar = plt.colorbar(img)
    cbar.ax.tick_params(labelcolor=fg)

    #saves the figure and returns the file path
    plt.tight_layout()
    plt.savefig(filepath, facecolor=bg, dpi=300)
    plt.close()
    return filepath

#function to create a polar plot of a given quantity
def polar_plot(results, wind_speeds, angles, quantity, font="Times New Roman", bg="#b4deff", fg="#03526d"):
    #gets 3 wind speeds, thne min, mid, and max
    wind_speeds_to_plot = [wind_speeds[0], wind_speeds[len(wind_speeds)//2], wind_speeds[-1]]
    labels = ["min", "mid", "max"]

    #creates the plot for all three speeds
    res = []
    for i, v in enumerate(wind_speeds_to_plot):
        #converts radians to degrees and gets the values
        angles_rad = np.radians(angles)
        values = [results[(v, a)][quantity] for a in angles]

        #creates the polar plot and plots it
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles_rad, values, color=fg, linewidth=2)

        #configures the plot
        ax.set_facecolor(bg)
        ax.set_title(f"{quantity.capitalize()} vs Wind Angle at {v:.1f} m/s", fontsize=16, color=fg, fontname=font, pad=20)
        ax.tick_params(colors=fg, labelsize=10)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.grid(True, color=fg, linestyle="--", linewidth=0.6)
        ax.spines['polar'].set_color(fg)
        ax.spines['polar'].set_linewidth(1.5)

        #sets up the filename and path
        filename = f"{labels[i]}_{quantity.lower()}_polar.png"
        filepath = os.path.join("plots", filename)

        #saves the plot and returns the filename
        plt.tight_layout()
        plt.savefig(filepath, facecolor=bg, dpi=300)
        plt.close()
        res.append(filepath)
    return res

#function to compute and plot Cp vs tip speed ratio
def plot_cp_vs_tsr(results, final_parameters, files, font="Times New Roman", bg="#b4deff", fg="#03526d"):
    #gets the paramaters and the highest radius
    rpm = final_parameters[0]
    omega = 2 * np.pi * rpm / 60
    R = max(float(file.split("_")[0]) for file in files)

    #groups results by wind speed
    wind_speeds = sorted(set(v for v, a in results if v > 0))

    #calculates the mean Cp and saves it 
    tsrs = []
    mean_cps = []
    for V in wind_speeds:
        tsr = omega * R / V
        cp_vals = [results[(V, a)]["Cp"] for a in sorted(set(a for vv, a in results if vv == V))]
        mean_cp = np.mean(cp_vals)
        tsrs.append(tsr)
        mean_cps.append(mean_cp)

    #creates and configures the plot of the cvr to the 
    plt.figure(figsize=(7, 5))
    plt.plot(tsrs, mean_cps, marker="o", color=fg, linewidth=2)
    plt.xlabel("Tip-Speed Ratio (TSR)", fontsize=13, color=fg, fontname=font)
    plt.ylabel("Mean Power Coefficient (Cp)", fontsize=13, color=fg, fontname=font)
    plt.title("Cp vs TSR", fontsize=16, color=fg, fontname=font, pad=12)
    plt.grid(True, linestyle="--", color=fg, alpha=0.5)
    plt.xticks(color=fg); plt.yticks(color=fg)
    plt.gca().spines['bottom'].set_color(fg); plt.gca().spines['left'].set_color(fg)
    plt.gca().spines['bottom'].set_linewidth(1.5); plt.gca().spines['left'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig("plots/cp_vs_tsr.png", facecolor=bg, dpi=300)
    plt.close()

    #returns the filepath
    return "plots/cp_vs_tsr.png"

#function to create a contour map of differnt quantaties
def contour_plot(results, angles, wind_speeds, title, quantity, filename, font="Times New Roman", bg="#b4deff", fg="#03526d"):
    #sets up the grid
    X, Y = np.meshgrid(angles, wind_speeds)
    Z = np.array([[results.get((v, a), {}).get(quantity, 0) for a in angles] for v in wind_speeds])

    #creates the plot of the quantity
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, Z, levels=30, cmap="cividis")
    plt.contour(X, Y, Z, levels=10, colors="black", linewidths=0.5)

    #adds the colorbar
    cbar = plt.colorbar(contour)
    cbar.ax.tick_params(labelsize=14, colors=fg)

    #configures the plot style
    plt.title(title, fontsize=16, color=fg, fontname=font, pad=15)
    plt.xlabel("Wind Direction (°)", fontsize=14, color=fg, fontname=font)
    plt.ylabel("Wind Speed (m/s)", fontsize=14, color=fg, fontname=font)
    plt.xticks(color=fg); plt.yticks(color=fg)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gca().spines["bottom"].set_color(fg); plt.gca().spines["left"].set_color(fg)
    plt.gca().spines["bottom"].set_linewidth(1.5); plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().set_facecolor(bg)

    #saves and returns the path
    plt.tight_layout()
    filepath = os.path.join("plots", filename)
    plt.savefig(filepath, facecolor=bg, dpi=300)
    plt.close()
    return filepath
