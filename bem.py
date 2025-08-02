import neuralfoil as nf
import numpy as np
import multiprocessing

#function that handles the precomputation of coefficents vs aoa for different reynolds numbers (polars) of different radials
def precompute_polars(args):
    dat_file, Re_min, Re_max, Re_step, alpha_min, alpha_max, alpha_steps, model_size = args
    
    #sets up the variabls needed to call NueralFoil to calculate coefficietns
    alphas = np.linspace(alpha_min, alpha_max, alpha_steps)
    polars = {}
    dat_file_path = f"radial_files/{dat_file}.dat"
    radius = float(dat_file.split("_")[0])

    #calculates the polars for each radial according to parameters
    for Re in np.arange(Re_min, Re_max + Re_step, Re_step):
        res = nf.get_aero_from_dat_file(filename=dat_file_path, alpha=alphas, Re=Re, model_size=model_size)
        polars[int(Re)] = {"alpha": alphas, "Cl": res["CL"], "Cd": res["CD"]}
    return radius, polars

#function that uses parralel programming to call the function to precompute polars
def parallel_precompute_polars(files, Re_min=5000, Re_max=1_500_000, Re_step=5000, alpha_min=-180, alpha_max=180, alpha_steps=361, model_size="large"):
    args_list = [(file, Re_min, Re_max, Re_step, alpha_min, alpha_max, alpha_steps, model_size) for file in files]

    n_cores = max(multiprocessing.cpu_count()-2, 1)

    #if a progress callback is provided is updates the progress bar
    progress_callback = None
    import inspect
    frame = inspect.currentframe()
    outer = frame.f_back
    if "update_precompute_progress" in outer.f_locals:
        progress_callback = outer.f_locals["update_precompute_progress"]

    #uses parralel programming to call the polar precomputation and update the progress bar
    results = []
    with multiprocessing.Pool(n_cores) as pool:
        for i, res in enumerate(pool.imap(precompute_polars, args_list)):
            results.append(res)
            if progress_callback:
                progress_callback(i+1, len(args_list))

    polars_dict = {radius: polar_data for radius, polar_data in results}
    return polars_dict

#function that performs the BEM calculations for a wind angle and radial according to a wind speed
def bem_loop(args):
    V_inf, wind_angles, omega, files, twist_angles, chord_lengths, rho, mu, blade_count, polars_dict, max_iter, epsilon = args

    local_results = {}
    #loops according to wind angles
    for theta in wind_angles:
        total_power = 0
        total_thrust = 0
        total_torque = 0

        #loops according to the file and its parameters
        for file, twist, chord in zip(files, twist_angles, chord_lengths):
            #sets up radius and sigma value for calculation
            r = float(file.split("_")[0])
            sigma = blade_count * chord / (2 * np.pi * r)

            #initial guesses for BEM convergence loop
            a, a_prime = 0.3, 0.01
            converged = False

            for _ in range(max_iter):
                #calculates components of velocity based on current induction factors
                V_axial = V_inf * np.cos(theta) * (1 - a)
                V_tan = omega * r * (1 + a_prime)
                V_rel = np.sqrt(V_axial**2 + V_tan**2)

                #uses velocity vectors to calculate angle of attack, factoring in twist angle
                phi = np.arctan2(V_axial, V_tan)
                alpha = np.degrees(phi) - twist

                #calculates the Reynolds numbre and rounds it to the nearest 5000
                Re = int((rho * V_rel * chord) / mu)
                Re = int(round(Re / 5000) * 5000)
                if r not in polars_dict or Re not in polars_dict[r]: break

                #gets the drag and lift coefficients according to interpolation of the polars
                polar = polars_dict[r][Re]
                Cl = np.interp(alpha, polar["alpha"], polar["Cl"])
                Cd = np.interp(alpha, polar["alpha"], polar["Cd"])

                #calculates normal and tangential force coefficients
                Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
                Ct = Cl * np.sin(phi) - Cd * np.cos(phi)

                #updates induction factors based on new coefficients
                a_new = 1 / (4 * np.sin(phi)**2 / (sigma * Cn) + 1)
                a_prime_new = 1 / (4 * np.sin(phi) * np.cos(phi) / (sigma * Ct) - 1)

                #checks loop for convergence
                if abs(a_new - a) < epsilon and abs(a_prime_new - a_prime) < epsilon:
                    a, a_prime = a_new, a_prime_new
                    converged = True
                    break

                #continues loop with newer induction factors
                a, a_prime = a_new, a_prime_new

            #skips iteration if unable to convergence
            if not converged: continue

            #calculates final components of velocity
            V_axial = V_inf * np.cos(theta) * (1 - a)
            V_tan = omega * r * (1 + a_prime)
            V_rel = np.sqrt(V_axial**2 + V_tan**2)
            phi = np.arctan2(V_axial, V_tan)

            #calculates the differential lift and drag
            dL = 0.5 * rho * V_rel**2 * chord * Cl
            dD = 0.5 * rho * V_rel**2 * chord * Cd

            #uses list and drag forces to calculate thrust and torque
            dT = dL * np.cos(phi) + dD * np.sin(phi)
            dQ = -r * (dL * np.sin(phi) - dD * np.cos(phi))

            #finds the average dr and uses it to update thrust and torque totally
            dr = (max(float(f.split("_")[0]) for f in files) - min(float(f.split("_")[0]) for f in files)) / len(files)
            total_thrust += dT * blade_count * dr
            total_torque += dQ * blade_count * dr

        #updates the local results for power, thurst, and torque
        total_power = total_torque * omega
        key = (round(V_inf, 3), round(np.degrees(theta), 1))
        local_results[key] = {"thrust": total_thrust,"torque": total_torque,"power": total_power}

    return local_results

#function that is used to solve the BEM from the main file
def bem_solver(files, twist_angles, chord_lengths, final_parameters, update_precompute_progress=None, update_bem_progress=None):
    rpm, min_wind, max_wind, wind_space, min_ang, max_ang, ang_space, rho, mu, sos, blade_count = final_parameters

    #splits wind speeds and angles and turns rpm into radians per second
    wind_speeds = np.linspace(min_wind, max_wind, wind_space + 1)
    wind_angles = np.radians(np.linspace(min_ang, max_ang, ang_space + 1))
    omega = rpm * 2 * np.pi / 60

    #get progress callbacks if provided
    import inspect
    frame = inspect.currentframe()
    outer = frame.f_back
    update_precompute_progress = outer.f_locals.get("update_precompute_progress", None)
    update_bem_progress = outer.f_locals.get("update_bem_progress", None)

    #precomputes the polars and stores it in a dictionary
    polars_dict = parallel_precompute_polars(files, Re_min=5000, Re_max=1_500_000, Re_step=5000, alpha_min=-180, alpha_max=180, alpha_steps=61, model_size="large")

    #saves the arguments needed for the calling of the bem loops
    args_list = [(V_inf, wind_angles, omega, files, twist_angles, chord_lengths, rho, mu, blade_count, polars_dict, 5000, 1e-5) for V_inf in wind_speeds]

    #defines the number of parralel operations and calls the bem loop 
    n_cores = max(multiprocessing.cpu_count()-2, 1)
    results_list = []
    with multiprocessing.Pool(n_cores) as pool:
        for i, res in enumerate(pool.imap(bem_loop, args_list)):
            results_list.append(res)
            if update_bem_progress:
                update_bem_progress(i+1, len(args_list))

    #merges all the results into one
    results = {}
    for res in results_list: results.update(res)

    #calculates the power coefficient for each setup based on blade radial area
    max_radius = max(float(file.split("_")[0]) for file in files)
    swept_area = np.pi * max_radius**2
    for (V_inf, theta_deg), data in results.items():
        if V_inf == 0:
            Cp = 0
        else:
            Cp = data["power"] / (0.5 * rho * V_inf**3 * swept_area)
        results[(V_inf, theta_deg)]["Cp"] = Cp

    return results